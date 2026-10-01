# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-pass structured decisions using the trained Qwen3.5 LM head."""

from collections.abc import Callable

import torch

from vllm.config import VllmConfig
from vllm.model_executor.layers.pooler import Pooler, PoolingParamsUpdate
from vllm.model_executor.layers.pooler.tokwise.methods import AllPool
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.tasks import PoolingTask
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.pool.metadata import PoolingMetadata

from .interfaces_base import default_pooling_type
from .qwen3_5 import Qwen3_5ForConditionalGeneration, Qwen3_5ProcessingInfo
from .qwen3_vl import Qwen3VLDummyInputsBuilder, Qwen3VLMultiModalProcessor


class InternDecisionPooler(Pooler):
    """Return raw symbol logits in field order, before candidate normalization.

    A marker's preceding token predicts its answer. Markers can span prefill
    chunks, so selection happens after AllPool has assembled each request.
    """

    def __init__(
        self,
        compute_logits: Callable[[torch.Tensor], torch.Tensor | None],
        marker_id: int,
        symbol_ids: list[int],
    ):
        super().__init__()
        self.method = AllPool()
        self.compute_logits = compute_logits
        self.marker_id = marker_id
        self.symbol_ids = symbol_ids

    def get_supported_tasks(self) -> set[PoolingTask]:
        return {"token_classify"}

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate(requires_token_ids=True)

    def forward(
        self, hidden_states: torch.Tensor, pooling_metadata: PoolingMetadata
    ) -> list[torch.Tensor | None]:
        sequences = self.method(hidden_states, pooling_metadata)
        token_ids = pooling_metadata.get_prompt_token_ids_cpu()
        outputs: list[torch.Tensor | None] = []
        for sequence, ids in zip(sequences, token_ids):
            if sequence is None:
                outputs.append(None)
                continue
            positions = (ids == self.marker_id).nonzero(as_tuple=True)[0] - 1
            if (positions < 0).any():
                raise ValueError("A decision marker must have a preceding token")
            if positions.numel() == 0:
                # Dummy profiling prompts need not contain decision markers.
                outputs.append(sequence.new_empty((0, len(self.symbol_ids))))
                continue
            selected = sequence[async_tensor_h2d(positions, sequence.device)]
            logits = self.compute_logits(selected)
            assert logits is not None
            outputs.append(logits[:, self.symbol_ids].float())
        return outputs


@default_pooling_type(tok_pooling_type="ALL")
@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class InternDecisionForTokenClassification(Qwen3_5ForConditionalGeneration):
    """Native token-classification runner for Intern-Decision checkpoints.

    Output columns follow ABC...XYZabc...xyz012...789. Apply softmax only
    over each field's candidates, with the desired calibration temperature.
    The official prompt compiler supplies the complete assistant skeleton.
    """

    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        parallel = vllm_config.parallel_config
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
            raise ValueError("Intern-Decision currently requires TP=1 and PP=1")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        marker = "<decision>"
        if marker not in self._tokenizer.get_added_vocab():
            raise ValueError("Intern-Decision requires its trained decision tokenizer")
        symbols = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        encoded = [
            self._tokenizer.encode(symbol, add_special_tokens=False)
            for symbol in symbols
        ]
        if any(len(ids) != 1 for ids in encoded):
            raise ValueError("Decision candidate symbols must be single tokens")
        self.pooler = InternDecisionPooler(
            self.language_model.compute_logits,
            self._tokenizer.convert_tokens_to_ids(marker),
            [ids[0] for ids in encoded],
        )
