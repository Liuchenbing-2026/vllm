# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Perplexity Decider's Qwen backbone and independent decision readout."""

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.pooler import Pooler, PoolingParamsUpdate
from vllm.model_executor.layers.pooler.seqwise.methods import LastPool
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.tasks import PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata

from .interfaces_base import default_pooling_type
from .qwen3_5 import Qwen3_5ForConditionalGeneration, Qwen3_5ProcessingInfo
from .qwen3_vl import Qwen3VLDummyInputsBuilder, Qwen3VLMultiModalProcessor
from .utils import WeightsMapper


class DeciderPooler(Pooler):
    """Return raw decision logits; callers restrict candidates before calibration."""

    def __init__(self, hidden_size: int, num_labels: int, dtype: torch.dtype):
        super().__init__()
        self.method = LastPool()
        self.readout = nn.Linear(hidden_size, num_labels, bias=False, dtype=dtype)

    def get_supported_tasks(self) -> set[PoolingTask]:
        return {"classify"}

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate()

    def forward(
        self, hidden_states: torch.Tensor, pooling_metadata: PoolingMetadata
    ) -> torch.Tensor:
        selected = self.method(hidden_states, pooling_metadata)
        return self.readout(selected.to(self.readout.weight.dtype)).float()


@default_pooling_type(seq_pooling_type="LAST")
@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class PplxDeciderForSequenceClassification(Qwen3_5ForConditionalGeneration):
    """Native decision checkpoints exported with their small readout tensor."""

    is_pooling_model = True

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "language_model.": "language_model.model.",
            "readout.": "pooler.readout.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        if vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError("Decider sequence pooling requires PP=1")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # This checkpoint has a small readout, not a vocabulary output head.
        del self.language_model.lm_head
        self.pooler = DeciderPooler(
            vllm_config.model_config.get_hidden_size(),
            self.config.num_labels,
            vllm_config.model_config.dtype,
        )
