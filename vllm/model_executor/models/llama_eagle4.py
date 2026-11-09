# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""EAGLE-4 draft support for vLLM 0.30.

This implements the dense, single-layer EAGLE-4 chain draft. It is derived
from the public checkpoint contract and is intentionally independent of the
EAGLE-4 source-available runtime.
"""

from collections.abc import Callable, Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.model_executor.models.utils import maybe_prefix


def clear_identity_d2t(model: nn.Module) -> None:
    """Remove EAGLE-3's identity mapping parameter for a full-vocab draft."""
    config = model.config
    vocab_size = getattr(config, "vocab_size", None)
    draft_vocab_size = getattr(config, "draft_vocab_size", vocab_size)
    if draft_vocab_size != vocab_size:
        raise ValueError(
            "EAGLE-4 full-vocabulary drafts cannot use a d2t mapping: "
            f"draft_vocab_size={draft_vocab_size}, vocab_size={vocab_size}."
        )
    # Assigning None unregisters EAGLE-3's candidate Parameter, so strict
    # loader tracking no longer expects a mapping absent from the checkpoint.
    model.draft_id_to_target_id = None


def low_rank_scores(
    hidden_states: torch.Tensor,
    down_proj_weight: torch.Tensor,
    score_head_weight: torch.Tensor,
    local_vocab_rows: int,
) -> torch.Tensor:
    """Return one TP rank's approximate logits for the real vocabulary rows."""
    low_rank = hidden_states.float() @ down_proj_weight.float().t()
    scores = low_rank @ score_head_weight.float().t()
    if scores.shape[-1] > local_vocab_rows:
        scores[:, local_vocab_rows:] = float("-inf")
    return scores


def gather_dot_reference(
    weight: torch.Tensor,
    indices: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Exact dot products for a small set of gathered vocabulary rows."""
    rows = weight[indices.long()].float()
    return torch.einsum("bkh,bh->bk", rows, hidden_states.float())


def merge_tp_candidates(
    local_logits: torch.Tensor,
    local_global_ids: torch.Tensor,
    num_candidates: int,
    *,
    all_gather: Callable[[torch.Tensor], torch.Tensor],
    tp_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge rank-major candidates and keep the best ``num_candidates`` per row."""
    num_rows = local_logits.shape[0]
    gathered_logits = all_gather(local_logits.float()).view(
        tp_size, num_rows, num_candidates
    )
    gathered_ids = all_gather(local_global_ids.long()).view(
        tp_size, num_rows, num_candidates
    )
    candidate_logits = gathered_logits.permute(1, 0, 2).reshape(
        num_rows, tp_size * num_candidates
    )
    candidate_ids = gathered_ids.permute(1, 0, 2).reshape(
        num_rows, tp_size * num_candidates
    )
    top_logits, positions = torch.topk(
        candidate_logits,
        num_candidates,
        dim=-1,
        sorted=False,
    )
    return top_logits, torch.gather(candidate_ids, 1, positions)


class LlamaForCausalLMEagle4(Eagle3LlamaForCausalLM):
    """Dense EAGLE-4 chain draft with a low-rank candidate LM head."""

    has_own_lm_head = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.vllm_config = vllm_config
        # EAGLE-4 validation requires a full-vocabulary draft. EAGLE-3's
        # constructor can retain a trainable identity d2t parameter when the
        # target and draft configs are observed through different wrappers.
        # Clear it explicitly so strict weight tracking does not expect a
        # mapping that is absent from the EAGLE-4 checkpoint.
        clear_identity_d2t(self)

        head_config = self.config.efficient_lm_head_config
        self.num_candidates = int(head_config["num_candidates"])
        self.candidate_rank = int(head_config["rank"])
        if self.candidate_rank <= 0 or self.num_candidates <= 0:
            raise ValueError("EAGLE-4 low-rank head dimensions must be positive.")
        if self.model.quant_config is not None:
            raise NotImplementedError(
                "EAGLE-4's candidate head currently requires an unquantized draft."
            )

        self.efficient_lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.candidate_rank,
            prefix=maybe_prefix(prefix, "efficient_lm_head"),
        )
        self.efficient_lm_head_down_proj = ReplicatedLinear(
            self.config.hidden_size,
            self.candidate_rank,
            bias=False,
            params_dtype=vllm_config.model_config.dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "efficient_lm_head_down_proj"),
            return_bias=False,
        )
        self.tp_size = get_tensor_model_parallel_world_size()

    def _local_candidates(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shard = self.efficient_lm_head.shard_indices
        local_vocab_rows = shard.org_vocab_end_index - shard.org_vocab_start_index
        if local_vocab_rows < self.num_candidates:
            raise ValueError(
                "EAGLE-4 num_candidates exceeds a TP rank's real vocabulary "
                f"rows: {self.num_candidates} > {local_vocab_rows}."
            )
        low_rank = self.efficient_lm_head_down_proj(hidden_states)
        scores = low_rank.float() @ self.efficient_lm_head.weight.float().t()
        scores[:, local_vocab_rows:] = float("-inf")
        local_ids = torch.topk(
            scores, self.num_candidates, dim=-1, sorted=False
        ).indices
        local_logits = gather_dot_reference(
            self.lm_head.weight, local_ids, hidden_states
        )
        return local_logits, local_ids + shard.org_vocab_start_index

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        local_logits, local_ids = self._local_candidates(hidden_states)
        if self.tp_size == 1:
            logits, token_ids = local_logits, local_ids
        else:
            logits, token_ids = merge_tp_candidates(
                local_logits,
                local_ids,
                self.num_candidates,
                all_gather=lambda tensor: tensor_model_parallel_all_gather(
                    tensor, dim=0
                ),
                tp_size=self.tp_size,
            )

        logit_scale = float(getattr(self.config, "logit_scale", 1.0))
        logits = logits.float() * logit_scale
        full_logits = logits.new_full(
            (hidden_states.shape[0], self.config.vocab_size),
            float("-inf"),
        )
        return full_logits.scatter_(1, token_ids.long(), logits)

    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # One candidate-head pass is cheaper than scoring the full vocabulary.
        local_logits, local_ids = self._local_candidates(hidden_states)
        if self.tp_size > 1:
            local_logits, local_ids = merge_tp_candidates(
                local_logits,
                local_ids,
                self.num_candidates,
                all_gather=lambda tensor: tensor_model_parallel_all_gather(
                    tensor, dim=0
                ),
                tp_size=self.tp_size,
            )
        positions = local_logits.argmax(dim=-1, keepdim=True)
        return torch.gather(local_ids, 1, positions).squeeze(1)

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        # Reuse the EAGLE-3 checkpoint layout and loader. EAGLE-4 adds its
        # candidate-head tensors at the top level, which the parent loader maps
        # to the modules registered above.
        loaded = super().load_weights(weights)
        # EAGLE-4 public checkpoints ship RMSNorm weights without bias terms.
        # Some runtime norm implementations still register a bias parameter;
        # initialize those to zero (identity) instead of failing strict
        # weight tracking.
        for name, param in self.named_parameters():
            if name in loaded or not name.endswith(".bias"):
                continue
            with torch.no_grad():
                param.zero_()
            loaded.add(name)
        return loaded
