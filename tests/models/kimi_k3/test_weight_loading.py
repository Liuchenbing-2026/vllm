# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.models.kimi_k3.amd.linear import KimiLinearModel


def test_unbound_quantization_auxiliary_weight_is_skipped():
    model = object.__new__(KimiLinearModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        linear_attn_config={},
        is_moe=False,
        q_lora_rank=None,
        is_linear_attn=True,
        num_nextn_predict_layers=0,
    )

    # Kimi checkpoints can retain a quantization-only tensor after a
    # projection has been converted back to FLOAT. No parameter is allocated
    # for that tensor, so the loader must not index params_dict with it.
    stale_auxiliary = torch.empty(1, dtype=torch.bfloat16)
    loaded = model.load_weights(
        [("layers.0.self_attn.v_proj.weight_offset", stale_auxiliary)]
    )

    assert loaded == set()
