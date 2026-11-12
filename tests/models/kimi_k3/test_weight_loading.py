# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
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
    # for the auxiliary tensor, but the projection itself is present, so the
    # loader must not index params_dict with the stale name.
    stale_auxiliary = torch.empty(1, dtype=torch.bfloat16)
    model._parameters["layers.0.self_attn.v_proj.weight"] = torch.nn.Parameter(
        torch.empty(4)
    )
    loaded = model.load_weights(
        [("layers.0.self_attn.v_proj.weight_offset", stale_auxiliary)]
    )

    assert loaded == set()


def test_unknown_projection_weight_still_fails():
    model = object.__new__(KimiLinearModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        linear_attn_config={},
        is_moe=False,
        q_lora_rank=None,
        is_linear_attn=True,
        num_nextn_predict_layers=0,
    )

    with pytest.raises(KeyError):
        model.load_weights(
            [("layers.0.self_attn.unknown_projection.weight", torch.empty(1))]
        )


@pytest.mark.parametrize(
    "auxiliary_suffix",
    [".weight_offset", ".weight_scale", ".scale_bias"],
)
def test_stale_fused_projection_auxiliary_is_skipped(auxiliary_suffix):
    model = object.__new__(KimiLinearModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        linear_attn_config={},
        is_moe=False,
        q_lora_rank=None,
        is_linear_attn=True,
        num_nextn_predict_layers=0,
    )
    model._parameters["layers.0.self_attn.in_proj_qkvgfab.weight"] = torch.nn.Parameter(
        torch.empty(4)
    )

    loaded = model.load_weights(
        [(f"layers.0.self_attn.v_proj{auxiliary_suffix}", torch.empty(1))]
    )

    assert loaded == set()


def test_unknown_offset_still_fails():
    model = object.__new__(KimiLinearModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        linear_attn_config={},
        is_moe=False,
        q_lora_rank=None,
        is_linear_attn=True,
        num_nextn_predict_layers=0,
    )

    with pytest.raises(KeyError):
        model.load_weights(
            [("layers.0.self_attn.unknown_projection.weight_offset", torch.empty(1))]
        )


def test_bound_offset_is_loaded():
    model = object.__new__(KimiLinearModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        linear_attn_config={},
        is_moe=False,
        q_lora_rank=None,
        is_linear_attn=True,
        num_nextn_predict_layers=0,
    )
    model._parameters["layers.0.self_attn.in_proj_qkvgfab.weight"] = (
        torch.nn.Parameter(torch.empty(4))
    )
    model._parameters["layers.0.self_attn.v_proj.weight_offset"] = (
        torch.nn.Parameter(torch.empty(4))
    )

    loaded = model.load_weights(
        [("layers.0.self_attn.v_proj.weight_offset", torch.ones(4))]
    )

    assert loaded == {"layers.0.self_attn.v_proj.weight_offset"}
