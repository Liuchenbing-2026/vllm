# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU dispatch tests; substituted kernels do not certify device arithmetic."""

from types import SimpleNamespace

import torch

import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as gdn_layer_module
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.verify_extension import (
    GDNSpecRows,
    LinearAttnVerifyExtension,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata


class RecordingExtension(LinearAttnVerifyExtension):
    """Writes ``fill`` to every spec row's output and records what it got."""

    def __init__(self, vllm_config=None, fill: float = 3.0):
        self.vllm_config = vllm_config
        self.fill = fill
        self.rows: list[GDNSpecRows] = []

    def forward_spec_rows(self, rows):
        self.rows.append(rows)
        rows.out.fill_(self.fill)


class _Calls:
    recurrent_a: torch.Tensor  # the gates the fake recurrence received

    def __init__(self):
        self.conv = 0
        self.recurrent = 0


def _layer(monkeypatch, calls: _Calls, metadata) -> QwenGatedDeltaNetAttention:
    def fake_conv(mixed_qkv_spec, *args, **kwargs):
        calls.conv += 1
        return mixed_qkv_spec + 100

    def fake_recurrent(**kwargs):
        calls.recurrent += 1
        calls.recurrent_a = kwargs["a"]
        return torch.full((1, kwargs["q"].shape[0], 2, 3), -1.0), None

    monkeypatch.setattr(gdn_layer_module, "causal_conv1d_update", fake_conv)
    monkeypatch.setattr(
        gdn_layer_module, "fused_sigmoid_gating_delta_rule_update", fake_recurrent
    )
    monkeypatch.setattr(gdn_layer_module, "is_conv_state_dim_first", lambda: True)
    monkeypatch.setattr(
        gdn_layer_module,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"layer": metadata}),
    )
    layer = QwenGatedDeltaNetAttention.__new__(QwenGatedDeltaNetAttention)
    torch.nn.Module.__init__(layer)
    layer.prefix = "layer"
    layer.enable_packed_recurrent_decode = False
    layer.activation = "silu"
    layer.conv1d = SimpleNamespace(weight=torch.zeros(5, 1, 4), bias=None)
    layer.kv_cache = (torch.zeros(2, 5, 3), torch.zeros(2, 2, 3, 3))
    layer.A_log = torch.zeros(2)
    layer.dt_bias = torch.ones(2)
    layer.tp_size = 1
    layer.num_k_heads, layer.num_v_heads, layer.head_k_dim, layer.head_v_dim = (
        1,
        2,
        2,
        3,
    )
    layer.rearrange_mixed_qkv = lambda x: (None, None, None) if x is None else (x, x, x)
    return layer


def _spec_metadata(extension) -> GDNAttentionMetadata:
    return GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=1,
        num_spec_decode_tokens=4,
        num_actual_tokens=4,
        spec_query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        spec_state_indices_tensor=torch.zeros(1, 4, dtype=torch.int32),
        spec_sequence_masks=torch.ones(1, dtype=torch.bool),
        num_accepted_tokens=torch.ones(1, dtype=torch.int32),
        verify_extension=extension,
    )


def _run_layer(monkeypatch, extension, padded: int = 0):
    """A spec-only batch of 4 rows; ``b`` / ``a`` are strided views of one
    projection and the buffers have ``padded`` rows past the batch."""
    metadata = _spec_metadata(extension)
    calls = _Calls()
    layer = _layer(monkeypatch, calls, metadata)
    out = torch.zeros(4 + padded, 2, 3)
    b, a = torch.zeros(4 + padded, 4).chunk(2, dim=-1)
    layer._forward_core(
        mixed_qkv=torch.zeros(4 + padded, 5), b=b, a=a, core_attn_out=out
    )
    return calls, out, a, metadata


def test_layer_runs_the_in_tree_kernels_without_an_extension(monkeypatch):
    calls, out, _, _ = _run_layer(monkeypatch, None)
    assert (calls.conv, calls.recurrent) == (1, 1)
    assert torch.all(out == -1.0)
    assert calls.recurrent_a.is_contiguous()


def test_the_extension_replaces_the_spec_kernels(monkeypatch):
    extension = RecordingExtension()
    calls, out, a, metadata = _run_layer(monkeypatch, extension, padded=2)
    assert (calls.conv, calls.recurrent) == (0, 0)
    assert torch.all(out[:4] == 3.0) and torch.all(out[4:] == 0)
    (rows,) = extension.rows
    # The projections and the output buffer arrive as the views they are, cut to
    # the batch: no copies.
    assert rows.a.data_ptr() == a.data_ptr() and not rows.a.is_contiguous()
    assert rows.mixed_qkv.shape[0] == 4 and rows.out.shape[0] == 4
    assert rows.layer_name == "layer"
    assert (rows.num_k_heads, rows.num_v_heads, rows.head_k_dim, rows.head_v_dim) == (
        1,
        2,
        2,
        3,
    )
    assert rows.state_indices is metadata.spec_state_indices_tensor
    assert rows.num_accepted_tokens is metadata.num_accepted_tokens
    assert rows.query_start_loc is metadata.spec_query_start_loc


def test_the_fused_norm_entry_passes_the_gate_views_through(monkeypatch):
    extension = RecordingExtension()
    metadata = _spec_metadata(extension)
    layer = _layer(monkeypatch, _Calls(), metadata)
    monkeypatch.setattr(layer, "_rms_norm_gated_cuda", lambda *args: None)
    b, a = torch.zeros(4, 4).chunk(2, dim=-1)
    out = torch.zeros(4, 2, 3)
    layer._forward_core_fused_norm(torch.zeros(4, 5), b, a, torch.zeros_like(out), out)
    (rows,) = extension.rows
    assert rows.a.data_ptr() == a.data_ptr() and not rows.a.is_contiguous()


def test_the_extension_gets_the_spec_rows_of_a_mixed_batch(monkeypatch):
    """Spec rows next to plain decode rows: the extension gets the spec rows
    only, the decode rows run the in-tree kernels, and both outputs land in
    their rows."""
    extension = RecordingExtension(fill=7.0)
    metadata = _spec_metadata(extension)
    calls = _Calls()
    layer = _layer(monkeypatch, calls, metadata)
    metadata.num_decodes = metadata.num_decode_tokens = 1
    metadata.num_actual_tokens = 4 + 1
    metadata.spec_token_indx = torch.arange(1, 4 + 1)
    metadata.non_spec_token_indx = torch.tensor([0])
    metadata.non_spec_state_indices_tensor = torch.tensor([1], dtype=torch.int32)
    metadata.non_spec_query_start_loc = torch.tensor([0, 1], dtype=torch.int32)
    out = torch.zeros(4 + 1, 2, 3)
    layer._forward_core(
        mixed_qkv=torch.zeros(4 + 1, 5),
        b=torch.zeros(4 + 1, 2),
        a=torch.zeros(4 + 1, 2),
        core_attn_out=out,
    )
    (rows,) = extension.rows
    assert rows.mixed_qkv.shape[0] == 4
    assert (calls.conv, calls.recurrent) == (1, 1)  # the decode row
    assert torch.all(out[0] == -1.0) and torch.all(out[1:] == 7.0)


def test_fused_mtp_decode_is_skipped_when_an_extension_is_present(monkeypatch):
    layer = QwenGatedDeltaNetAttention.__new__(QwenGatedDeltaNetAttention)
    torch.nn.Module.__init__(layer)
    state_dtype = next(iter(gdn_layer_module.FUSED_GDN_STATE_DTYPES))
    layer.kv_cache = (torch.zeros(1), torch.zeros(1, dtype=state_dtype))
    layer.gdn_decode_kernel = "cuda"
    layer.num_v_heads = layer.num_k_heads = 2
    monkeypatch.setattr(
        torch.ops._C, "fused_gdn_decode_post_conv_mtp", object(), raising=False
    )
    # Without an extension the fused MTP kernel is eligible ...
    assert layer._can_use_fused_gdn_mtp_decode(_spec_metadata(None)) is True
    # ... with one, the layer must go through the path that hands it the rows.
    with_extension = _spec_metadata(RecordingExtension())
    assert layer._can_use_fused_gdn_mtp_decode(with_extension) is False
