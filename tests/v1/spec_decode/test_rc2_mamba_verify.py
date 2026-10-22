# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba2 CPU dispatch tests, with substituted device kernels."""

from types import SimpleNamespace

import torch

import vllm.model_executor.layers.mamba.mamba_mixer2 as mixer_module
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
from vllm.model_executor.layers.mamba.verify_extension import (
    LinearAttnVerifyExtension,
    Mamba2SpecRows,
)
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadata


class RecordingMamba2Extension(LinearAttnVerifyExtension):
    def __init__(self, fill=3.0):
        self.fill = fill
        self.rows = []

    def forward_mamba2_spec_rows(self, rows):
        self.rows.append(rows)
        rows.out.fill_(self.fill)


class _Calls:
    def __init__(self):
        self.conv = self.ssu = self.prefill_conv = self.prefill_scan = 0


def _layer(monkeypatch, calls: _Calls, metadata) -> MambaMixer2:
    def fake_conv(x, *args, **kwargs):
        calls.conv += 1
        return x

    def fake_ssu(*args, out, **kwargs):
        calls.ssu += 1
        out.fill_(-1.0)

    def fake_prefill_conv(x, *args, **kwargs):
        calls.prefill_conv += 1
        return x

    def fake_prefill_scan(x, *args, out, **kwargs):
        calls.prefill_scan += 1
        out.fill_(-2.0)
        return torch.zeros(1, 2, 3, 4)

    monkeypatch.setattr(mixer_module, "causal_conv1d_update", fake_conv)
    monkeypatch.setattr(mixer_module, "selective_state_update", fake_ssu)
    monkeypatch.setattr(mixer_module, "causal_conv1d_fn", fake_prefill_conv)
    monkeypatch.setattr(
        mixer_module, "mamba_chunk_scan_combined_varlen", fake_prefill_scan
    )
    monkeypatch.setattr(mixer_module, "is_conv_state_dim_first", lambda: True)
    monkeypatch.setattr(
        mixer_module,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"layer": metadata}),
    )
    layer = MambaMixer2.__new__(MambaMixer2)
    torch.nn.Module.__init__(layer)
    layer.prefix = "layer"
    layer.cache_config = SimpleNamespace(mamba_cache_mode="none", mamba_block_size=16)
    layer.num_spec = 4 - 1
    layer.use_replayssm = False
    layer.mamba_config = SimpleNamespace(
        enable_stochastic_rounding=False, stochastic_rounding_philox_rounds=0
    )
    layer.tp_size = 1
    layer.num_heads, layer.head_dim = 2, 3
    layer.n_groups, layer.ssm_state_size = 1, 4
    layer.tped_intermediate_size = 6
    layer.tped_conv_size, layer.tped_dt_size = 14, 2
    layer.activation = "silu"
    layer.conv1d = SimpleNamespace(bias=None)
    layer.conv_weights = torch.zeros(14, 4)
    layer.kv_cache = (
        torch.zeros(3, 14, 3 + 4 - 1),
        torch.zeros(3, 2, 3, 4),
    )
    layer.A, layer.D, layer.dt_bias = (
        -torch.ones(2),
        torch.ones(2),
        torch.ones(2),
    )
    layer.is_blackwell = False
    layer.split_hidden_states_B_C_fn = lambda t: torch.split(
        t, [6, 1 * 4, 1 * 4], dim=-1
    )
    return layer


def _metadata(extension, num_prefill_tokens: int = 0) -> Mamba2AttentionMetadata:
    has_prefill = num_prefill_tokens > 0
    return Mamba2AttentionMetadata(
        num_prefills=int(has_prefill),
        num_prefill_tokens=num_prefill_tokens,
        num_decodes=1,
        num_decode_tokens=4,
        num_reqs=1 + int(has_prefill),
        has_initial_states_p=None,
        num_computed_tokens_p=None,
        block_idx_last_scheduled_token=None,
        block_idx_first_scheduled_token_p=None,
        block_idx_last_computed_token=None,
        block_idx_last_scheduled_token_prev_step=None,
        query_start_loc_p=(
            torch.tensor([0, num_prefill_tokens], dtype=torch.int32)
            if has_prefill
            else None
        ),
        state_indices_tensor_p=(
            torch.tensor([2], dtype=torch.int32) if has_prefill else None
        ),
        state_indices_tensor_d=torch.tensor([[1, 1, 1, 1]], dtype=torch.int32),
        query_start_loc_d=torch.tensor([0, 4], dtype=torch.int32),
        num_accepted_tokens=torch.ones(1, dtype=torch.int32),
        seq_lens=torch.tensor([40], dtype=torch.int32),
        verify_extension=extension,
    )


def _run(monkeypatch, extension, num_prefill_tokens: int = 0, padded: int = 2):
    metadata = _metadata(extension, num_prefill_tokens)
    calls = _Calls()
    layer = _layer(monkeypatch, calls, metadata)
    tokens = 4 + num_prefill_tokens + padded
    projected = torch.zeros(tokens, 6 + 14 + 2)
    out = torch.zeros(tokens, 6)
    layer.conv_ssm_forward(projected, out)
    return calls, out, projected, metadata


def test_layer_runs_the_in_tree_kernels_without_an_extension(monkeypatch):
    calls, out, _, _ = _run(monkeypatch, None)
    assert (calls.conv, calls.ssu) == (1, 1)
    assert torch.all(out[:4] == -1.0)


def test_the_extension_replaces_the_decode_kernels(monkeypatch):
    extension = RecordingMamba2Extension()
    calls, out, projected, metadata = _run(monkeypatch, extension)
    assert (calls.conv, calls.ssu) == (0, 0)
    assert torch.all(out[:4] == 3.0) and torch.all(out[4:] == 0)
    (rows,) = extension.rows
    assert isinstance(rows, Mamba2SpecRows)
    # The projections and the output arrive as views cut to the decode rows.
    assert rows.hidden_states_B_C.shape == (4, 14)
    assert rows.hidden_states_B_C.data_ptr() == projected[:, 6:].data_ptr()
    assert rows.dt.shape == (4, 2) and not rows.dt.is_contiguous()
    assert rows.out.shape == (4, 6)
    assert rows.out.data_ptr() == out.data_ptr()
    assert (rows.num_heads, rows.head_dim, rows.n_groups, rows.ssm_state_size) == (
        2,
        3,
        1,
        4,
    )
    assert rows.layer_name == "layer"
    assert rows.state_indices is metadata.state_indices_tensor_d
    assert rows.num_accepted_tokens is metadata.num_accepted_tokens
    assert rows.query_start_loc is metadata.query_start_loc_d


def test_the_extension_gets_only_the_decode_rows_of_a_mixed_batch(monkeypatch):
    extension = RecordingMamba2Extension(fill=7.0)
    calls, out, _, _ = _run(monkeypatch, extension, num_prefill_tokens=5, padded=0)
    (rows,) = extension.rows
    assert isinstance(rows, Mamba2SpecRows)
    assert rows.hidden_states_B_C.shape[0] == 4
    assert (calls.conv, calls.ssu) == (0, 0)
    assert (calls.prefill_conv, calls.prefill_scan) == (1, 1)
    assert torch.all(out[:4] == 7.0) and torch.all(out[4:] == -2.0)
