# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``ExtractHiddenStatesProposer._resolve_slot_mapping``.

The cache-only layers used by ``extract_hidden_states`` may belong to a KV
cache group whose block size differs from the group behind
``common_attn_metadata.slot_mapping``.  Writing the cache with the wrong block
size scatters the writes into unallocated blocks, and the read then returns
mostly zeros plus a few bf16-overflow NaN/Inf values.
"""

from types import SimpleNamespace

import torch

from vllm.v1.spec_decode.extract_hidden_states import ExtractHiddenStatesProposer


def _proposer(**kwargs) -> SimpleNamespace:
    defaults = {
        "attn_layer_names": ["layer.0", "layer.1"],
        "runner": None,
        "kv_cache_gid": -1,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_prefers_runner_per_layer_mapping():
    per_layer = torch.tensor([11, 12, 13])
    fallback = torch.tensor([99])
    attn_metadata = SimpleNamespace(slot_mapping=fallback)

    resolved = ExtractHiddenStatesProposer._resolve_slot_mapping(
        _proposer(), {"layer.0": per_layer, "layer.1": per_layer}, attn_metadata
    )

    assert resolved is per_layer


def test_falls_back_to_runner_block_table():
    block_table = SimpleNamespace(
        slot_mapping=SimpleNamespace(gpu=torch.tensor([1, 2, 3, 4]))
    )
    runner = SimpleNamespace(input_batch=SimpleNamespace(block_table=[block_table]))
    attn_metadata = SimpleNamespace(
        slot_mapping=torch.tensor([0, 0, 0, 0]), num_actual_tokens=2
    )

    resolved = ExtractHiddenStatesProposer._resolve_slot_mapping(
        _proposer(runner=runner, kv_cache_gid=0), None, attn_metadata
    )

    assert resolved.tolist() == [1, 2]


def test_falls_back_to_common_attn_metadata():
    fallback = torch.tensor([7, 8])
    attn_metadata = SimpleNamespace(slot_mapping=fallback)

    resolved = ExtractHiddenStatesProposer._resolve_slot_mapping(
        _proposer(), None, attn_metadata
    )

    assert resolved is fallback
