# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for `MinTokensLogitsProcessor` with vocab-parallel logits.

Regression tests for vllm-project/vllm#61966: the processor's index tensors
hold global token ids, so applying them to rank-local (vocab-sharded) logits
indexed past the local width and crashed the rank.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm import SamplingParams
from vllm.v1.sample.logits_processor import BatchUpdate
from vllm.v1.sample.logits_processor import builtin as builtin_mod
from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor

VOCAB_SIZE = 248320
TP_SIZE = 8
SHARD_WIDTH = VOCAB_SIZE // TP_SIZE  # 31040
# Global stop-token ids owned by rank 7 at tp_size=8.
STOP_TOKEN_IDS = [248044, 248046]


def _make_processor(
    monkeypatch,
    *,
    vocab_size: int = VOCAB_SIZE,
    tp_size: int = TP_SIZE,
    rank: int = 0,
    stop_token_ids: list[int] | None = None,
    structured_outputs: dict | None = None,
) -> MinTokensLogitsProcessor:
    monkeypatch.setattr(builtin_mod, "_tp_rank", lambda: rank)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(get_vocab_size=lambda: vocab_size),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp_size),
    )
    processor = MinTokensLogitsProcessor(
        vllm_config, torch.device("cpu"), is_pin_memory=False
    )
    params = SamplingParams(
        min_tokens=1,
        stop_token_ids=list(
            STOP_TOKEN_IDS if stop_token_ids is None else stop_token_ids
        ),
        structured_outputs=structured_outputs,
    )
    processor.update_state(
        BatchUpdate(
            batch_size=1,
            removed=(),
            added=((0, params, None, []),),
            moved=(),
        )
    )
    return processor


def _local_index(token_id: int, rank: int, shard_width: int = SHARD_WIDTH) -> int:
    return token_id - rank * shard_width


@pytest.mark.parametrize("rank", range(TP_SIZE))
def test_sharded_logits_mask_only_owned_stop_tokens(monkeypatch, rank):
    """Rank-local logits must not go out of bounds on any rank."""
    processor = _make_processor(monkeypatch, rank=rank)
    logits = torch.zeros(1, SHARD_WIDTH)

    applied = processor.apply(logits)

    if rank == STOP_TOKEN_IDS[0] // SHARD_WIDTH:
        for token_id in STOP_TOKEN_IDS:
            assert torch.isneginf(applied[0, _local_index(token_id, rank)])
    else:
        # Ids owned by another rank are simply not present in this shard.
        assert applied.isfinite().all()


def test_all_gathered_logits_keep_global_behaviour(monkeypatch):
    """The default all-gathered path still masks the global stop-token ids."""
    processor = _make_processor(monkeypatch, rank=0)
    logits = torch.zeros(1, VOCAB_SIZE)

    applied = processor.apply(logits)

    assert all(torch.isneginf(applied[0, token_id]) for token_id in STOP_TOKEN_IDS)


def test_all_gathered_logits_are_chosen_for_padded_vocab(monkeypatch):
    """A padded vocab must not be mistaken for a rank-local shard."""
    vocab_size, tp_size = 1010, 2
    # pad_vocab_size(1010, 64) == 1024 -> shard width 512.
    processor = _make_processor(
        monkeypatch,
        vocab_size=vocab_size,
        tp_size=tp_size,
        rank=1,
        stop_token_ids=[1009],
    )

    all_gathered = torch.zeros(1, 1024)
    assert torch.isneginf(processor.apply(all_gathered)[0, 1009])

    sharded = torch.zeros(1, 512)
    assert torch.isneginf(processor.apply(sharded)[0, 1009 - 512])


def test_single_rank_is_not_sharded(monkeypatch):
    """tp_size == 1 keeps the identity mapping regardless of padding."""
    processor = _make_processor(
        monkeypatch, vocab_size=1010, tp_size=1, rank=0, stop_token_ids=[1009]
    )

    applied = processor.apply(torch.zeros(1, 1010))

    assert torch.isneginf(applied[0, 1009])


def test_tp1_never_selects_shard_local_slices(monkeypatch):
    """Without vocab parallelism the index tensors are used verbatim."""
    processor = _make_processor(monkeypatch, tp_size=1, rank=0, stop_token_ids=[7, 8])

    for width in (16, VOCAB_SIZE):
        assert not processor._use_shard_local(torch.zeros(1, width))
        slices = processor._slices_for(torch.zeros(1, width))
        assert slices[0] is processor.logits_slice
        assert slices[1] is processor.restore_logits_slice


def test_spec_decode_sharded_logits(monkeypatch):
    """The spec-decode entry point translates ids the same way."""
    rank = STOP_TOKEN_IDS[0] // SHARD_WIDTH
    processor = _make_processor(monkeypatch, rank=rank)
    logits = torch.zeros(3, SHARD_WIDTH)

    applied = processor.apply_with_spec_decode(logits, num_draft_tokens=[3])

    # min_tokens=1 with 3 draft positions: only the first is still masked.
    assert all(
        torch.isneginf(applied[0, _local_index(token_id, rank)])
        for token_id in STOP_TOKEN_IDS
    )
    assert applied[1:].isfinite().all()


def test_restore_path_uses_shard_local_ids(monkeypatch):
    """Structured-output restore indexes the same coordinate system as masking."""
    rank = STOP_TOKEN_IDS[0] // SHARD_WIDTH
    processor = _make_processor(
        monkeypatch,
        rank=rank,
        structured_outputs={"choice": ["yes", "no"]},
    )
    local_column = _local_index(STOP_TOKEN_IDS[0], rank)
    logits = torch.full((1, SHARD_WIDTH), float("-inf"))
    logits[0, local_column] = 5.0

    applied = processor.apply(logits)

    # Masking emptied the row, so the stop token is restored to its value.
    assert applied[0, local_column] == 5.0
