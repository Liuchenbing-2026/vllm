# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for tree seeding; model/device work is simulated explicitly."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm.v1.worker.gpu.spec_decode.autoregressive.speculator as ar
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers


@pytest.mark.parametrize("full_graph", [False, True])
@pytest.mark.parametrize("auxiliary", [False, True])
def test_tree_seed_runs_one_prefill_without_decode_or_mutating_graph_width(
    monkeypatch, full_graph, auxiliary
):
    batch = InputBatch.make_dummy(1, 2, InputBuffers(1, 2, torch.device("cpu")))
    batch.seq_lens_cpu_upper_bound.fill_(5)
    events = []
    desc = SimpleNamespace(
        num_tokens=2,
        cg_mode=CUDAGraphMode.FULL if full_graph else CUDAGraphMode.NONE,
    )
    combine = Mock(side_effect=lambda x: x[:, :3])
    dispatch = Mock(return_value=(desc, None))
    monkeypatch.setattr(ar, "dispatch_cg_and_sync_dp", dispatch)
    monkeypatch.setattr(
        ar, "prepare_prefill_inputs", lambda *a: events.append("inputs")
    )
    decode = Mock(side_effect=AssertionError("Tree seeding must not decode AR steps"))
    monkeypatch.setattr(ar, "prepare_decode_inputs", decode)
    spec = SimpleNamespace(
        method="eagle4",
        num_speculative_steps=8,
        max_model_len=64,
        model=SimpleNamespace(combine_hidden_states=combine),
        _copy_request_inputs=Mock(),
        last_token_indices=None,
        current_draft_step=None,
        input_buffers=None,
        max_num_reqs=1,
        pcp_manager=None,
        hidden_states=torch.empty(2, 3),
        prefill_cudagraph_manager=SimpleNamespace(
            run_fullgraph=lambda _: events.append("graph")
        ),
        dp_size=1,
        dp_rank=0,
        _prepare_eplb_forward=Mock(),
        on_prefill_begin=lambda _: events.append("begin"),
        on_prefill_end=lambda _: events.append("end"),
        _prefill=lambda *a, **kw: events.append("prefill"),
        draft_tokens=torch.arange(8).view(1, 8),
    )
    hidden = torch.ones(2, 3)
    aux = [hidden * 2, hidden * 3] if auxiliary else None
    result = ar.AutoRegressiveSpeculator.propose(
        spec,
        batch,
        {},
        {},
        hidden,
        aux,
        None,
        None,
        None,
        None,
        None,
        None,
        num_speculative_tokens=1,
    )
    assert result.tolist() == [[0]]
    assert events == ["inputs", "begin", "graph" if full_graph else "prefill", "end"]
    assert spec.num_speculative_steps == 8
    assert spec.draft_max_seq_len == 6
    assert torch.equal(spec.hidden_states, hidden * (2 if auxiliary else 1))
    assert combine.call_count == int(auxiliary)
    assert dispatch.call_count == 1
    decode.assert_not_called()


@pytest.mark.parametrize("width", [0, 2, 9, -1])
def test_unsupported_width_fails_before_touching_batch_or_state(width):
    spec = SimpleNamespace(num_speculative_steps=8)
    with pytest.raises(ValueError, match="configured draft width"):
        ar.AutoRegressiveSpeculator.propose(
            spec,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            num_speculative_tokens=width,
        )
