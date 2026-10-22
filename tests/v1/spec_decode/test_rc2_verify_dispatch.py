# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise rc2 runner hooks using CPU tensors and simulated device operations."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm.v1.worker.gpu.model_runner as runner_module
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator


class MinimalSpeculator(BaseSpeculator):
    def init_cudagraph_manager(self, cudagraph_mode):
        pass

    def capture(self):
        pass

    def propose(self, *args, **kwargs):
        raise AssertionError("Not used by the forward-only test")


def test_default_hooks_preserve_chain_behavior():
    spec = MinimalSpeculator()
    assert spec.overridden_hooks() == set()
    assert spec.make_rejection_sampler(object()) is None
    assert spec.prepare_verify_inputs(None, {}) is None
    assert spec.post_accept(None, None, None, None) is None


@pytest.mark.parametrize("custom", [False, True])
def test_sampler_selection_keeps_fallback_arguments_and_checks_watermarks(
    monkeypatch, custom
):
    fallback = Mock()
    monkeypatch.setattr(runner_module, "RejectionSampler", fallback)
    monkeypatch.setattr(
        runner_module, "speculative_target_watermark_key", lambda _: "key"
    )
    selected = object() if custom else None
    spec = SimpleNamespace(make_rejection_sampler=Mock(return_value=selected))
    runner = SimpleNamespace(
        sampler=object(),
        speculative_config=object(),
        speculator=spec,
        device="cpu",
        vllm_config=SimpleNamespace(
            watermark_config=None, _check_watermarking_unsupported=Mock()
        ),
    )
    result = GPUModelRunner._build_rejection_sampler(runner)
    spec.make_rejection_sampler.assert_called_once_with(runner.sampler)
    if custom:
        assert result is selected
        fallback.assert_not_called()
        runner.vllm_config._check_watermarking_unsupported.assert_called_once_with(
            custom_sampler=True
        )
    else:
        assert result is fallback.return_value
        fallback.assert_called_once_with(
            runner.sampler, runner.speculative_config, "cpu", watermark_key="key"
        )


@pytest.mark.parametrize("full_graph", [False, True])
@pytest.mark.parametrize("dummy_run", [False, True])
@pytest.mark.parametrize("skip_attn", [False, True])
def test_verify_positions_are_staged_before_forward_or_replay(
    monkeypatch, full_graph, dummy_run, skip_attn
):
    buffers = InputBuffers(2, 4, torch.device("cpu"))
    batch = InputBatch.make_dummy(2, 4, buffers)
    batch.req_ids = ["a", "b"]
    scheduled = SchedulerOutput.make_empty()
    scheduled.num_scheduled_tokens = {"a": 2, "b": 2}
    scheduled.total_num_scheduled_tokens = 4
    desc = SimpleNamespace(
        num_tokens=4,
        num_reqs=2,
        max_query_len=2,
        cg_mode=CUDAGraphMode.FULL if full_graph else CUDAGraphMode.NONE,
        num_ubatches=1,
        num_active_loras=0,
    )
    events = []

    def verify(input_batch, model_inputs, *, dummy_run=False):
        events.append(("verify", dummy_run))
        assert model_inputs["positions"].data_ptr() == input_batch.positions.data_ptr()
        model_inputs["positions"].fill_(7)

    expected_positions = buffers.positions[:4].clone()
    if not (dummy_run and skip_attn):
        expected_positions.fill_(7)

    def forward(**kwargs):
        events.append(("forward", dummy_run))
        assert torch.equal(buffers.positions[:4], expected_positions)
        return torch.zeros(4, 3)

    runner = SimpleNamespace(
        update_pp_decode_requests=lambda: None,
        finish_requests=lambda _: None,
        free_states=lambda _: None,
        add_requests=lambda _: None,
        update_requests=lambda _: None,
        block_tables=Mock(),
        input_buffers=buffers,
        cudagraph_manager=SimpleNamespace(run_fullgraph=lambda _: forward()),
        gather_batch_req_state=lambda *a: (
            SimpleNamespace(num_tokens=4, has_prefill=False),
            2,
        ),
        pcp_manager=None,
        lora_config=None,
        is_encoder_decoder=False,
        dp_size=1,
        dp_rank=0,
        dcp_size=1,
        dcp_rank=0,
        cp_interleave=1,
        parallel_config=object(),
        ubatch_runner=None,
        decode_query_len=2,
        observability_config=SimpleNamespace(cudagraph_metrics=False),
        prepare_inputs=lambda *a: batch,
        prepare_attn=lambda _: ([], []),
        prepare_dummy_attn=lambda *a: ([], []),
        model_state=SimpleNamespace(
            preprocess_state=lambda *a: None,
            prepare_attn=lambda *a, **kw: {},
            prepare_inputs=lambda *a: {},
        ),
        kv_cache_config=object(),
        attn_groups=[],
        req_states=SimpleNamespace(num_computed_tokens=SimpleNamespace(gpu=None)),
        uses_inputs_embeds=False,
        is_first_pp_rank=True,
        is_last_pp_rank=True,
        speculator=SimpleNamespace(prepare_verify_inputs=verify),
        speculator_hooks=frozenset({"prepare_verify_inputs"}),
        eplb=Mock(),
        step_timing=Mock(),
        kv_connector=Mock(),
        vllm_config=object(),
        model_config=object(),
        model=forward,
        use_aux_hidden_state_outputs=False,
        routed_experts_capturer=None,
    )
    monkeypatch.setattr(
        runner_module, "dispatch_cg_and_sync_dp", lambda *a, **kw: (desc, None)
    )
    monkeypatch.setattr(runner_module, "build_slot_mappings_by_layer", lambda *a: {})
    monkeypatch.setattr(
        runner_module, "set_forward_context", lambda *a, **kw: nullcontext()
    )
    if dummy_run and skip_attn and full_graph:
        with pytest.raises(AssertionError, match="Attention metadata must be prepared"):
            GPUModelRunner.execute_model(
                runner, scheduled, dummy_run=True, skip_attn_for_dummy_run=True
            )
        assert events == []
        return
    GPUModelRunner.execute_model(
        runner, scheduled, dummy_run=dummy_run, skip_attn_for_dummy_run=skip_attn
    )
    expected_events = [("forward", dummy_run)]
    if not (dummy_run and skip_attn):
        expected_events.insert(0, ("verify", dummy_run))
    assert events == expected_events
    assert runner.execute_model_state.hidden_states.shape == (4, 3)


def test_post_accept_runs_before_state_commit_and_next_draft(monkeypatch):
    batch = InputBatch.make_dummy(1, 2, InputBuffers(1, 2, torch.device("cpu")))
    batch.req_ids = ["a"]
    events = []
    output = SimpleNamespace(sampled_token_ids=torch.ones(1, 1, dtype=torch.int32))
    counts = (torch.ones(1, dtype=torch.int32), torch.ones(1, dtype=torch.int32))

    def accept(b, sampled, num_sampled, num_rejected):
        assert b is batch and sampled is output
        assert num_sampled is counts[0] and num_rejected is counts[1]
        events.append("accept")

    def propose(*args, **kwargs):
        events.append("propose")
        return torch.ones(1, 1, dtype=torch.int32)

    runner = SimpleNamespace(
        execute_model_state=SimpleNamespace(
            input_batch=batch,
            attn_metadata={},
            slot_mappings_by_layer={},
            hidden_states=torch.zeros(2, 4),
            aux_hidden_states=None,
            dp_sync=None,
            finished_req_ids=set(),
            ec_connector_output=None,
            routed_experts=None,
            cudagraph_stats=None,
        ),
        is_last_pp_rank=True,
        pcp_manager=None,
        pp_handler=None,
        speculator=SimpleNamespace(
            supports_mm_inputs=False, post_accept=accept, propose=propose
        ),
        speculator_hooks=frozenset({"post_accept"}),
        model=SimpleNamespace(compute_logits=lambda x: x),
        sample=lambda *a: (output, *counts),
        prompt_logprobs_worker=SimpleNamespace(compute_prompt_logprobs=lambda *a: {}),
        req_states=SimpleNamespace(
            all_token_ids=SimpleNamespace(gpu=None),
            num_computed_tokens=SimpleNamespace(gpu=None),
            prompt_len=SimpleNamespace(np=None),
            draft_tokens=torch.zeros(1, 1, dtype=torch.int32),
            last_sampled_tokens=None,
            next_prefill_tokens=None,
        ),
        main_stream=None,
        output_copy_stream=None,
        check_ep_fault=False,
        kv_connector=Mock(),
        postprocess_sampled=lambda *a: events.append("commit"),
        sampler=SimpleNamespace(
            sampling_states=SimpleNamespace(
                temperature=SimpleNamespace(gpu=None), seeds=SimpleNamespace(gpu=None)
            )
        ),
        vllm_config=SimpleNamespace(watermark_config=None),
        num_speculative_steps=1,
        draft_tokens_handler=Mock(),
        eplb=Mock(),
        adaptive_verification=None,
        _draft_workspace_lane=0,
    )
    monkeypatch.setattr(runner_module, "AsyncOutput", lambda **kw: kw)
    monkeypatch.setattr(runner_module, "use_workspace_lane", lambda _: nullcontext())
    GPUModelRunner.sample_tokens(runner, None)
    assert events == ["accept", "commit", "propose"]
