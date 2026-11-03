# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard the native attention contract and raw, last-token decision readout."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.models.config import PplxDeciderConfig
from vllm.model_executor.models.pplx_decider import DeciderJointPooler, DeciderPooler


@pytest.mark.parametrize(
    "mode,pooling",
    [("unknown", "last"), (None, "last"), ("noncausal_full_attention", "mean")],
)
def test_rejects_incompatible_checkpoint_contract(mode, pooling):
    config = SimpleNamespace(
        decision_attention_mode=mode,
        decision_pooling=pooling,
        num_labels=255,
        architectures=["PplxDeciderForSequenceClassification"],
    )
    model = SimpleNamespace(hf_config=config, hf_text_config=SimpleNamespace())
    with pytest.raises(ValueError):
        PplxDeciderConfig.verify_and_update_model_config(model)


def test_enables_noncausal_full_attention_on_nested_text_config():
    config = SimpleNamespace(
        decision_attention_mode="noncausal_full_attention",
        decision_pooling="last",
        num_labels=255,
        architectures=["PplxDeciderForSequenceClassification"],
    )
    text = SimpleNamespace()
    PplxDeciderConfig.verify_and_update_model_config(
        SimpleNamespace(hf_config=config, hf_text_config=text)
    )
    assert config.is_causal is False
    assert text.is_causal is False


def test_readout_selects_each_last_token_and_preserves_raw_logits():
    pooler = DeciderPooler(2, 3, torch.bfloat16)
    with torch.no_grad():
        pooler.readout.weight.copy_(torch.tensor([[1, 0], [0, 1], [-1, 1]]))
    hidden = torch.tensor(
        [[100, 100], [2, 3], [200, 200], [4, 6]], dtype=torch.bfloat16
    )
    metadata = SimpleNamespace(
        get_pooling_cursor=lambda: SimpleNamespace(
            last_token_indices_gpu=torch.tensor([1, 3])
        )
    )
    result = pooler(hidden, metadata)
    torch.testing.assert_close(result, torch.tensor([[2.0, 3.0, 1.0], [4.0, 6.0, 2.0]]))
    assert result.dtype == torch.float32
    assert pooler.get_supported_tasks() == {"classify"}


@pytest.mark.parametrize("joint,causal", [(False, True), (False, False), (True, False)])
def test_version_specific_attention_and_readout(joint, causal):
    config = SimpleNamespace(
        architectures=[
            "PplxDeciderForTokenClassification"
            if joint
            else "PplxDeciderForSequenceClassification"
        ],
        decision_attention_mode="causal" if causal else "noncausal_full_attention",
        decision_pooling="inline_marker" if joint else "last",
        num_labels=255,
    )
    text = SimpleNamespace()
    PplxDeciderConfig.verify_and_update_model_config(
        SimpleNamespace(hf_config=config, hf_text_config=text)
    )
    assert text.is_causal == causal


def test_joint_readout_keeps_all_markers_and_request_boundaries():
    """Last-token pooling would lose earlier questions and read a template token."""
    fake_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            enable_chunked_prefill=False,
            async_scheduling=False,
        )
    )
    with patch(
        "vllm.model_executor.layers.pooler.tokwise.methods.get_current_vllm_config",
        return_value=fake_config,
    ):
        pooler = DeciderJointPooler(2, 3, torch.bfloat16)
    with torch.no_grad():
        pooler.readout.weight.copy_(torch.tensor([[1, 0], [0, 1], [-1, 1]]))
    hidden = torch.tensor(
        [[2, 3], [99, 99], [4, 6], [99, 99], [7, 8], [99, 99]],
        dtype=torch.bfloat16,
    )
    params = SimpleNamespace(step_tag_id=42, returned_token_ids=None)
    metadata = SimpleNamespace(
        pooling_params=[params, params],
        get_pooling_cursor=lambda: SimpleNamespace(
            num_scheduled_tokens_cpu=torch.tensor([4, 2])
        ),
        get_prompt_token_ids_cpu=lambda: [
            torch.tensor([42, 0, 42, 0]),
            torch.tensor([42, 0]),
        ],
    )
    outputs = pooler(hidden, metadata)
    torch.testing.assert_close(
        outputs[0], torch.tensor([[2.0, 3.0, 1.0], [4.0, 6.0, 2.0]])
    )
    torch.testing.assert_close(outputs[1], torch.tensor([[7.0, 8.0, 1.0]]))
    params.step_tag_id = None
    with pytest.raises(ValueError, match="requires a marker"):
        pooler(hidden, metadata)
