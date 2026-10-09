# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard the native attention contract and raw, last-token decision readout."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.config import PplxDeciderConfig
from vllm.model_executor.models.pplx_decider import DeciderPooler


@pytest.mark.parametrize(
    "mode,pooling",
    [("causal", "last"), (None, "last"), ("noncausal_full_attention", "mean")],
)
def test_rejects_incompatible_checkpoint_contract(mode, pooling):
    config = SimpleNamespace(
        decision_attention_mode=mode, decision_pooling=pooling, num_labels=255
    )
    model = SimpleNamespace(hf_config=config, hf_text_config=SimpleNamespace())
    with pytest.raises(ValueError):
        PplxDeciderConfig.verify_and_update_model_config(model)


def test_enables_noncausal_full_attention_on_nested_text_config():
    config = SimpleNamespace(
        decision_attention_mode="noncausal_full_attention",
        decision_pooling="last",
        num_labels=255,
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
