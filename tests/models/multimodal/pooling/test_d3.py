# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard the native attention contract and raw, last-token decision readout."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.config import D3Config
from vllm.model_executor.models.d3 import D3DecisionPooler

# These tests use CPU tensors and mocked backbones, never a device process group.
pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize(
    "mode,pooling",
    [("causal", "last"), (None, "last"), ("noncausal_full_attention", "mean")],
)
def test_rejects_incompatible_checkpoint_contract(mode, pooling):
    config = SimpleNamespace(
        decision_attention_mode=mode,
        decision_pooling=pooling,
        num_labels=255,
        decision_readout_dtype="float32",
    )
    model = SimpleNamespace(hf_config=config, hf_text_config=SimpleNamespace())
    with pytest.raises(ValueError):
        D3Config.verify_and_update_model_config(model)


def test_enables_noncausal_full_attention_on_nested_text_config():
    config = SimpleNamespace(
        decision_attention_mode="noncausal_full_attention",
        decision_pooling="last",
        num_labels=255,
        decision_readout_dtype="float32",
    )
    text = SimpleNamespace()
    D3Config.verify_and_update_model_config(
        SimpleNamespace(hf_config=config, hf_text_config=text)
    )
    assert config.is_causal is False
    assert text.is_causal is False


def test_readout_selects_each_last_token_and_preserves_raw_logits():
    pooler = D3DecisionPooler(2, 3, torch.float32)
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


def test_readout_preserves_fp32_weights_with_bf16_hidden_states():
    pooler = D3DecisionPooler(2, 2, torch.float32)
    weight = torch.tensor([[1.001, 0.1234567], [-0.9876543, 0.002]])
    with torch.no_grad():
        pooler.readout.weight.copy_(weight)
    hidden = torch.tensor([[3.0, 2.0]], dtype=torch.bfloat16)
    metadata = SimpleNamespace(
        get_pooling_cursor=lambda: SimpleNamespace(
            last_token_indices_gpu=torch.tensor([0])
        )
    )
    torch.testing.assert_close(
        pooler(hidden, metadata), hidden.float() @ weight.T, rtol=0, atol=0
    )
    assert pooler.readout.weight.dtype == torch.float32


def test_rejects_bf16_readout_contract():
    config = SimpleNamespace(
        decision_attention_mode="noncausal_full_attention",
        decision_pooling="last",
        num_labels=255,
        decision_readout_dtype="bfloat16",
    )
    with pytest.raises(ValueError, match="float32"):
        D3Config.verify_and_update_model_config(
            SimpleNamespace(hf_config=config, hf_text_config=SimpleNamespace())
        )


def test_bf16_backbone_keeps_fp32_readout(monkeypatch):
    from vllm.model_executor.models.d3 import D3ForSequenceClassification
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration

    def initialize_backbone(self, **kwargs):
        torch.nn.Module.__init__(self)
        self.config = SimpleNamespace(num_labels=3)
        self.language_model = torch.nn.Module()
        self.language_model.lm_head = torch.nn.Linear(2, 8)

    monkeypatch.setattr(
        Qwen3_5ForConditionalGeneration, "__init__", initialize_backbone
    )
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        model_config=SimpleNamespace(get_hidden_size=lambda: 2, dtype=torch.bfloat16),
    )
    model = D3ForSequenceClassification(vllm_config=config)
    assert model.pooler.readout.weight.dtype == torch.float32
    assert not hasattr(model.language_model, "lm_head")


def test_http_contract_rejects_unsupported_video_and_server_paths(monkeypatch):
    # The schema does not execute the external checkpoint renderer.
    monkeypatch.setitem(
        sys.modules,
        "d3_runtime",
        SimpleNamespace(D3=object, answer_codes=None, load_processor=None),
    )
    path = Path(__file__).parents[4] / "examples/pooling/classify/serve_d3.py"
    spec = importlib.util.spec_from_file_location("serve_d3_contract_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    request = module.DecisionRequest
    request(model="d3", state="A state", questions={})
    with pytest.raises(ValueError, match="videos"):
        request(state="A state", questions={}, videos=["clip.mp4"])
    with pytest.raises(ValueError, match="image data URLs"):
        request(state="A state", questions={}, images=["/tmp/private.png"])
    from fastapi.testclient import TestClient

    client = TestClient(module.create_app(None))
    for extra in [{"videos": ["clip.mp4"]}, {"images": ["/tmp/private.png"]}]:
        response = client.post(
            "/v1/systemone", json={"state": "A state", "questions": {}, **extra}
        )
        assert response.status_code == 422
