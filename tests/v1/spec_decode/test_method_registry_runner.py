# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A speculative method registered by a plugin is built by Model Runner V2."""

from types import SimpleNamespace

import pytest

from vllm import envs
from vllm.config import SpeculativeConfig, VllmConfig
from vllm.v1.spec_decode import registry
from vllm.v1.spec_decode.registry import (
    SpeculativeMethodSpec,
    register_speculative_method,
)
from vllm.v1.worker.gpu.spec_decode import init_speculator


class RecordingSpeculator:
    def __init__(self, vllm_config, device):
        self.vllm_config = vllm_config
        self.device = device


@pytest.fixture
def registered_config(monkeypatch):
    isolated: dict[str, SpeculativeMethodSpec] = {}
    monkeypatch.setattr(registry, "_method_registry", lambda: isolated)
    register_speculative_method(
        "unit_test_runner_method",
        SpeculativeMethodSpec(f"{__name__}:RecordingSpeculator"),
    )
    return SpeculativeConfig(method="unit_test_runner_method", num_speculative_tokens=3)


def test_init_speculator_builds_the_registered_class(registered_config):
    vllm_config = SimpleNamespace(speculative_config=registered_config)
    speculator = init_speculator(vllm_config, "cpu")
    assert isinstance(speculator, RecordingSpeculator)
    assert speculator.vllm_config is vllm_config
    assert speculator.device == "cpu"


def _use_v2(config: SpeculativeConfig) -> bool:
    fake = SimpleNamespace(
        attention_config=SimpleNamespace(hisparse_config=None),
        speculative_config=config,
    )
    return VllmConfig.use_v2_model_runner.fget(fake)  # type: ignore[attr-defined]


def test_registered_method_selects_model_runner_v2(registered_config, monkeypatch):
    monkeypatch.setattr(envs, "VLLM_USE_V2_MODEL_RUNNER", None)
    assert _use_v2(registered_config) is True


def test_registered_method_rejects_forced_v1(registered_config, monkeypatch):
    monkeypatch.setattr(envs, "VLLM_USE_V2_MODEL_RUNNER", False)
    with pytest.raises(ValueError, match="requires Model Runner V2"):
        _use_v2(registered_config)


def test_builtin_method_keeps_the_env_choice(monkeypatch):
    config = SpeculativeConfig(method="ngram", num_speculative_tokens=3)
    monkeypatch.setattr(envs, "VLLM_USE_V2_MODEL_RUNNER", False)
    assert _use_v2(config) is False
