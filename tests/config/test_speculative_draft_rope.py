# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for EAGLE draft RoPE reporting (#61231).

An EAGLE draft builds its rotary embedding from its own config, so a draft
checkpoint that omits the rope keys runs at base 10000 with no scaling even
when the target runs at base 500000 with llama3 scaling. The resolution must
be logged, and a mismatch with the target must raise a warning naming both
numbers. The values themselves must never be rewritten: a head trained
against base 10000 is self-consistent at 10000, so adopting the target's base
would be the regression.
"""

import logging

import pytest
from transformers import PreTrainedConfig

from vllm.config.model import ModelConfig
from vllm.config.parallel import ParallelConfig
from vllm.config.speculative import SpeculativeConfig

# All repos are public; only config/tokenizer-config files are fetched.
EAGLE3_DRAFT = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"  # no rope keys -> base 10000
LLAMA3_TARGET = "unsloth/Meta-Llama-3.1-8B-Instruct"  # base 500000 + llama3 scaling

_LOGGER = "vllm.config.speculative"
_DRAFT_INFO_MSG = "Draft model resolves RoPE base"
_MISMATCH_MSG = "differs from the target"


class _StubConfig:
    """Minimal HF-config stand-in; only attribute lookups are exercised."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@pytest.fixture
def vllm_caplog(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch):
    """Make caplog see vLLM logger records (vLLM sets propagate=False)."""
    monkeypatch.setattr(logging.getLogger("vllm"), "propagate", True)
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        yield caplog


def _info_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if _DRAFT_INFO_MSG in r.getMessage()]


def _mismatch_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if _MISMATCH_MSG in r.getMessage()]


@pytest.mark.cpu_test
def test_missing_rope_keys_resolve_to_10000_default():
    assert SpeculativeConfig._resolve_rope_parameters(_StubConfig()) == (
        10000,
        "default",
    )


@pytest.mark.cpu_test
def test_rope_parameters_take_precedence_over_top_level():
    hf_config = _StubConfig(
        rope_parameters={"rope_theta": 500000.0, "rope_type": "llama3"},
        rope_theta=10000.0,
    )
    assert SpeculativeConfig._resolve_rope_parameters(hf_config) == (500000.0, "llama3")


@pytest.mark.cpu_test
def test_legacy_rope_scaling_is_read():
    hf_config = _StubConfig(
        rope_scaling={"rope_theta": 500000.0, "rope_type": "llama3"},
        rope_theta=10000.0,
    )
    assert SpeculativeConfig._resolve_rope_parameters(hf_config) == (500000.0, "llama3")


@pytest.mark.cpu_test
def test_top_level_rope_theta_without_scaling():
    hf_config = _StubConfig(rope_theta=1000000.0)
    assert SpeculativeConfig._resolve_rope_parameters(hf_config) == (
        1000000.0,
        "default",
    )


@pytest.mark.cpu_test
def test_rope_type_defaults_when_only_base_is_given():
    hf_config = _StubConfig(rope_parameters={"rope_theta": 1234.0})
    assert SpeculativeConfig._resolve_rope_parameters(hf_config) == (1234.0, "default")


@pytest.mark.cpu_test
def test_warns_when_draft_default_differs_from_target(
    vllm_caplog: pytest.LogCaptureFixture,
):
    draft_hf_config = _StubConfig()  # omits the rope keys entirely
    target_hf_config = _StubConfig(
        rope_parameters={"rope_theta": 500000.0, "rope_type": "llama3"}
    )
    SpeculativeConfig._maybe_log_and_warn_draft_rope(draft_hf_config, target_hf_config)

    assert _info_records(vllm_caplog)
    mismatch = _mismatch_records(vllm_caplog)
    assert len(mismatch) == 1
    message = mismatch[0].getMessage()
    assert "10000" in message
    assert "500000" in message
    assert "default" in message
    assert "llama3" in message


@pytest.mark.cpu_test
def test_no_warning_when_draft_matches_target(
    vllm_caplog: pytest.LogCaptureFixture,
):
    rope_parameters = {"rope_theta": 500000.0, "rope_type": "llama3"}
    SpeculativeConfig._maybe_log_and_warn_draft_rope(
        _StubConfig(rope_parameters=dict(rope_parameters)),
        _StubConfig(rope_parameters=dict(rope_parameters)),
    )

    assert _info_records(vllm_caplog)
    assert not _mismatch_records(vllm_caplog)


@pytest.mark.cpu_test
def test_no_warning_when_both_stay_on_the_default(
    vllm_caplog: pytest.LogCaptureFixture,
):
    SpeculativeConfig._maybe_log_and_warn_draft_rope(_StubConfig(), _StubConfig())

    assert _info_records(vllm_caplog)
    assert not _mismatch_records(vllm_caplog)


@pytest.mark.cpu_test
def test_warns_when_only_rope_type_differs(
    vllm_caplog: pytest.LogCaptureFixture,
):
    draft_hf_config = _StubConfig(
        rope_parameters={"rope_theta": 500000.0, "rope_type": "default"}
    )
    target_hf_config = _StubConfig(
        rope_parameters={"rope_theta": 500000.0, "rope_type": "llama3"}
    )
    SpeculativeConfig._maybe_log_and_warn_draft_rope(draft_hf_config, target_hf_config)

    assert len(_mismatch_records(vllm_caplog)) == 1


@pytest.mark.cpu_test
def test_missing_target_config_only_logs(vllm_caplog: pytest.LogCaptureFixture):
    SpeculativeConfig._maybe_log_and_warn_draft_rope(_StubConfig(), None)

    assert _info_records(vllm_caplog)
    assert not _mismatch_records(vllm_caplog)


@pytest.mark.cpu_test
def test_draft_rope_is_reported_not_rewritten(vllm_caplog: pytest.LogCaptureFixture):
    """The draft's rope must survive the mismatch untouched."""
    draft_hf_config = PreTrainedConfig(rope_theta=10000.0)
    before = draft_hf_config.to_dict()
    target_hf_config = _StubConfig(
        rope_parameters={"rope_theta": 500000.0, "rope_type": "llama3"}
    )

    SpeculativeConfig._maybe_log_and_warn_draft_rope(draft_hf_config, target_hf_config)

    assert draft_hf_config.to_dict() == before
    assert SpeculativeConfig._resolve_rope_parameters(draft_hf_config) == (
        10000.0,
        "default",
    )
    assert len(_mismatch_records(vllm_caplog)) == 1


@pytest.mark.cpu_test
def test_eagle3_draft_rope_mismatch_is_reported(
    vllm_caplog: pytest.LogCaptureFixture,
):
    """End-to-end: the referenced EAGLE-3 draft omits its rope keys, so the
    mismatch is reported and the draft keeps its own (10000) resolution."""
    target_model_config = ModelConfig(LLAMA3_TARGET)
    speculative_config = SpeculativeConfig(
        target_model_config=target_model_config,
        target_parallel_config=ParallelConfig(),
        model=EAGLE3_DRAFT,
        method="eagle3",
        num_speculative_tokens=3,
    )

    draft_hf_config = speculative_config.draft_model_config.hf_config
    draft_base, _ = SpeculativeConfig._resolve_rope_parameters(draft_hf_config)
    target_base, _ = SpeculativeConfig._resolve_rope_parameters(
        target_model_config.hf_text_config
    )
    assert target_base == 500000.0
    assert draft_base == 10000
    assert len(_mismatch_records(vllm_caplog)) >= 1
