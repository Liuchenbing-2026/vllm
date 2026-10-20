# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for config aliases that share one config class.

``kimi_k2`` and ``deepseek_v32`` both resolve to ``DeepseekV3Config``.
``PretrainedConfig.to_dict`` serializes ``type(self).model_type``, so
registering one alias must not rewrite the identity of configs loaded under
another alias.
"""

import json
import pickle
import subprocess
import sys

import pytest
from transformers import CONFIG_MAPPING

from vllm.transformers_utils.config import (
    _CONFIG_REGISTRY,
    _register_config_class,
    get_config,
)


def _write_config(root, name: str, model_type: str) -> str:
    path = root / name
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": model_type,
                "architectures": ["DeepseekV3ForCausalLM"],
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "vocab_size": 128,
            }
        )
    )
    return str(path)


@pytest.mark.parametrize(
    ("first_model_type", "second_model_type"),
    [("kimi_k2", "deepseek_v32"), ("deepseek_v32", "kimi_k2")],
)
def test_loading_another_alias_keeps_serialized_identity(
    tmp_path, first_model_type, second_model_type
):
    first = get_config(
        _write_config(tmp_path, "first", first_model_type),
        trust_remote_code=False,
    )
    assert first.to_dict()["model_type"] == first_model_type

    second = get_config(
        _write_config(tmp_path, "second", second_model_type),
        trust_remote_code=False,
    )
    assert second.to_dict()["model_type"] == second_model_type
    assert first.to_dict()["model_type"] == first_model_type


def test_loading_same_alias_keeps_serialized_identity(tmp_path):
    first = get_config(
        _write_config(tmp_path, "first", "kimi_k2"),
        trust_remote_code=False,
    )
    get_config(_write_config(tmp_path, "second", "kimi_k2"), trust_remote_code=False)

    assert first.to_dict()["model_type"] == "kimi_k2"


def test_alias_registration_does_not_mutate_shared_class():
    config_class = _CONFIG_REGISTRY["deepseek_v32"]
    original_model_type = config_class.model_type

    _register_config_class("kimi_k2", config_class)
    _register_config_class("deepseek_v32", config_class)

    assert config_class.model_type == original_model_type
    assert CONFIG_MAPPING["kimi_k2"].model_type == "kimi_k2"
    assert CONFIG_MAPPING["deepseek_v32"].model_type == "deepseek_v32"
    assert issubclass(CONFIG_MAPPING["kimi_k2"], config_class)
    assert issubclass(CONFIG_MAPPING["deepseek_v32"], config_class)


def test_alias_config_roundtrips_through_pickle(tmp_path):
    config = get_config(
        _write_config(tmp_path, "model", "kimi_k2"),
        trust_remote_code=False,
    )

    restored = pickle.loads(pickle.dumps(config))

    assert isinstance(restored, type(config))
    assert restored.to_dict() == config.to_dict()
    assert restored.to_dict()["model_type"] == "kimi_k2"


def test_alias_config_unpickles_in_fresh_process(tmp_path):
    config = get_config(
        _write_config(tmp_path, "model", "kimi_k2"),
        trust_remote_code=False,
    )
    payload = pickle.dumps(config)

    script = (
        "import pickle, sys\n"
        "config = pickle.loads(sys.stdin.buffer.read())\n"
        "print(config.to_dict()['model_type'])\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        input=payload,
        capture_output=True,
        check=True,
        timeout=600,
    )

    assert result.stdout.strip().splitlines()[-1] == b"kimi_k2"
