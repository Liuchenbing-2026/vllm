# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from pydantic import ValidationError
from transformers import LlamaConfig

from vllm.config import ModelConfig


@pytest.fixture
def local_model(tmp_path):
    LlamaConfig(
        architectures=["LlamaForCausalLM"],
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=32,
        max_position_embeddings=32,
    ).save_pretrained(tmp_path)
    return str(tmp_path)


@pytest.mark.parametrize("seed", [-1, -2, 2**32, 2**63])
def test_engine_seed_rejected_during_configuration(local_model, seed):
    """Reject seeds that would otherwise crash NumPy in worker startup."""
    with pytest.raises(ValidationError, match="seed"):
        ModelConfig(model=local_model, seed=seed, max_model_len=8)


@pytest.mark.parametrize("seed", [0, 1, 2**32 - 1])
def test_engine_seed_preserves_numpy_supported_boundaries(local_model, seed):
    """The complete NumPy seed range remains valid, including zero."""
    config = ModelConfig(model=local_model, seed=seed, max_model_len=8)
    assert config.seed == seed


def test_default_engine_seed_survives_engine_args(local_model):
    """Adding field constraints must not turn the CLI default into FieldInfo."""
    from vllm.engine.arg_utils import EngineArgs

    args = EngineArgs(model=local_model, max_model_len=8)
    assert args.seed == 0
    assert args.create_model_config().seed == 0
    assert ModelConfig(model=local_model, max_model_len=8).seed == 0
