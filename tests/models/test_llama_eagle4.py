# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.llama_eagle4 import (
    clear_identity_d2t,
    gather_dot_reference,
    low_rank_scores,
    merge_tp_candidates,
)

pytestmark = pytest.mark.skip_global_cleanup


def test_low_rank_scores_masks_vocab_padding():
    hidden = torch.tensor([[1.0, 2.0]])
    down = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    head = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [10.0, 10.0],
        ]
    )

    scores = low_rank_scores(hidden, down, head, local_vocab_rows=2)

    assert torch.equal(scores[:, :2], torch.tensor([[1.0, 2.0]]))
    assert torch.isneginf(scores[0, 2])


def test_exact_candidate_rescoring_matches_full_dot_products():
    weight = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [-1.0, 0.5]],
    )
    hidden = torch.tensor([[0.25, -0.5], [2.0, 1.0]])
    indices = torch.tensor([[2, 0], [1, 2]])

    logits = gather_dot_reference(weight, indices, hidden)
    expected = torch.gather(hidden @ weight.t(), 1, indices)

    assert torch.allclose(logits, expected)


def test_tp_candidate_merge_is_rank_major_and_global_topk():
    # Two ranks, two rows, two candidates per rank.
    local_logits = torch.tensor([[1.0, 2.0], [5.0, 6.0]])
    local_ids = torch.tensor([[0, 1], [2, 3]])

    def fake_all_gather(value):
        return torch.cat([value, value + 10], dim=0)

    logits, ids = merge_tp_candidates(
        local_logits,
        local_ids,
        2,
        all_gather=fake_all_gather,
        tp_size=2,
    )

    assert logits.shape == (2, 2)
    assert ids.shape == (2, 2)
    assert set(ids[0].tolist()) == {10, 11}
    assert set(ids[1].tolist()) == {12, 13}
    assert torch.equal(logits, torch.tensor([[12.0, 11.0], [16.0, 15.0]]))


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"num_hidden_layers": 2}, "single draft"),
        ({"num_aux_hidden_states": 2}, "three auxiliary"),
        (
            {"efficient_lm_head_config": {"type": "pca", "rank": 8, "num_candidates": 16}},
            "type must be",
        ),
        (
            {
                "efficient_lm_head_config": {
                    "type": "weighted_low_rank",
                    "rank": 8,
                    "num_candidates": 16,
                }
            },
            "type must be",
        ),
    ],
)
def test_eagle4_config_validation_rejects_unsupported_structures(
    override, message
):
    from vllm.config.speculative import _validate_eagle4_draft_config

    base = {
        "num_hidden_layers": 1,
        "num_aux_hidden_states": 3,
        "hidden_size": 16,
        "target_hidden_size": 16,
        "vocab_size": 100,
        "draft_vocab_size": 100,
        "eagle_config": {
            "eagle_aux_hidden_state_layer_ids": [1, 2, 3],
        },
        "efficient_lm_head_config": {
            "type": "low_rank",
            "rank": 8,
            "num_candidates": 16,
        },
    }
    base.update(override)
    draft = SimpleNamespace(hf_config=SimpleNamespace(**base))
    target = SimpleNamespace(
        hf_text_config=SimpleNamespace(hidden_size=16, vocab_size=100)
    )

    with pytest.raises((ValueError, NotImplementedError), match=message):
        _validate_eagle4_draft_config(target, draft)


def test_eagle4_config_normalizes_aux_layer_outputs_to_vllm_inputs():
    from vllm.config.speculative import _validate_eagle4_draft_config

    hf_config = SimpleNamespace(
        num_hidden_layers=1,
        num_aux_hidden_states=3,
        hidden_size=16,
        target_hidden_size=16,
        vocab_size=100,
        draft_vocab_size=100,
        eagle_config={"eagle_aux_hidden_state_layer_ids": [3, 31, 59]},
        efficient_lm_head_config={
            "type": "low_rank",
            "rank": 8,
            "num_candidates": 16,
        },
    )
    draft = SimpleNamespace(hf_config=hf_config)
    target = SimpleNamespace(
        hf_text_config=SimpleNamespace(hidden_size=16, vocab_size=100)
    )

    _validate_eagle4_draft_config(target, draft)
    _validate_eagle4_draft_config(target, draft)

    assert hf_config.eagle_aux_hidden_state_layer_ids == (4, 32, 60)


def test_eagle4_config_rejects_target_vocab_mismatch():
    from vllm.config.speculative import _validate_eagle4_draft_config

    hf_config = SimpleNamespace(
        num_hidden_layers=1,
        num_aux_hidden_states=3,
        hidden_size=16,
        target_hidden_size=16,
        vocab_size=100,
        draft_vocab_size=100,
        eagle_config={"eagle_aux_hidden_state_layer_ids": [3, 4, 5]},
        efficient_lm_head_config={
            "type": "low_rank",
            "rank": 8,
            "num_candidates": 16,
        },
    )
    draft = SimpleNamespace(hf_config=hf_config)
    target = SimpleNamespace(
        hf_text_config=SimpleNamespace(hidden_size=16, vocab_size=101)
    )

    with pytest.raises(ValueError, match="match the target vocabulary"):
        _validate_eagle4_draft_config(target, draft)


def test_eagle4_uses_candidate_head_for_greedy_drafting():
    from vllm.config.speculative import _configure_eagle4_runtime

    config = SimpleNamespace()
    config.draft_sample_method = "greedy"
    config.enable_adaptive_verification = False
    config.parallel_drafting = False
    config.use_local_argmax_reduction = False

    _configure_eagle4_runtime(config)

    assert config.use_local_argmax_reduction


def test_eagle4_rejects_tree_runtime_on_vllm_030():
    from vllm.config.speculative import _configure_eagle4_runtime

    config = SimpleNamespace(
        draft_sample_method="greedy",
        enable_adaptive_verification=False,
        parallel_drafting=True,
        use_local_argmax_reduction=False,
    )

    with pytest.raises(NotImplementedError, match="tree drafting"):
        _configure_eagle4_runtime(config)


def test_eagle4_rejects_adaptive_verification_without_full_logits():
    from vllm.config.speculative import _configure_eagle4_runtime

    config = SimpleNamespace(
        draft_sample_method="greedy",
        enable_adaptive_verification=True,
        parallel_drafting=False,
        use_local_argmax_reduction=False,
    )

    with pytest.raises(NotImplementedError, match="adaptive verification"):
        _configure_eagle4_runtime(config)


def test_eagle4_keeps_independent_full_vocabulary_lm_head():
    from vllm.model_executor.models.llama_eagle4 import (
        LlamaForCausalLMEagle4,
    )

    assert LlamaForCausalLMEagle4.has_own_lm_head is True


def test_eagle4_clears_identity_d2t_parameter():
    from torch import nn

    model = nn.Module()
    model.config = SimpleNamespace(vocab_size=100, draft_vocab_size=100)
    model.draft_id_to_target_id = nn.Parameter(
        torch.zeros(100, dtype=torch.long), requires_grad=False
    )

    clear_identity_d2t(model)

    assert model.draft_id_to_target_id is None
    assert "draft_id_to_target_id" not in dict(model.named_parameters())


def test_eagle4_rejects_reduced_vocabulary_d2t_mapping():
    from torch import nn

    model = nn.Module()
    model.config = SimpleNamespace(vocab_size=100, draft_vocab_size=80)
    model.draft_id_to_target_id = nn.Parameter(
        torch.zeros(80, dtype=torch.long), requires_grad=False
    )

    with pytest.raises(ValueError, match="cannot use a d2t mapping"):
        clear_identity_d2t(model)


def test_qwen38_checkpoint_keys_map_to_eagle3_loader_targets(monkeypatch):
    from vllm.model_executor.models import llama_eagle3
    from vllm.model_executor.models.llama_eagle4 import (
        LlamaForCausalLMEagle4,
    )

    captured = {}

    class RecordingLoader:
        def __init__(self, model):
            assert model is draft

        def load_weights(self, weights, mapper):
            captured["names"] = [name for name, _ in weights]
            captured["mapper"] = mapper

    draft = object.__new__(LlamaForCausalLMEagle4)
    draft.use_parallel_drafting = False
    draft.model = SimpleNamespace(
        use_aux_hidden_state=True,
        norm_before_fc=False,
    )
    monkeypatch.setattr(llama_eagle3, "AutoWeightsLoader", RecordingLoader)
    monkeypatch.setattr(
        llama_eagle3,
        "process_eagle_weight",
        lambda model, name: captured.setdefault("processed", []).append(name),
    )

    checkpoint_names = [
        "efficient_lm_head.weight",
        "efficient_lm_head_down_proj.weight",
        "embed_tokens.weight",
        "fc.weight",
        "fc_norm.1.weight",
        "layers.0.hidden_norm.weight",
        "layers.0.input_layernorm.weight",
        "layers.0.mlp.down_proj.weight",
        "layers.0.mlp.gate_proj.weight",
        "layers.0.mlp.up_proj.weight",
        "layers.0.post_attention_layernorm.weight",
        "layers.0.self_attn.k_proj.weight",
        "layers.0.self_attn.o_proj.weight",
        "layers.0.self_attn.q_proj.weight",
        "layers.0.self_attn.v_proj.weight",
        "lm_head.weight",
        "norm.weight",
    ]
    tensors = [(name, torch.empty(0)) for name in checkpoint_names]

    draft.load_weights(iter(tensors))

    expected = [
        "efficient_lm_head.weight",
        "efficient_lm_head_down_proj.weight",
        "model.embed_tokens.weight",
        "model.fc.weight",
        "model.fc_norm.1.weight",
        "model.layers.0.hidden_norm.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "lm_head.weight",
        "model.norm.weight",
    ]
    assert captured["names"] == expected
    # The actual AutoWeightsLoader applies these stacked mappings after this
    # prefix/member normalization.
    stacked = captured["mapper"].orig_to_new_stacked
    assert stacked[".q_proj"] == (".qkv_proj", "q")
    assert stacked[".k_proj"] == (".qkv_proj", "k")
    assert stacked[".v_proj"] == (".qkv_proj", "v")
    assert stacked[".gate_proj"] == (".gate_up_proj", 0)
    assert stacked[".up_proj"] == (".gate_up_proj", 1)
