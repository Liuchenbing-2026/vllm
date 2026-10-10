# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regressions for last-token selection and FP32 decision head loading."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.models.qwen3_5 import Qwen3_5MoeForConditionalGeneration
from vllm.model_executor.models.startlux_decision import (
    StartLuxDecisionMoeForSequenceClassification,
    StartLuxDecisionPooler,
)


@pytest.mark.skip_global_cleanup
class TestStartLuxDecisionPooler(unittest.TestCase):
    def test_constructor_loads_tokenizer_without_base_tokenizer_attribute(self):
        # v0.26 Qwen3.5 does not set _tokenizer in its constructor.
        config = SimpleNamespace(
            parallel_config=SimpleNamespace(pipeline_parallel_size=1),
            model_config=SimpleNamespace(get_hidden_size=lambda: 4),
        )
        tokenizer = SimpleNamespace(
            encode=lambda text, **kwargs: [ord(text) - ord("A")]
        )
        with (
            patch.object(
                Qwen3_5MoeForConditionalGeneration,
                "__init__",
                lambda self, **kwargs: torch.nn.Module.__init__(self),
            ),
            patch(
                "vllm.model_executor.models.startlux_decision."
                "cached_tokenizer_from_config",
                return_value=tokenizer,
            ),
        ):
            model = StartLuxDecisionMoeForSequenceClassification(vllm_config=config)
            self.assertEqual(model.pooler.symbol_ids, list(range(26)))
            self.assertEqual(model.pooler.letter_weight.shape, (26, 4))
            tokenizer.encode = lambda *args, **kwargs: [1, 2]
            with self.assertRaisesRegex(ValueError, "single tokens"):
                StartLuxDecisionMoeForSequenceClassification(vllm_config=config)

    def test_last_positions_keep_raw_candidate_logits(self):
        # Nonconsecutive vocabulary rows catch TP-local/vocabulary index confusion.
        pooler = StartLuxDecisionPooler([1, 6, 3], 4)
        weight = torch.arange(32).reshape(8, 4).to(torch.bfloat16) / 7
        pooler.load_head(weight)
        hidden = torch.arange(20).reshape(5, 4).to(torch.bfloat16) / 3
        metadata = SimpleNamespace(
            get_pooling_cursor=lambda: SimpleNamespace(
                last_token_indices_gpu=torch.tensor([1, 4])
            )
        )
        actual = pooler(hidden, metadata)
        expected = hidden[[1, 4]].float() @ weight[[1, 6, 3]].float().T
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertGreater(actual.max().item(), 1)  # No premature softmax.

    def test_tied_embeddings_and_untied_head_load_identically(self):
        class Model(StartLuxDecisionMoeForSequenceClassification):
            def __init__(self, tied):
                torch.nn.Module.__init__(self)
                self.config = SimpleNamespace(
                    get_text_config=lambda: SimpleNamespace(tie_word_embeddings=tied)
                )
                self.pooler = StartLuxDecisionPooler([1, 3], 2)

        weight = torch.arange(10).reshape(5, 2).to(torch.bfloat16)
        loader = patch.object(
            Qwen3_5MoeForConditionalGeneration,
            "load_weights",
            lambda self, weights: {name for name, _ in weights},
        )
        loader.start()
        self.addCleanup(loader.stop)
        for tied, name in [
            (True, "model.language_model.embed_tokens.weight"),
            (False, "lm_head.weight"),
        ]:
            model = Model(tied)
            self.assertEqual(model.load_weights(iter([(name, weight)])), {name})
            torch.testing.assert_close(
                model.pooler.letter_weight, weight[[1, 3]].float()
            )
        with self.assertRaisesRegex(ValueError, "no StartLux output head"):
            Model(False).load_weights(iter([("other.weight", weight)]))


if __name__ == "__main__":
    unittest.main()
