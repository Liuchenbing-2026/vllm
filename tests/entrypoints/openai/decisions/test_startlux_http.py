# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise native decision startup and typed answers over real HTTP."""

import json
import math
from typing import Any

import pytest
import requests
from transformers import AutoTokenizer

from tests.utils import RemoteOpenAIServer
from vllm.transformers_utils.repo_utils import get_hf_file_to_dict


@pytest.fixture(scope="module")
def decision_server():
    model = "startlux-models/StartLux-Decision-4B"
    tokenizer = AutoTokenizer.from_pretrained(model)
    metadata = get_hf_file_to_dict("decision_config.json", model)
    assert metadata is not None
    temperatures = metadata["temperature_by_type"]
    args = [
        "--served-model-name",
        "startlux-decision",
        "--runner",
        "pooling",
        "--hf-overrides",
        json.dumps(
            {
                "architectures": ["StartLuxDecisionForSequenceClassification"],
                "num_labels": 26,
            }
        ),
        "--pooler-config",
        '{"task":"classify","use_activation":false}',
        "--enforce-eager",
        "--max-model-len",
        "1024",
        "--max-num-batched-tokens",
        "1024",
        "--max-num-seqs",
        "2",
        "--gpu-memory-utilization",
        "0.3",
        "--no-enable-chunked-prefill",
        "--no-enable-prefix-caching",
        "--limit-mm-per-prompt",
        '{"image":0,"video":0}',
    ]
    with RemoteOpenAIServer(model, args, max_wait_seconds=300) as server:
        yield server, tokenizer, temperatures


def test_native_decisions_contract(decision_server):
    """The real route must preserve answer types, mappings and token accounting."""
    server, tokenizer, temperatures = decision_server
    state = "A customer was charged twice. They request a refund. Nobody is in danger."
    questions: list[dict[str, Any]] = [
        {
            "type": "choice",
            "name": "team",
            "instructions": "Choose the team.",
            "choices": [{"value": "billing"}, {"value": "shipping"}],
        },
        {"type": "predicate", "name": "danger", "instructions": "Is anyone in danger?"},
        {
            "type": "score",
            "name": "impact",
            "instructions": "Rate the impact.",
            "levels": [{"label": "low"}, {"label": "high"}],
        },
    ]
    # Reference rendering follows the checkpoint's fixed text template, without
    # constructing the implementation's strategy or changing engine internals.
    options = [("billing", "shipping"), ("yes", "no"), ("0: low", "1: high")]
    ids = []
    for question, labels in zip(questions, options):
        rendered = tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": "Apply the criterion to the evidence. "
                    "Choose exactly one listed option. "
                    "Answer with its letter only.",
                },
                {
                    "role": "user",
                    "content": f"Evidence:\n{state}\n\n"
                    f"Question: {question['instructions']}\nOptions:\n"
                    + "\n".join(
                        f"{letter}) {label}" for letter, label in zip("AB", labels)
                    ),
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids.append(tokenizer.encode(rendered, add_special_tokens=False))
    response = requests.post(
        server.url_for("v1", "decisions"),
        json={
            "model": "startlux-decision",
            "input": state,
            "questions": questions,
        },
        timeout=45,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] == "startlux-decision"
    assert [a["name"] for a in body["answers"]] == ["team", "danger", "impact"]
    assert [a["type"] for a in body["answers"]] == ["choice", "predicate", "score"]
    for answer, kind in zip(body["answers"], ("choice", "noul", "score")):
        if kind == "noul":
            assert math.isfinite(answer["probability"])
            assert 0 <= answer["probability"] <= 1
        else:
            probabilities = [p["probability"] for p in answer["probabilities"]]
            assert len(probabilities) == 2
            assert all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
            assert sum(probabilities) == pytest.approx(1)
            assert answer["confidence"] == pytest.approx(max(probabilities))
            if kind == "choice":
                values = [p["value"] for p in answer["probabilities"]]
                assert values == ["billing", "shipping"]
                assert answer["choice"] in [
                    v for v, p in zip(values, probabilities) if p == max(probabilities)
                ]
            else:
                assert [p["value"] for p in answer["probabilities"]] == [0, 1]
                assert [p["label"] for p in answer["probabilities"]] == ["low", "high"]
                assert answer["score"] == pytest.approx(probabilities[1])
    assert body["usage"]["input_tokens"] == sum(map(len, ids))
    assert body["usage"]["output_tokens"] == 0
    assert body["usage"]["total_tokens"] == sum(map(len, ids))

    # Compare the public true probability with an independent yes/no label read.
    reference = requests.post(
        server.url_for("classify"),
        json={
            "model": "startlux-decision",
            "input": ids[1],
            "use_activation": False,
        },
        timeout=45,
    )
    assert reference.status_code == 200, reference.text
    logits = reference.json()["data"][0]["probs"][:2]
    scaled = [value / temperatures["noul"] for value in logits]
    weights = [math.exp(value - max(scaled)) for value in scaled]
    yes_probability = weights[0] / sum(weights)
    assert body["answers"][1]["probability"] == pytest.approx(yes_probability)


@pytest.mark.parametrize("case", ["joint", "too_many_choices", "too_many_levels"])
def test_native_decisions_rejects_unsupported_requests(decision_server, case):
    """Reject unsupported contracts at the real route, before scoring."""
    server, _, _ = decision_server
    question = {
        "type": "choice",
        "instructions": "Choose.",
        "choices": [{"value": "first"}, {"value": "second"}],
    }
    request = {"model": "startlux-decision", "input": "test", "questions": [question]}
    if case == "joint":
        request["joint"] = True
    elif case == "too_many_choices":
        question["choices"] = [{"value": str(i)} for i in range(27)]
    else:
        request["questions"] = [
            {
                "type": "score",
                "instructions": "Rate.",
                "levels": [{"label": str(i)} for i in range(11)],
            }
        ]
    response = requests.post(
        server.url_for("v1", "decisions"), json=request, timeout=10
    )
    assert response.status_code == 400, response.text
    error = response.json()
    assert "answers" not in error
    assert error.get("error", error).get("message")
