# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.utils import construct_tool_dicts
from vllm.exceptions import VLLMValidationError
from vllm.tool_parsers.gemma4_engine_tool_parser import Gemma4EngineToolParser
from vllm.tool_parsers.utils import get_json_schema_from_tools


def _responses_request(
    *,
    tool_choice: dict | str,
    tools: list[dict] | None = None,
) -> ResponsesRequest:
    return ResponsesRequest.model_validate(
        {
            "input": "What is the current time in UTC?",
            "tools": tools
            if tools is not None
            else [
                {
                    "type": "function",
                    "name": "get_temperature",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
                {
                    "type": "function",
                    "name": "get_time",
                    "parameters": {
                        "type": "object",
                        "properties": {"timezone": {"type": "string"}},
                    },
                },
            ],
            "tool_choice": tool_choice,
        }
    )


def test_allowed_tools_filters_tools_shown_to_model() -> None:
    request = _responses_request(
        tool_choice={
            "type": "allowed_tools",
            "mode": "required",
            "tools": [{"type": "function", "name": "get_time"}],
        }
    )

    tool_dicts = construct_tool_dicts(request.tools, request.tool_choice)

    assert tool_dicts is not None
    assert [tool["function"]["name"] for tool in tool_dicts] == ["get_time"]


def test_allowed_tools_rejects_unknown_function() -> None:
    with pytest.raises(VLLMValidationError, match="not found"):
        _responses_request(
            tool_choice={
                "type": "allowed_tools",
                "mode": "required",
                "tools": [{"type": "function", "name": "missing_tool"}],
            }
        )


def test_allowed_tools_rejects_function_without_tools() -> None:
    with pytest.raises(VLLMValidationError, match="tools"):
        _responses_request(
            tools=[],
            tool_choice={
                "type": "allowed_tools",
                "mode": "required",
                "tools": [{"type": "function", "name": "get_time"}],
            },
        )


def test_allowed_tools_required_schema_contains_only_allowed_function() -> None:
    request = _responses_request(
        tool_choice={
            "type": "allowed_tools",
            "mode": "required",
            "tools": [{"type": "function", "name": "get_time"}],
        }
    )

    schema = get_json_schema_from_tools(request.tool_choice, request.tools)

    assert isinstance(schema, dict)
    names = {
        entry["properties"]["name"]["enum"][0] for entry in schema["items"]["anyOf"]
    }
    assert names == {"get_time"}


def test_allowed_tools_auto_does_not_force_json_tool_call() -> None:
    request = _responses_request(
        tool_choice={
            "type": "allowed_tools",
            "mode": "auto",
            "tools": [{"type": "function", "name": "get_time"}],
        }
    )

    assert get_json_schema_from_tools(request.tool_choice, request.tools) is None


def test_allowed_tools_required_preserves_gemma4_native_markers() -> None:
    request = _responses_request(
        tool_choice={
            "type": "allowed_tools",
            "mode": "required",
            "tools": [{"type": "function", "name": "get_time"}],
        }
    )
    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = {
        "<|tool_call>": 1,
        "<tool_call|>": 2,
        '<|"|>': 3,
        "<|channel>": 4,
        "<channel|>": 5,
    }
    parser = Gemma4EngineToolParser(tokenizer)

    parser.adjust_request(request)

    assert request.skip_special_tokens is False
    assert request.text is None
