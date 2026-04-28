# SPDX-FileCopyrightText: 2025-present Blue Guardrails
#
# SPDX-License-Identifier: Apache-2.0

"""Offline semconv replay tests for JSON fixtures captured by real LLM integration tests."""

import json
import os
from pathlib import Path
from typing import Any

import pytest
from haystack.dataclasses import ChatMessage

from blueguardrails_haystack.semconv import (
    convert_image_outputs_to_output_messages,
    convert_input_messages,
    convert_output_messages,
    convert_parts_to_input_messages,
    convert_parts_to_output_messages,
    convert_plain_text_to_input_messages,
    convert_plain_text_to_output_messages,
    convert_tool_definitions,
)


def _fixture_dir() -> Path:
    return Path(os.getenv("BLUEGUARDRAILS_LLM_FIXTURE_DIR", "tests/fixtures/llm_io"))


def _fixture_paths() -> list[Path]:
    fixture_dir = _fixture_dir()
    if not fixture_dir.exists():
        return []
    return sorted(fixture_dir.glob("*.json"))


def _is_chat_message_dict(value: Any) -> bool:
    return isinstance(value, dict) and "role" in value and "content" in value


def _load_chat_messages(values: list[dict[str, Any]]) -> list[ChatMessage]:
    return [ChatMessage.from_dict(value) for value in values]


def _normalize_tool_definitions(value: Any) -> Any:
    """Remove provider-added JSON Schema defaults from tool definitions.

    Args:
        value: Tool definition value to normalize.

    Returns:
        Normalized tool definition value.
    """
    if isinstance(value, dict):
        return {
            key: _normalize_tool_definitions(item)
            for key, item in value.items()
            if not (key == "additionalProperties" and item is False)
        }
    if isinstance(value, list):
        return [_normalize_tool_definitions(item) for item in value]
    return value


def _assert_fixture_conversion(path: Path) -> None:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    semconv = fixture.get("semconv", {})
    raw_init = fixture.get("raw_init", {})
    raw_input = fixture["raw_input"]
    raw_output = fixture["raw_output"]

    expected_input = semconv.get("gen_ai.input.messages")
    if expected_input is not None:
        if raw_input.get("messages"):
            actual_input = json.loads(convert_input_messages(_load_chat_messages(raw_input["messages"])))
        elif "prompt" in raw_input:
            actual_input = json.loads(convert_plain_text_to_input_messages(raw_input["prompt"]))
        elif "parts" in raw_input:
            actual_input = json.loads(convert_parts_to_input_messages(raw_input["parts"]))
        else:
            raise AssertionError(f"fixture {path} has no supported raw input shape")
        assert actual_input == expected_input

    expected_output = semconv.get("gen_ai.output.messages")
    if expected_output is not None:
        replies = raw_output.get("replies") or []
        if replies and _is_chat_message_dict(replies[0]):
            actual_output = json.loads(convert_output_messages(_load_chat_messages(replies)))
        elif replies and isinstance(replies[0], str):
            meta = raw_output.get("meta") or []
            finish_reasons = [item.get("finish_reason") for item in meta if isinstance(item, dict)] or None
            actual_output = json.loads(convert_plain_text_to_output_messages(replies, finish_reasons))
        elif raw_output.get("images"):
            actual_output = json.loads(convert_image_outputs_to_output_messages(raw_output["images"]))
        elif raw_output.get("files"):
            actual_output = json.loads(convert_parts_to_output_messages(raw_output["files"]))
        elif raw_output.get("parts"):
            actual_output = json.loads(convert_parts_to_output_messages(raw_output["parts"]))
        else:
            raise AssertionError(f"fixture {path} has no supported raw output shape")
        assert actual_output == expected_output

    expected_tools = semconv.get("gen_ai.tool.definitions")
    if expected_tools is not None:
        raw_tools = raw_input.get("tools") if "tools" in raw_input else raw_init.get("tools")
        actual_tools = json.loads(convert_tool_definitions(raw_tools))
        assert _normalize_tool_definitions(actual_tools) == _normalize_tool_definitions(expected_tools)


def test_recorded_llm_fixtures_replay_semconv_conversion() -> None:
    """Captured provider I/O JSON can be replayed without live provider calls."""
    paths = _fixture_paths()
    if not paths:
        pytest.skip(
            f"no LLM fixtures found in {_fixture_dir()}; set BLUEGUARDRAILS_RECORD_LLM_FIXTURES=1 "
            "when running integration tests"
        )

    for path in paths:
        _assert_fixture_conversion(path)
