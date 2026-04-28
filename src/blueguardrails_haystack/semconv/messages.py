# SPDX-FileCopyrightText: 2025-present BlueGuardrails
#
# SPDX-License-Identifier: Apache-2.0

"""Convert Haystack messages and generator payloads to OTel GenAI JSON."""

import json
from collections.abc import Iterable
from typing import Any

from haystack.dataclasses import ChatMessage

from blueguardrails_haystack.semconv.finish_reasons import extract_finish_reason, normalize_finish_reason
from blueguardrails_haystack.semconv.parts import (
    chat_content_part_to_semconv,
    generated_image_to_semconv,
    plain_parts_to_semconv,
)


def _json(value: Any) -> str:
    return json.dumps(value, default=str)


def _chat_message_to_semconv(msg: ChatMessage) -> dict[str, Any]:
    """Convert a Haystack chat message to a GenAI message."""
    parts = [converted for part in msg._content if (converted := chat_content_part_to_semconv(part))]
    result: dict[str, Any] = {"role": msg.role.value, "parts": parts}
    if msg.name:
        result["name"] = msg.name
    return result


def _add_finish_reason(message: dict[str, Any], raw_finish_reason: Any) -> dict[str, Any]:
    finish_reason = normalize_finish_reason(raw_finish_reason)
    if finish_reason is not None:
        message["finish_reason"] = finish_reason
    return message


def _assistant_message(parts: list[dict[str, Any]], finish_reason: Any | None = None) -> dict[str, Any]:
    return _add_finish_reason({"role": "assistant", "parts": parts}, finish_reason)


def convert_input_messages(messages: list[ChatMessage]) -> str:
    """Convert input chat messages to GenAI JSON."""
    return _json([_chat_message_to_semconv(message) for message in messages])


def convert_output_messages(replies: list[ChatMessage]) -> str:
    """Convert output chat messages to GenAI JSON."""
    output_messages: list[dict[str, Any]] = []
    for reply in replies:
        message = _chat_message_to_semconv(reply)
        _add_finish_reason(message, extract_finish_reason(reply.meta))
        output_messages.append(message)
    return _json(output_messages)


def convert_plain_text_to_input_messages(prompt: str) -> str:
    """Convert a text prompt to GenAI input-message JSON."""
    return _json([{"role": "user", "parts": [{"type": "text", "content": prompt}]}])


def convert_parts_to_input_messages(parts: Iterable[Any]) -> str:
    """Convert multimodal input parts to GenAI input-message JSON."""
    return _json([{"role": "user", "parts": plain_parts_to_semconv(parts)}])


def convert_plain_text_to_output_messages(replies: list[str], finish_reasons: Iterable[Any] | None = None) -> str:
    """Convert text replies to GenAI output-message JSON."""
    raw_finish_reasons = list(finish_reasons or [])
    messages = [
        _assistant_message(
            [{"type": "text", "content": reply}],
            raw_finish_reasons[index] if index < len(raw_finish_reasons) else None,
        )
        for index, reply in enumerate(replies)
    ]
    return _json(messages)


def convert_image_outputs_to_output_messages(images: Iterable[Any], finish_reasons: Iterable[Any] | None = None) -> str:
    """Convert generated images to GenAI output-message JSON."""
    raw_finish_reasons = list(finish_reasons or [])
    messages: list[dict[str, Any]] = []
    for index, image in enumerate(images):
        part = generated_image_to_semconv(image)
        if part:
            messages.append(
                _assistant_message([part], raw_finish_reasons[index] if index < len(raw_finish_reasons) else None)
            )
    return _json(messages)


def convert_parts_to_output_messages(parts: Iterable[Any], finish_reason: Any | None = None) -> str:
    """Convert multimodal output parts to GenAI output-message JSON."""
    return _json([_assistant_message(plain_parts_to_semconv(parts), finish_reason)])
