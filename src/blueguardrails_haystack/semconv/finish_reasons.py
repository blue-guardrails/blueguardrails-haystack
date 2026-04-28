# SPDX-FileCopyrightText: 2025-present BlueGuardrails
#
# SPDX-License-Identifier: Apache-2.0

"""Provider finish-reason normalization for OTel GenAI semconv."""

from typing import Any

from blueguardrails_haystack._utils import AnyMapping, is_mapping

_FINISH_REASON_MAP = {
    # OpenAI/OpenAI-compatible, Mistral, Hugging Face, xAI.
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_call",
    "function_call": "tool_call",
    "tool_call": "tool_call",
    "content_filter": "content_filter",
    "error": "error",
    # OpenAI Responses / xAI statuses.
    "completed": "stop",
    "max_output_tokens": "length",
    "cancelled": "error",
    "canceled": "error",
    "failed": "error",
    # Anthropic / AWS Bedrock Converse.
    "compaction": "stop",
    "end_turn": "stop",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_call",
    "pause_turn": "stop",
    "refusal": "content_filter",
    "content_filtered": "content_filter",
    "guardrail_intervened": "content_filter",
    # Google/Gemini.
    "safety": "content_filter",
    "recitation": "content_filter",
    "language": "error",
    "blocklist": "content_filter",
    "prohibited_content": "content_filter",
    "spii": "content_filter",
    "malformed_function_call": "error",
    "image_safety": "content_filter",
    "unexpected_tool_call": "error",
    # Hugging Face / Cohere.
    "eos_token": "stop",
    "complete": "stop",
}

_UNKNOWN_FINISH_REASON_VALUES = {
    "",
    "none",
    "null",
    "unknown",
    "unspecified",
    "finish_reason_unspecified",
    "other",
}


def extract_finish_reason(meta: AnyMapping) -> Any:
    """Extract a raw finish reason from provider metadata."""
    for key in ("finish_reason", "stop_reason", "stopReason", "native_finish_reason"):
        if meta.get(key) is not None:
            return meta[key]

    for key in ("incomplete_details", "incompleteDetails"):
        details = meta.get(key)
        if is_mapping(details):
            if details.get("reason") is not None:
                return details["reason"]
            continue
        details_object: Any = details
        reason = getattr(details_object, "reason", None)
        if reason is not None:
            return reason

    return meta.get("status")


def normalize_finish_reason(value: Any) -> str | None:
    """Normalize a provider finish reason to the OTel GenAI vocabulary."""
    if value is None:
        return None

    # Google enums and some SDK enum-like values expose either ``value`` or ``name``.
    # Prefer a string value when present; otherwise use the enum name because many
    # Python enums expose numeric values.
    raw = getattr(value, "value", value)
    if not isinstance(raw, str):
        raw = getattr(value, "name", raw)
    if raw is None:
        return None

    key = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    key = key.rsplit(".", 1)[-1]

    candidates = [key]
    if key.startswith("finish_reason_"):
        candidates.append(key.removeprefix("finish_reason_"))

    for candidate in candidates:
        if candidate in _FINISH_REASON_MAP:
            return _FINISH_REASON_MAP[candidate]
        if candidate in _UNKNOWN_FINISH_REASON_VALUES:
            return None
    return None
