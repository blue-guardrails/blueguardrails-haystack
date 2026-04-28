# SPDX-FileCopyrightText: 2025-present BlueGuardrails
#
# SPDX-License-Identifier: Apache-2.0

"""Normalize provider usage metadata to OTel GenAI usage attributes."""

from typing import Any

from blueguardrails_haystack._utils import (
    AnyMapping,
    first_present,
    is_list,
    is_mapping,
    mapping_numeric_items,
    nested_first_present,
    snake_to_lower_camel,
    to_int,
)

_EMPTY_MAPPING: AnyMapping = {}


def _mapping_or_empty(value: Any) -> AnyMapping:
    """Return a mapping view for dynamic provider dictionaries."""
    return value if is_mapping(value) else _EMPTY_MAPPING


def _sum_token_details(value: Any) -> int | None:
    """Sum token counts from provider modality-detail lists."""
    if not is_list(value):
        return None

    total = 0
    found = False
    for item in value:
        if is_mapping(item):
            token_count = first_present(item, "token_count", "tokenCount")
        else:
            token_count = getattr(item, "token_count", None)
            if token_count is None:
                token_count = getattr(item, "tokenCount", None)

        token_count_int = to_int(token_count)
        if token_count_int is not None:
            total += token_count_int
            found = True
    return total if found else None


def _add_usage_detail(details: dict[str, int], key: str, value: Any) -> None:
    """Add a usage detail when value is a non-zero integer."""
    int_value = to_int(value)
    if int_value:
        details[key] = int_value


def _add_openai_token_detail_attrs(details: dict[str, int], usage: AnyMapping) -> None:
    """Add OpenAI Chat Completions and Responses token-detail fields."""
    prompt_details = first_present(usage, "prompt_tokens_details", "promptTokensDetails")
    input_details = first_present(usage, "input_tokens_details", "inputTokensDetails")
    completion_details = first_present(usage, "completion_tokens_details", "completionTokensDetails")
    output_details = first_present(usage, "output_tokens_details", "outputTokensDetails")

    _add_usage_detail(
        details, "input_audio_tokens", first_present(_mapping_or_empty(prompt_details), "audio_tokens", "audioTokens")
    )
    _add_usage_detail(
        details, "input_audio_tokens", first_present(_mapping_or_empty(input_details), "audio_tokens", "audioTokens")
    )

    for key, value in mapping_numeric_items(completion_details):
        _add_usage_detail(details, key, value)
    _add_usage_detail(
        details,
        "output_audio_tokens",
        first_present(_mapping_or_empty(completion_details), "audio_tokens", "audioTokens"),
    )

    _add_usage_detail(
        details,
        "reasoning_tokens",
        first_present(_mapping_or_empty(output_details), "reasoning_tokens", "reasoningTokens"),
    )
    _add_usage_detail(
        details, "output_audio_tokens", first_present(_mapping_or_empty(output_details), "audio_tokens", "audioTokens")
    )


def _item_value(item: Any, *keys: str) -> Any:
    if is_mapping(item):
        return first_present(item, *keys)
    for key in keys:
        value = getattr(item, key, None)
        if value is not None:
            return value
    return None


def _add_gemini_modality_details(details: dict[str, int], usage: AnyMapping) -> None:
    """Add Gemini modality token counts to usage details."""
    sources = (
        ("prompt_tokens_details", "promptTokensDetails", "prompt_tokens", "input_audio_tokens"),
        ("cache_tokens_details", "cacheTokensDetails", "cache_tokens", "cache_audio_read_tokens"),
        ("candidates_tokens_details", "candidatesTokensDetails", "candidates_tokens", "output_audio_tokens"),
        ("tool_use_prompt_tokens_details", "toolUsePromptTokensDetails", "tool_use_prompt_tokens", None),
    )

    for snake_key, camel_key, suffix, audio_detail_key in sources:
        value = first_present(usage, snake_key, camel_key)
        if not is_list(value):
            continue

        for item in value:
            modality = _item_value(item, "modality")
            token_count = _item_value(item, "token_count", "tokenCount")
            if not isinstance(modality, str):
                continue

            _add_usage_detail(details, f"{modality.lower()}_{suffix}", token_count)
            if audio_detail_key and modality.upper() == "AUDIO":
                _add_usage_detail(details, audio_detail_key, token_count)


def extract_usage_attributes(meta: AnyMapping, provider: str) -> dict[str, int]:
    """Extract OTel GenAI usage attributes from provider metadata.

    Args:
        meta: Haystack reply metadata or provider usage metadata.
        provider: Normalized GenAI provider name.

    Returns:
        Mapping of OTel attribute names to integer values.
    """
    usage_value = meta.get("usage")
    usage = usage_value if is_mapping(usage_value) else meta

    input_tokens = to_int(
        first_present(
            usage,
            "prompt_tokens",
            "input_tokens",
            "inputTokens",
            "prompt_token_count",
            "promptTokenCount",
        )
    )
    output_tokens = to_int(
        first_present(
            usage,
            "completion_tokens",
            "output_tokens",
            "outputTokens",
            "candidates_token_count",
            "candidatesTokenCount",
        )
    )

    cache_read_tokens = to_int(
        first_present(
            usage,
            "cache_read_input_tokens",
            "cacheReadInputTokens",
            "cache_read_tokens",
            "cached_content_token_count",
            "cachedContentTokenCount",
            "prompt_cache_hit_tokens",
            "cached_tokens",
        )
    )
    if cache_read_tokens is None:
        cache_read_tokens = to_int(
            nested_first_present(
                usage,
                ("prompt_tokens_details", "cached_tokens"),
                ("promptTokensDetails", "cachedTokens"),
                ("input_tokens_details", "cached_tokens"),
                ("inputTokensDetails", "cachedTokens"),
            )
        )
    if cache_read_tokens is None:
        cache_read_tokens = _sum_token_details(first_present(usage, "cache_tokens_details", "cacheTokensDetails"))

    cache_creation_tokens = to_int(
        first_present(
            usage,
            "cache_creation_input_tokens",
            "cacheCreationInputTokens",
            "cache_write_input_tokens",
            "cacheWriteInputTokens",
            "cache_write_tokens",
        )
    )

    # OTel requires gen_ai.usage.input_tokens to include cached input tokens.
    # OpenAI/DeepSeek/Gemini already include cache tokens in prompt/input totals;
    # Anthropic/Bedrock report read/write cache tokens separately.
    if provider in {"anthropic", "aws.bedrock"}:
        cache_total = (cache_read_tokens or 0) + (cache_creation_tokens or 0)
        if input_tokens is not None:
            input_tokens += cache_total
        elif cache_total:
            input_tokens = cache_total

    thoughts_tokens = to_int(first_present(usage, "thoughts_token_count", "thoughtsTokenCount"))
    if provider in {"gcp.gemini", "gcp.vertex_ai"} and thoughts_tokens is not None:
        output_tokens = (output_tokens or 0) + thoughts_tokens

    usage_details: dict[str, int] = {}

    explicit_details = usage.get("details")
    if is_mapping(explicit_details):
        for detail_key, detail_value in mapping_numeric_items(explicit_details):
            _add_usage_detail(usage_details, detail_key, detail_value)

    _add_usage_detail(usage_details, "cache_read_tokens", cache_read_tokens)
    _add_usage_detail(usage_details, "cache_write_tokens", cache_creation_tokens)
    _add_openai_token_detail_attrs(usage_details, usage)
    _add_gemini_modality_details(usage_details, usage)

    _add_usage_detail(
        usage_details,
        "cached_content_tokens",
        first_present(usage, "cached_content_token_count", "cachedContentTokenCount"),
    )
    _add_usage_detail(usage_details, "thoughts_tokens", thoughts_tokens)
    _add_usage_detail(
        usage_details,
        "tool_use_prompt_tokens",
        first_present(usage, "tool_use_prompt_token_count", "toolUsePromptTokenCount"),
    )

    for detail_key in (
        "input_audio_tokens",
        "output_audio_tokens",
        "cache_audio_read_tokens",
        "reasoning_tokens",
        "accepted_prediction_tokens",
        "rejected_prediction_tokens",
    ):
        _add_usage_detail(usage_details, detail_key, first_present(usage, detail_key, snake_to_lower_camel(detail_key)))

    attributes: dict[str, int] = {}
    if input_tokens is not None:
        attributes["gen_ai.usage.input_tokens"] = input_tokens
    if output_tokens is not None:
        attributes["gen_ai.usage.output_tokens"] = output_tokens
    if cache_read_tokens is not None:
        attributes["gen_ai.usage.cache_read.input_tokens"] = cache_read_tokens
    if cache_creation_tokens is not None:
        attributes["gen_ai.usage.cache_creation.input_tokens"] = cache_creation_tokens
    for detail_key, detail_value in sorted(usage_details.items()):
        attributes[f"gen_ai.usage.details.{detail_key}"] = detail_value
    return attributes
