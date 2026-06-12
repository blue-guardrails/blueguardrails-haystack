# SPDX-FileCopyrightText: 2025-present Blue Guardrails
#
# SPDX-License-Identifier: Apache-2.0

"""Normalize provider usage metadata to OTel GenAI usage attributes."""

from typing import Any

from blueguardrails_haystack._utils import (
    AnyMapping,
    first_present,
    is_list,
    is_mapping,
    nested_first_present,
    to_int,
)


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


def _extract_reasoning_tokens(usage: AnyMapping) -> int | None:
    """Extract output reasoning token counts from known provider usage shapes."""
    reasoning_tokens = to_int(
        first_present(
            usage,
            "reasoning_output_tokens",
            "reasoningOutputTokens",
            "reasoning_tokens",
            "reasoningTokens",
            "thoughts_token_count",
            "thoughtsTokenCount",
        )
    )
    if reasoning_tokens is not None:
        return reasoning_tokens

    return to_int(
        nested_first_present(
            usage,
            ("completion_tokens_details", "reasoning_tokens"),
            ("completionTokensDetails", "reasoningTokens"),
            ("output_tokens_details", "reasoning_tokens"),
            ("outputTokensDetails", "reasoningTokens"),
            ("details", "reasoning_tokens"),
            ("details", "reasoningTokens"),
            ("details", "reasoning_output_tokens"),
            ("details", "reasoningOutputTokens"),
        )
    )


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

    reasoning_tokens = _extract_reasoning_tokens(usage)

    # Gemini reports thoughts separately from candidate tokens. OTel requires reasoning tokens
    # to be included in gen_ai.usage.output_tokens.
    if provider in {"gcp.gemini", "gcp.vertex_ai"} and reasoning_tokens is not None:
        output_tokens = (output_tokens or 0) + reasoning_tokens
    elif output_tokens is None and reasoning_tokens:
        output_tokens = reasoning_tokens

    attributes: dict[str, int] = {}
    if input_tokens is not None:
        attributes["gen_ai.usage.input_tokens"] = input_tokens
    if output_tokens is not None:
        attributes["gen_ai.usage.output_tokens"] = output_tokens
    if cache_read_tokens is not None:
        attributes["gen_ai.usage.cache_read.input_tokens"] = cache_read_tokens
    if cache_creation_tokens is not None:
        attributes["gen_ai.usage.cache_creation.input_tokens"] = cache_creation_tokens
    if reasoning_tokens:
        attributes["gen_ai.usage.reasoning.output_tokens"] = reasoning_tokens
    return attributes
