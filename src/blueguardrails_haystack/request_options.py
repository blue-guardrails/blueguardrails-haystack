# SPDX-FileCopyrightText: 2025-present Blue Guardrails
#
# SPDX-License-Identifier: Apache-2.0

"""Extract and map generation request options to OTel GenAI attributes."""

from collections.abc import Callable, Iterator
from typing import Any

from blueguardrails_haystack._utils import AnyMapping, first_present, is_mapping, is_non_string_iterable

REQUEST_OPTION_FIELDS = (
    "temperature",
    "top_p",
    "topP",
    "top_k",
    "topK",
    "candidate_count",
    "candidateCount",
    "max_tokens",
    "max_completion_tokens",
    "max_output_tokens",
    "maxTokens",
    "stop",
    "stop_sequences",
    "stopSequences",
    "presence_penalty",
    "frequency_penalty",
    "seed",
    "stream",
)

REQUEST_OPTION_ATTRS: tuple[tuple[tuple[str, ...], str, Callable[[Any], Any]], ...] = (
    (("temperature",), "gen_ai.request.temperature", float),
    (("top_p", "topP"), "gen_ai.request.top_p", float),
    (("top_k", "topK"), "gen_ai.request.top_k", float),
    (("max_tokens", "max_completion_tokens", "max_output_tokens", "maxTokens"), "gen_ai.request.max_tokens", int),
    (("presence_penalty",), "gen_ai.request.presence_penalty", float),
    (("frequency_penalty",), "gen_ai.request.frequency_penalty", float),
    (("n", "candidate_count", "candidateCount"), "gen_ai.request.choice.count", int),
    (("seed",), "gen_ai.request.seed", int),
)


def _string_keyed_copy(mapping: AnyMapping) -> dict[str, Any]:
    """Return a plain dict containing only string-keyed options."""
    return {key: item_value for key, item_value in mapping.items() if isinstance(key, str)}


def options_from_object(value: Any) -> dict[str, Any]:
    """Extract known generation options from a mapping or provider config object."""
    if is_mapping(value):
        return _string_keyed_copy(value)

    options: dict[str, Any] = {}
    for attr in REQUEST_OPTION_FIELDS:
        option_value = getattr(value, attr, None)
        if option_value is not None:
            options[attr] = option_value
    return options


def request_options_from_input(value: AnyMapping) -> dict[str, Any]:
    """Collect generation request options from Haystack component input values."""
    request_options: dict[str, Any] = {}
    for key in ("generation_kwargs", "generation_config"):
        options = value.get(key)
        if options is not None:
            request_options.update(options_from_object(options))

    if value.get("model") is not None:
        request_options["model"] = value["model"]
    return request_options


def component_request_options(instance: Any) -> dict[str, Any]:
    """Extract initialization-time generation options from a generator instance."""
    request_options: dict[str, Any] = {}

    for attr in ("generation_kwargs", "_generation_kwargs", "kwargs"):
        value = getattr(instance, attr, None)
        if is_mapping(value):
            request_options.update(_string_keyed_copy(value))

    for attr in ("generation_config", "_generation_config"):
        value = getattr(instance, attr, None)
        if value is not None:
            request_options.update(options_from_object(value))

    max_length = getattr(instance, "max_length", None)
    if max_length is not None:
        request_options.setdefault("max_tokens", max_length)

    if (
        getattr(instance, "streaming_callback", None) is not None
        or getattr(instance, "_streaming_callback", None) is not None
    ):
        request_options.setdefault("stream", True)

    return request_options


def iter_request_option_attributes(request_options: AnyMapping) -> Iterator[tuple[str, Any]]:
    """Yield OTel request-option attributes from generation request options."""
    for keys, attr, caster in REQUEST_OPTION_ATTRS:
        value = first_present(request_options, *keys)
        if value is None:
            continue
        try:
            yield attr, caster(value)
        except (ValueError, TypeError):
            continue

    stop_sequences = first_present(request_options, "stop", "stop_sequences", "stopSequences")
    if stop_sequences is not None:
        if isinstance(stop_sequences, str):
            stop_sequences = [stop_sequences]
        if is_non_string_iterable(stop_sequences):
            yield "gen_ai.request.stop_sequences", [str(sequence) for sequence in stop_sequences]

    if "stream" in request_options:
        yield "gen_ai.request.stream", bool(request_options["stream"])
