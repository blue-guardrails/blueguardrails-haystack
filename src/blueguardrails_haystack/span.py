# SPDX-FileCopyrightText: 2025-present BlueGuardrails
#
# SPDX-License-Identifier: Apache-2.0

"""Haystack span implementations used by the BlueGuardrails sidecar tracer."""

from typing import Any, TypeGuard

from haystack import logging
from haystack.dataclasses import ChatMessage
from haystack.tracing.tracer import Span

from blueguardrails_haystack._utils import AnyMapping, is_list, is_mapping, is_non_string_iterable
from blueguardrails_haystack.request_options import iter_request_option_attributes, request_options_from_input
from blueguardrails_haystack.semconv import (
    convert_image_outputs_to_output_messages,
    convert_input_messages,
    convert_output_messages,
    convert_parts_to_input_messages,
    convert_parts_to_output_messages,
    convert_plain_text_to_input_messages,
    convert_plain_text_to_output_messages,
    convert_tool_definitions,
    extract_finish_reason,
    normalize_finish_reason,
)
from blueguardrails_haystack.usage import extract_usage_attributes

logger = logging.getLogger(__name__)


def _is_chat_message_list(value: object) -> TypeGuard[list[ChatMessage]]:
    """Return whether a dynamic value is a list of Haystack chat messages."""
    return is_list(value) and all(isinstance(item, ChatMessage) for item in value)


def _is_str_list(value: object) -> TypeGuard[list[str]]:
    """Return whether a dynamic value is a list of strings."""
    return is_list(value) and all(isinstance(item, str) for item in value)


class BlueGuardrailsSpan(Span):
    """Map Haystack span tags to GenAI attributes on an OTel span."""

    def __init__(self, otel_span: Any, is_chat: bool) -> None:
        """Initialize the span wrapper.

        Args:
            otel_span: OpenTelemetry span that receives GenAI attributes.
            is_chat: Whether this span represents a chat generator call.
        """
        self._span = otel_span
        self._is_chat = is_chat

    def set_tag(self, key: str, value: Any) -> None:
        """Map a standard Haystack tag to BlueGuardrails attributes."""
        try:
            if key in ("haystack.component.name", "haystack.component.type"):
                self._span.set_attribute(key, str(value))
            elif key == "haystack.component.model":
                self._set_request_model(str(value))
        except Exception as error:
            logger.warning("BlueGuardrails tracer skipped tag", key=key, error=repr(error))

    def set_content_tag(self, key: str, value: Any) -> None:
        """Map Haystack component input/output content to GenAI attributes."""
        if not is_mapping(value):
            return

        try:
            if key.endswith(".input"):
                self._handle_input(value)
            elif key.endswith(".output"):
                self._handle_output(value)
        except Exception as error:
            logger.warning("BlueGuardrails tracer skipped content tag", key=key, error=repr(error))

    def _get_attribute(self, key: str, default: Any = None) -> Any:
        try:
            return getattr(self._span, "attributes", {}).get(key, default)
        except Exception:
            return default

    def _set_request_model(self, model: str) -> None:
        if not model:
            return
        self._span.set_attribute("gen_ai.request.model", model)
        self._update_span_name()

    def _set_response_model(self, model: str) -> None:
        if not model:
            return
        self._span.set_attribute("gen_ai.response.model", model)
        if not self._get_attribute("gen_ai.request.model"):
            self._set_request_model(model)
        else:
            self._update_span_name()

    def _update_span_name(self) -> None:
        request_model = self._get_attribute("gen_ai.request.model")
        if not request_model:
            return
        op_name = self._get_attribute("gen_ai.operation.name", "chat" if self._is_chat else "text_completion")
        self._span.update_name(f"{op_name} {request_model}")

    def set_component_config(self, config: dict[str, Any]) -> None:
        """Capture generator configuration from component initialization."""
        model = config.get("model")
        if model:
            self._set_request_model(str(model))

        server = config.get("server")
        if is_mapping(server):
            self._set_server_attributes(server)

        request_options = config.get("request_options")
        if is_mapping(request_options):
            self._set_request_options(request_options)

        if "tools" in config:
            self._set_tool_definitions(config["tools"])

    def _set_server_attributes(self, server: AnyMapping) -> None:
        address = server.get("address")
        if address:
            self._span.set_attribute("server.address", str(address))

        port = server.get("port")
        if port is None:
            return
        try:
            self._span.set_attribute("server.port", int(port))
        except (ValueError, TypeError):
            pass

    def _set_request_options(self, request_options: AnyMapping) -> None:
        request_model = str(request_options.get("model") or "")
        if request_model:
            self._set_request_model(request_model)

        for attr, value in iter_request_option_attributes(request_options):
            self._span.set_attribute(attr, value)

    def _set_tool_definitions(self, tools: Any) -> None:
        self._span.set_attribute("gen_ai.tool.definitions", convert_tool_definitions(tools))

    def _handle_input(self, value: AnyMapping) -> None:
        request_options = request_options_from_input(value)
        self._set_request_options(request_options)

        if "streaming_callback" in value and "stream" not in request_options:
            self._span.set_attribute("gen_ai.request.stream", value["streaming_callback"] is not None)

        if "tools" in value and value["tools"] is not None:
            self._set_tool_definitions(value["tools"])

        if self._is_chat and "messages" in value:
            messages = value["messages"]
            if _is_chat_message_list(messages):
                self._span.set_attribute("gen_ai.input.messages", convert_input_messages(messages))
        elif "prompt" in value:
            self._span.set_attribute(
                "gen_ai.input.messages", convert_plain_text_to_input_messages(str(value["prompt"]))
            )
        elif "parts" in value:
            parts = value["parts"]
            if is_non_string_iterable(parts):
                self._span.set_attribute("gen_ai.input.messages", convert_parts_to_input_messages(parts))

    @staticmethod
    def _metas_from_value(value: Any) -> list[AnyMapping]:
        if is_list(value):
            return [meta for meta in value if is_mapping(meta)]
        if is_mapping(value):
            return [value]
        return []

    def _handle_output(self, value: AnyMapping) -> None:
        replies = value.get("replies")
        if replies:
            if _is_chat_message_list(replies):
                metas = [reply.meta for reply in replies if is_mapping(reply.meta)]
                self._span.set_attribute("gen_ai.output.messages", convert_output_messages(replies))
                self._finalize_output_metadata(metas)
                return
            if _is_str_list(replies):
                metas = self._metas_from_value(value.get("meta"))
                self._span.set_attribute(
                    "gen_ai.output.messages",
                    convert_plain_text_to_output_messages(replies, [extract_finish_reason(meta) for meta in metas]),
                )
                self._finalize_output_metadata(metas)
                return

        metas = self._metas_from_value(value.get("meta"))
        finish_reasons = [extract_finish_reason(meta) for meta in metas]

        images = value.get("images")
        if is_list(images) and images:
            self._span.set_attribute(
                "gen_ai.output.messages",
                convert_image_outputs_to_output_messages(images, finish_reasons),
            )
            self._span.set_attribute("gen_ai.output.type", "image")
            self._finalize_output_metadata(metas)
            return

        for key in ("files", "parts"):
            parts = value.get(key)
            if is_list(parts) and parts:
                self._span.set_attribute(
                    "gen_ai.output.messages",
                    convert_parts_to_output_messages(parts, finish_reasons[0] if finish_reasons else None),
                )
                self._finalize_output_metadata(metas)
                return

    def _finalize_output_metadata(self, metas: list[AnyMapping]) -> None:
        self._set_finish_reasons(metas)
        if metas:
            self._set_response_id(metas[0])
            self._set_response_metadata(metas[0])
        self._ensure_response_model_from_request()

    def _ensure_response_model_from_request(self) -> None:
        if not self._get_attribute("gen_ai.response.model"):
            request_model = self._get_attribute("gen_ai.request.model")
            if request_model:
                self._span.set_attribute("gen_ai.response.model", str(request_model))

    def _set_finish_reasons(self, metas: list[AnyMapping]) -> None:
        finish_reasons = [
            finish_reason
            for meta in metas
            if (finish_reason := normalize_finish_reason(extract_finish_reason(meta))) is not None
        ]
        if finish_reasons:
            self._span.set_attribute("gen_ai.response.finish_reasons", finish_reasons)

    def _set_response_id(self, meta: AnyMapping) -> None:
        response_id = meta.get("id") or meta.get("response_id")
        if response_id is not None:
            self._span.set_attribute("gen_ai.response.id", str(response_id))

    def _set_response_metadata(self, meta: AnyMapping) -> None:
        model = meta.get("model") or meta.get("model_id") or meta.get("modelId")
        if model:
            self._set_response_model(str(model))

        provider = str(self._get_attribute("gen_ai.provider.name", ""))
        for attr, value in extract_usage_attributes(meta, provider).items():
            self._span.set_attribute(attr, value)

    def raw_span(self) -> Any:
        """Return the wrapped OpenTelemetry span."""
        return self._span

    def get_correlation_data_for_logs(self) -> dict[str, Any]:
        """Return log-correlation data for Haystack's tracing API."""
        return {}


class CompositeSpan(Span):
    """Forward span operations to the user's span and the BlueGuardrails span."""

    def __init__(self, original: Span, blueguardrails: Span) -> None:
        """Initialize a span that forwards to user and BlueGuardrails spans.

        Args:
            original: User-configured Haystack span.
            blueguardrails: BlueGuardrails span.
        """
        self._original = original
        self._blueguardrails = blueguardrails

    @property
    def original(self) -> Span:
        """Return the wrapped user span."""
        return self._original

    @property
    def blueguardrails(self) -> Span:
        """Return the wrapped BlueGuardrails span."""
        return self._blueguardrails

    def set_tag(self, key: str, value: Any) -> None:
        """Forward a standard Haystack tag to both spans."""
        self._original.set_tag(key, value)
        self._blueguardrails.set_tag(key, value)

    def set_content_tag(self, key: str, value: Any) -> None:
        """Forward a Haystack content tag to both spans."""
        self._original.set_content_tag(key, value)
        self._blueguardrails.set_content_tag(key, value)

    def raw_span(self) -> Any:
        """Return the user's raw span."""
        return self._original.raw_span()

    def get_correlation_data_for_logs(self) -> dict[str, Any]:
        """Return the user's log-correlation data."""
        return self._original.get_correlation_data_for_logs()
