# SPDX-FileCopyrightText: 2025-present Blue Guardrails
#
# SPDX-License-Identifier: Apache-2.0

"""Haystack span implementations used by the Blue Guardrails sidecar tracer."""

from collections.abc import Iterable
from typing import Any

from haystack import logging
from haystack.dataclasses import ChatMessage
from haystack.tracing import Span

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


class BGSpan(Span):
    """Map Haystack span tags to GenAI attributes on an OTel span."""

    def __init__(self, otel_span: Any, is_chat: bool) -> None:
        self._span = otel_span
        self._is_chat = is_chat

    def set_tag(self, key: str, value: Any) -> None:
        try:
            if key in ("haystack.component.name", "haystack.component.type"):
                self._span.set_attribute(key, str(value))
            elif key == "haystack.component.model":
                self._set_request_model(str(value))
        except Exception as error:
            logger.warning("Blue Guardrails tracer skipped tag", key=key, error=repr(error))

    def set_content_tag(self, key: str, value: Any) -> None:
        """Map Haystack component input/output content to GenAI attributes."""
        if not isinstance(value, dict):
            return

        try:
            if key.endswith(".input"):
                self._handle_input(value)
            elif key.endswith(".output"):
                self._handle_output(value)
        except Exception as error:
            logger.warning("Blue Guardrails tracer skipped content tag", key=key, error=repr(error))

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
        if isinstance(server, dict):
            self._set_server_attributes(server)

        request_options = config.get("request_options")
        if isinstance(request_options, dict):
            self._set_request_options(request_options)

        if "tools" in config:
            self._set_tool_definitions(config["tools"])

    def _set_server_attributes(self, server: dict[str, Any]) -> None:
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

    def _set_request_options(self, request_options: dict[str, Any]) -> None:
        request_model = str(request_options.get("model") or "")
        if request_model:
            self._set_request_model(request_model)

        for attr, value in iter_request_option_attributes(request_options):
            self._span.set_attribute(attr, value)

    def _set_tool_definitions(self, tools: Any) -> None:
        self._span.set_attribute("gen_ai.tool.definitions", convert_tool_definitions(tools))

    def _handle_input(self, value: dict[str, Any]) -> None:
        request_options = request_options_from_input(value)
        self._set_request_options(request_options)

        if "streaming_callback" in value and "stream" not in request_options:
            self._span.set_attribute("gen_ai.request.stream", value["streaming_callback"] is not None)

        if "tools" in value and value["tools"] is not None:
            self._set_tool_definitions(value["tools"])

        if self._is_chat and "messages" in value:
            messages = value["messages"]
            if isinstance(messages, list) and all(isinstance(message, ChatMessage) for message in messages):
                self._span.set_attribute("gen_ai.input.messages", convert_input_messages(messages))
        elif "prompt" in value:
            self._span.set_attribute("gen_ai.input.messages", convert_plain_text_to_input_messages(str(value["prompt"])))
        elif "parts" in value:
            parts = value["parts"]
            if isinstance(parts, Iterable) and not isinstance(parts, (str, bytes)):
                self._span.set_attribute("gen_ai.input.messages", convert_parts_to_input_messages(parts))

    @staticmethod
    def _metas_from_value(value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [meta for meta in value if isinstance(meta, dict)]
        if isinstance(value, dict):
            return [value]
        return []

    def _handle_output(self, value: dict[str, Any]) -> None:
        replies = value.get("replies")
        if isinstance(replies, list) and replies:
            if all(isinstance(reply, ChatMessage) for reply in replies):
                metas = [reply.meta for reply in replies if isinstance(reply.meta, dict)]
                self._span.set_attribute("gen_ai.output.messages", convert_output_messages(replies))
                self._finalize_output_metadata(metas)
                return
            if all(isinstance(reply, str) for reply in replies):
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
        if isinstance(images, list) and images:
            self._span.set_attribute("gen_ai.output.messages", convert_image_outputs_to_output_messages(images, finish_reasons))
            self._span.set_attribute("gen_ai.output.type", "image")
            self._finalize_output_metadata(metas)
            return

        for key in ("files", "parts"):
            parts = value.get(key)
            if isinstance(parts, list) and parts:
                self._span.set_attribute(
                    "gen_ai.output.messages",
                    convert_parts_to_output_messages(parts, finish_reasons[0] if finish_reasons else None),
                )
                self._finalize_output_metadata(metas)
                return

    def _finalize_output_metadata(self, metas: list[dict[str, Any]]) -> None:
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

    def _set_finish_reasons(self, metas: list[dict[str, Any]]) -> None:
        finish_reasons = [
            finish_reason
            for meta in metas
            if (finish_reason := normalize_finish_reason(extract_finish_reason(meta))) is not None
        ]
        if finish_reasons:
            self._span.set_attribute("gen_ai.response.finish_reasons", finish_reasons)

    def _set_response_id(self, meta: dict[str, Any]) -> None:
        response_id = meta.get("id") or meta.get("response_id")
        if response_id is not None:
            self._span.set_attribute("gen_ai.response.id", str(response_id))

    def _set_response_metadata(self, meta: dict[str, Any]) -> None:
        model = meta.get("model") or meta.get("model_id") or meta.get("modelId")
        if model:
            self._set_response_model(str(model))

        provider = str(self._get_attribute("gen_ai.provider.name", ""))
        for attr, value in extract_usage_attributes(meta, provider).items():
            self._span.set_attribute(attr, value)

    def raw_span(self) -> Any:
        return self._span

    def get_correlation_data_for_logs(self) -> dict[str, Any]:
        return {}


class CompositeSpan(Span):
    """Forward span operations to the user's span and the Blue Guardrails span."""

    def __init__(self, original: Span, bg: Span) -> None:
        self._original = original
        self._bg = bg

    def set_tag(self, key: str, value: Any) -> None:
        self._original.set_tag(key, value)
        self._bg.set_tag(key, value)

    def set_content_tag(self, key: str, value: Any) -> None:
        self._original.set_content_tag(key, value)
        self._bg.set_content_tag(key, value)

    def raw_span(self) -> Any:
        return self._original.raw_span()

    def get_correlation_data_for_logs(self) -> dict[str, Any]:
        return self._original.get_correlation_data_for_logs()
