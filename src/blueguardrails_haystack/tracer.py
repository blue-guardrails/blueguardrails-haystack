# SPDX-FileCopyrightText: 2025-present Blue Guardrails
#
# SPDX-License-Identifier: Apache-2.0

"""Public tracing API for the Blue Guardrails Haystack sidecar."""

import contextlib
import uuid
from collections.abc import Iterator
from contextvars import ContextVar
from typing import Any

from opentelemetry.context import Context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import SpanKind, StatusCode

from haystack.tracing import Span, Tracer
from haystack.tracing.tracer import NullSpan

from blueguardrails_haystack._utils import (
    first_present as _first_present,
    mapping_numeric_items as _mapping_numeric_items,
    nested_first_present as _nested_first_present,
    snake_to_lower_camel as _snake_to_lower_camel,
    to_int as _to_int,
)
from blueguardrails_haystack.component_config import (
    _extract_component_config,
    _extract_component_model,
    _extract_component_request_options,
    _extract_component_server,
    _extract_component_tools,
    _extract_options_from_object,
    _extract_server_from_object,
    _server_attrs_from_url,
    extract_component_config,
)
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
    infer_provider_name,
    normalize_finish_reason,
)
from blueguardrails_haystack.span import BGSpan, CompositeSpan

_PIPELINE_RUN_OPERATIONS = frozenset({"haystack.pipeline.run", "haystack.async_pipeline.run", "haystack.agent.run"})
_RUN_ID_TAG_ATTRIBUTE = "gen_ai.agent.run.tags.pipeline_run_id"
_CONVERSATION_TAG_ATTRIBUTE_PREFIX = "gen_ai.conversation.tags."
_HAYSTACK_COMPONENT_NAME_CONVERSATION_TAG = f"{_CONVERSATION_TAG_ATTRIBUTE_PREFIX}haystack_component_name"

_run_id_var: ContextVar[str | None] = ContextVar("bg_run_id", default=None)


class BGTracer(Tracer):
    """Create OTel spans for Haystack generator components."""

    def __init__(self, provider: TracerProvider, conversation_tags: dict[str, str] | None = None) -> None:
        self._tracer = provider.get_tracer("blueguardrails-haystack")
        self._conversation_tags = conversation_tags or {}

    @contextlib.contextmanager
    def trace(
        self, operation_name: str, tags: dict[str, Any] | None = None, parent_span: Span | None = None
    ) -> Iterator[Span]:
        """Trace a Haystack operation when it is a generator run."""
        tags = tags or {}
        component_type = tags.get("haystack.component.type", "")

        if operation_name in _PIPELINE_RUN_OPERATIONS:
            token = _run_id_var.set(str(uuid.uuid4()))
            try:
                yield NullSpan()
            finally:
                _run_id_var.reset(token)
            return

        if not (component_type and component_type.endswith("Generator")):
            yield NullSpan()
            return

        is_chat = component_type.endswith("ChatGenerator")
        op_name = "chat" if is_chat else "text_completion"
        component_name = str(tags.get("haystack.component.name", "unknown") or "unknown")
        provider_name = infer_provider_name(component_type)

        # Start a root span so Blue Guardrails does not change the user's active OTel context.
        otel_span = self._tracer.start_span(name=f"{op_name} {component_name}", kind=SpanKind.CLIENT, context=Context())
        try:
            otel_span.set_attribute("gen_ai.operation.name", op_name)
            otel_span.set_attribute("gen_ai.provider.name", provider_name)

            run_id = _run_id_var.get()
            if run_id:
                otel_span.set_attribute(_RUN_ID_TAG_ATTRIBUTE, run_id)

            for key, value in self._conversation_tags.items():
                otel_span.set_attribute(f"{_CONVERSATION_TAG_ATTRIBUTE_PREFIX}{key}", str(value))

            otel_span.set_attribute(_HAYSTACK_COMPONENT_NAME_CONVERSATION_TAG, component_name)

            span = BGSpan(otel_span, is_chat)
            if tags:
                span.set_tags(tags)

            try:
                yield span
            except Exception as error:
                otel_span.set_status(StatusCode.ERROR)
                otel_span.set_attribute("error.type", type(error).__name__)
                raise
        finally:
            otel_span.end()

    def current_span(self) -> Span | None:
        return None


from blueguardrails_haystack.proxy import (  # noqa: E402
    _BGSidecarProxy,
    _enter_bg_trace,
    _exit_bg_trace,
    _patch_pipeline_component_span_for_models,
    _set_bg_component_config,
    install_bg_tracer,
)

__all__ = [
    "BGSpan",
    "BGTracer",
    "CompositeSpan",
    "_BGSidecarProxy",
    "_enter_bg_trace",
    "_exit_bg_trace",
    "_extract_component_config",
    "_extract_component_model",
    "_extract_component_request_options",
    "_extract_component_server",
    "_extract_component_tools",
    "_extract_options_from_object",
    "_extract_server_from_object",
    "_first_present",
    "_mapping_numeric_items",
    "_nested_first_present",
    "_patch_pipeline_component_span_for_models",
    "_server_attrs_from_url",
    "_set_bg_component_config",
    "_snake_to_lower_camel",
    "_to_int",
    "convert_image_outputs_to_output_messages",
    "convert_input_messages",
    "convert_output_messages",
    "convert_parts_to_input_messages",
    "convert_parts_to_output_messages",
    "convert_plain_text_to_input_messages",
    "convert_plain_text_to_output_messages",
    "convert_tool_definitions",
    "extract_component_config",
    "extract_finish_reason",
    "infer_provider_name",
    "install_bg_tracer",
    "normalize_finish_reason",
]
