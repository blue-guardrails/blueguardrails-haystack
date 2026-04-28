# SPDX-FileCopyrightText: 2025-present Blue Guardrails
#
# SPDX-License-Identifier: Apache-2.0

"""Public tracing API for the Blue Guardrails Haystack sidecar."""

import contextlib
import os
import uuid
from collections.abc import Generator
from contextvars import ContextVar
from typing import Any

from haystack.tracing.tracer import NullSpan, Span, Tracer
from haystack.utils import Secret
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
from opentelemetry.trace import SpanKind, StatusCode

from blueguardrails_haystack.semconv import infer_provider_name
from blueguardrails_haystack.span import BlueGuardrailsSpan

_DEFAULT_ENDPOINT = "https://api.blueguardrails.com/v1/traces"
_DEFAULT_SERVICE_NAME = "blueguardrails-haystack"
_DEFAULT_TRACE_NAME = "blueguardrails-haystack"
_PIPELINE_RUN_OPERATIONS = frozenset({"haystack.pipeline.run", "haystack.async_pipeline.run", "haystack.agent.run"})
_RUN_ID_TAG_ATTRIBUTE = "gen_ai.agent.run.tags.pipeline_run_id"
_CONVERSATION_TAG_ATTRIBUTE_PREFIX = "gen_ai.conversation.tags."
_HAYSTACK_COMPONENT_NAME_CONVERSATION_TAG = f"{_CONVERSATION_TAG_ATTRIBUTE_PREFIX}haystack_component_name"

_run_id_var: ContextVar[str | None] = ContextVar("blueguardrails_run_id", default=None)


class BlueGuardrailsTracer(Tracer):
    """Create OTel spans for Haystack generator components."""

    def __init__(self, provider: TracerProvider, conversation_tags: dict[str, str] | None = None) -> None:
        """Initialize the tracer.

        Args:
            provider: OpenTelemetry tracer provider used for Blue Guardrails spans.
            conversation_tags: Tags to attach to every exported GenAI span.
        """
        self._tracer = provider.get_tracer("blueguardrails-haystack")
        self._conversation_tags = conversation_tags or {}

    @contextlib.contextmanager
    def trace(
        self, operation_name: str, tags: dict[str, Any] | None = None, parent_span: Span | None = None
    ) -> Generator[Span]:
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

            span = BlueGuardrailsSpan(otel_span, is_chat)
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
        """Return the current span for Haystack's tracing API."""
        return None


def _resolve_api_key(api_key: str | Secret | None) -> str:
    """Resolve an API key from an explicit value or ``BLUEGUARDRAILS_API_KEY``."""
    if api_key is None:
        resolved_key = os.getenv("BLUEGUARDRAILS_API_KEY")
    elif isinstance(api_key, Secret):
        resolved_key = api_key.resolve_value()
    else:
        resolved_key = api_key

    if not resolved_key or not resolved_key.strip():
        raise ValueError(
            "Blue Guardrails API key is required. Set BLUEGUARDRAILS_API_KEY or pass api_key to "
            "configure_blueguardrails_tracer()."
        )
    return resolved_key.strip()


def create_blueguardrails_tracer(
    *,
    name: str,
    endpoint: str,
    api_key: str | Secret | None,
    sample_rate: float,
    tags: dict[str, str] | None,
) -> BlueGuardrailsTracer:
    """Create a Blue Guardrails tracer with the default OTLP exporter."""
    if not 0.0 <= sample_rate <= 1.0:
        raise ValueError("sample_rate must be between 0.0 and 1.0")

    resolved_key = _resolve_api_key(api_key)
    resource = Resource.create(
        {
            "service.name": _DEFAULT_SERVICE_NAME,
            "haystack.pipeline.name": name,
        }
    )
    provider = TracerProvider(
        sampler=TraceIdRatioBased(sample_rate),
        resource=resource,
    )
    exporter = OTLPSpanExporter(endpoint=endpoint, headers={"Authorization": f"Bearer {resolved_key}"})
    provider.add_span_processor(BatchSpanProcessor(exporter))
    return BlueGuardrailsTracer(provider, conversation_tags=tags)
