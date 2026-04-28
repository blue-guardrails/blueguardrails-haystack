# SPDX-FileCopyrightText: 2025-present BlueGuardrails
#
# SPDX-License-Identifier: Apache-2.0

"""Integration tests: BlueGuardrails tracer alongside existing Haystack tracers.

Verifies that the sidecar proxy correctly multiplexes spans so:
- The user's existing tracing backend gets all standard Haystack spans
- BlueGuardrails gets only Generator spans with GenAI semconv attributes
- Neither tracer interferes with the other
- Changing the user tracer after BlueGuardrails install doesn't break anything

Covers both OpenTelemetry and Datadog-style (context-managed) tracers.
"""

import contextlib
import json
import os
from collections.abc import Iterator
from typing import Any

import opentelemetry.trace
from conftest import reset_haystack_tracing_state
from haystack import Pipeline, component, tracing
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool
from haystack.tracing import OpenTelemetryTracer, Span, Tracer, utils as tracing_utils
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from blueguardrails_haystack.proxy import configure_blueguardrails_tracer
from blueguardrails_haystack.tracer import BlueGuardrailsTracer


@component
class MockChatGenerator:
    """A mock ChatGenerator that returns predefined replies with metadata."""

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage]) -> dict:
        reply = ChatMessage.from_assistant(
            "Hello! I'm a mock response.",
            meta={
                "model": "gpt-4o-mock",
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 15, "completion_tokens": 8},
            },
        )
        return {"replies": [reply]}


@component
class MockPlainGenerator:
    """A mock non-chat Generator with a configured model but no output metadata."""

    def __init__(self) -> None:
        self.model = "plain-mock-model"

    @component.output_types(replies=list[str])
    def run(self, prompt: str) -> dict:
        return {"replies": [f"Plain response to: {prompt}"]}


@component
class MockImageGenerator:
    """A mock image generator returning a URL output."""

    def __init__(self) -> None:
        self.model = "image-mock-model"

    @component.output_types(images=list[str], revised_prompt=str)
    def run(self, prompt: str) -> dict:
        return {"images": ["https://example.com/generated.png"], "revised_prompt": prompt}


@component
class MockInitConfigChatGenerator:
    """A mock ChatGenerator with init-time generation kwargs and tools."""

    def __init__(self) -> None:
        self.model = "init-config-model"
        self.generation_kwargs = {"temperature": 0.25, "max_tokens": 17}
        self.tools = [
            Tool(
                name="lookup",
                description="Look up a value",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                function=lambda query: query,
            )
        ]

    @component.output_types(replies=list[ChatMessage])
    def run(
        self,
        messages: list[ChatMessage],
        generation_kwargs: dict[str, Any] | None = None,
        tools: list[Tool] | None = None,
    ) -> dict:
        return {"replies": [ChatMessage.from_assistant("init config response", meta={"model": self.model})]}


@component
class MockEndpointChatGenerator:
    """A mock ChatGenerator with an init-time provider endpoint."""

    def __init__(self) -> None:
        self.model = "endpoint-model"
        self.api_base_url = "https://llm-proxy.example.com:9443/v1"

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage]) -> dict:
        return {"replies": [ChatMessage.from_assistant("endpoint response", meta={"model": self.model})]}


def _make_blueguardrails_tracer():
    """Create a BlueGuardrails tracer with an in-memory exporter.

    Returns:
        BlueGuardrails tracer and in-memory exporter.
    """
    blueguardrails_exporter = InMemorySpanExporter()
    blueguardrails_provider = TracerProvider(resource=Resource.create({"service.name": "blueguardrails"}))
    blueguardrails_provider.add_span_processor(SimpleSpanProcessor(blueguardrails_exporter))
    blueguardrails_tracer = BlueGuardrailsTracer(blueguardrails_provider)
    return blueguardrails_tracer, blueguardrails_exporter


def _make_user_otel_tracer():
    """Create a user OTel tracer with an in-memory exporter.

    Returns:
        User tracer, in-memory exporter, and tracer provider.
    """
    user_exporter = InMemorySpanExporter()
    user_provider = TracerProvider(resource=Resource.create({"service.name": "user-app"}))
    user_provider.add_span_processor(SimpleSpanProcessor(user_exporter))
    user_otel_tracer = OpenTelemetryTracer(user_provider.get_tracer("haystack"))
    return user_otel_tracer, user_exporter, user_provider


def _setup_tracers():
    """Install user OTel tracing and BlueGuardrails tracing.

    Returns:
        User exporter, BlueGuardrails exporter, and user tracer provider.
    """
    user_otel_tracer, user_exporter, user_provider = _make_user_otel_tracer()
    tracing.enable_tracing(user_otel_tracer)

    blueguardrails_tracer, blueguardrails_exporter = _make_blueguardrails_tracer()
    configure_blueguardrails_tracer(blueguardrails_tracer)

    return user_exporter, blueguardrails_exporter, user_provider


class TestOtelCoexistence:
    def setup_method(self):
        os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"
        reset_haystack_tracing_state()

    def teardown_method(self):
        reset_haystack_tracing_state()
        os.environ.pop("HAYSTACK_CONTENT_TRACING_ENABLED", None)

    def test_both_tracers_receive_correct_spans(self):
        user_exporter, blueguardrails_exporter, _ = _setup_tracers()

        pipe = Pipeline()
        pipe.add_component("llm", MockChatGenerator())
        result = pipe.run({"llm": {"messages": [ChatMessage.from_user("Hi")]}})

        assert result["llm"]["replies"][0].text == "Hello! I'm a mock response."

        # User's OTel: gets pipeline.run + component spans
        user_spans = user_exporter.get_finished_spans()
        user_span_names = [s.name for s in user_spans]
        assert len(user_spans) >= 2, f"Expected >=2 user spans, got: {user_span_names}"

        # BlueGuardrails: gets exactly 1 generator span with GenAI semconv
        blueguardrails_spans = blueguardrails_exporter.get_finished_spans()
        assert len(blueguardrails_spans) == 1, (
            f"Expected 1 BlueGuardrails span, got {len(blueguardrails_spans)}: {[s.name for s in blueguardrails_spans]}"
        )

        blueguardrails_span = blueguardrails_spans[0]
        assert blueguardrails_span.attributes["gen_ai.operation.name"] == "chat"
        assert blueguardrails_span.attributes["gen_ai.response.model"] == "gpt-4o-mock"
        assert blueguardrails_span.attributes["gen_ai.usage.input_tokens"] == 15
        assert blueguardrails_span.attributes["gen_ai.usage.output_tokens"] == 8

        input_msgs = json.loads(blueguardrails_span.attributes["gen_ai.input.messages"])
        assert input_msgs[0]["role"] == "user"

        output_msgs = json.loads(blueguardrails_span.attributes["gen_ai.output.messages"])
        assert output_msgs[0]["role"] == "assistant"

    def test_user_otel_spans_not_polluted_with_genai_semconv(self):
        """BlueGuardrails GenAI attributes must not leak into user's OTel spans."""
        user_exporter, _, _ = _setup_tracers()

        pipe = Pipeline()
        pipe.add_component("llm", MockChatGenerator())
        pipe.run({"llm": {"messages": [ChatMessage.from_user("Hi")]}})

        for span in user_exporter.get_finished_spans():
            for attr_key in span.attributes:
                assert not attr_key.startswith("gen_ai."), (
                    f"User's OTel span '{span.name}' has unexpected GenAI attribute '{attr_key}'"
                )

    def test_blueguardrails_only_captures_generator_spans(self):
        """BlueGuardrails must not create spans for non-generator components."""
        user_exporter, blueguardrails_exporter, _ = _setup_tracers()

        pipe = Pipeline()
        pipe.add_component("prompt_builder", ChatPromptBuilder())
        pipe.add_component("llm", MockChatGenerator())
        pipe.connect("prompt_builder.prompt", "llm.messages")

        messages = [ChatMessage.from_user("Tell me about {{topic}}")]
        pipe.run(
            {
                "prompt_builder": {
                    "template": messages,
                    "template_variables": {"topic": "testing"},
                }
            }
        )

        # User's OTel: pipeline.run + prompt_builder + llm = at least 3 spans
        user_spans = user_exporter.get_finished_spans()
        assert len(user_spans) >= 3, f"Expected >=3 user spans, got: {[s.name for s in user_spans]}"

        # BlueGuardrails: only the generator
        blueguardrails_spans = blueguardrails_exporter.get_finished_spans()
        assert len(blueguardrails_spans) == 1
        assert blueguardrails_spans[0].name == "chat gpt-4o-mock"

    def test_blueguardrails_captures_content_even_when_content_tracing_disabled(self):
        """BlueGuardrails must capture input/output even if HAYSTACK_CONTENT_TRACING_ENABLED is false."""
        os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "false"
        reset_haystack_tracing_state()
        user_exporter, blueguardrails_exporter, _ = _setup_tracers()

        pipe = Pipeline()
        pipe.add_component("llm", MockChatGenerator())
        pipe.run({"llm": {"messages": [ChatMessage.from_user("secret input")]}})

        blueguardrails_spans = blueguardrails_exporter.get_finished_spans()
        assert len(blueguardrails_spans) == 1

        # BlueGuardrails still captures content
        assert "gen_ai.input.messages" in blueguardrails_spans[0].attributes
        assert "gen_ai.output.messages" in blueguardrails_spans[0].attributes

        # User's OTel should not have content tags because content tracing is off.
        content_keys = {"haystack.component.input", "haystack.component.output"}
        for span in user_exporter.get_finished_spans():
            for attr_key in span.attributes:
                assert attr_key not in content_keys, (
                    f"User span '{span.name}' has content attribute '{attr_key}' despite content tracing being off"
                )

    def test_pipeline_run_id_correlation(self):
        """BlueGuardrails exposes the per-run correlation ID as the pipeline_run_id agent run tag."""
        _, blueguardrails_exporter, _ = _setup_tracers()

        pipe = Pipeline()
        pipe.add_component("llm", MockChatGenerator())
        pipe.run({"llm": {"messages": [ChatMessage.from_user("Hi")]}})

        blueguardrails_spans = blueguardrails_exporter.get_finished_spans()
        assert len(blueguardrails_spans) == 1
        assert "gen_ai.agent.run.tags.pipeline_run_id" in blueguardrails_spans[0].attributes
        assert "haystack.pipeline.run_id" not in blueguardrails_spans[0].attributes

    def test_component_name_is_added_as_conversation_tag(self):
        """Each generator span gets a component-name conversation tag."""
        _, blueguardrails_exporter, _ = _setup_tracers()

        pipe = Pipeline()
        pipe.add_component("first_llm", MockChatGenerator())
        pipe.add_component("second_llm", MockChatGenerator())
        pipe.run(
            {
                "first_llm": {"messages": [ChatMessage.from_user("Hi first")]},
                "second_llm": {"messages": [ChatMessage.from_user("Hi second")]},
            }
        )

        blueguardrails_spans = blueguardrails_exporter.get_finished_spans()
        assert len(blueguardrails_spans) == 2
        attrs_by_component = {
            span.attributes["haystack.component.name"]: span.attributes for span in blueguardrails_spans
        }
        for component_name in ("first_llm", "second_llm"):
            assert (
                attrs_by_component[component_name]["gen_ai.conversation.tags.haystack_component_name"] == component_name
            )

    def test_blueguardrails_extracts_request_model_from_generator_instance(self):
        """BlueGuardrails captures request/response model even when a generator output has no metadata."""
        _, blueguardrails_exporter, _ = _setup_tracers()

        pipe = Pipeline()
        pipe.add_component("gen", MockPlainGenerator())
        pipe.run({"gen": {"prompt": "Hi"}})

        blueguardrails_spans = blueguardrails_exporter.get_finished_spans()
        assert len(blueguardrails_spans) == 1
        attrs = blueguardrails_spans[0].attributes
        assert blueguardrails_spans[0].name == "text_completion plain-mock-model"
        assert attrs["gen_ai.request.model"] == "plain-mock-model"
        assert attrs["gen_ai.response.model"] == "plain-mock-model"
        assert json.loads(attrs["gen_ai.input.messages"])[0]["parts"][0]["content"] == "Hi"
        assert json.loads(attrs["gen_ai.output.messages"])[0]["parts"][0]["content"] == "Plain response to: Hi"

    def test_blueguardrails_captures_image_outputs(self):
        """BlueGuardrails captures multimodal image outputs as GenAI semconv uri/blob parts."""
        _, blueguardrails_exporter, _ = _setup_tracers()

        pipe = Pipeline()
        pipe.add_component("image_gen", MockImageGenerator())
        pipe.run({"image_gen": {"prompt": "Draw a blue square"}})

        span = blueguardrails_exporter.get_finished_spans()[0]
        attrs = span.attributes
        assert attrs["gen_ai.output.type"] == "image"
        output_messages = json.loads(attrs["gen_ai.output.messages"])
        assert output_messages == [
            {
                "role": "assistant",
                "parts": [{"type": "uri", "modality": "image", "uri": "https://example.com/generated.png"}],
            }
        ]

    def test_blueguardrails_captures_init_generation_kwargs_and_tools(self):
        """BlueGuardrails captures generator init-time generation_kwargs and tool definitions."""
        _, blueguardrails_exporter, _ = _setup_tracers()

        pipe = Pipeline()
        pipe.add_component("llm", MockInitConfigChatGenerator())
        pipe.run({"llm": {"messages": [ChatMessage.from_user("Hi")]}})

        blueguardrails_spans = blueguardrails_exporter.get_finished_spans()
        assert len(blueguardrails_spans) == 1
        attrs = blueguardrails_spans[0].attributes
        assert attrs["gen_ai.request.temperature"] == 0.25
        assert attrs["gen_ai.request.max_tokens"] == 17

        tool_defs = json.loads(attrs["gen_ai.tool.definitions"])
        assert tool_defs == [
            {
                "type": "function",
                "name": "lookup",
                "description": "Look up a value",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ]

    def test_blueguardrails_captures_server_address_and_port_from_generator_endpoint(self):
        """BlueGuardrails captures provider endpoint host/port from generator init-time config."""
        _, blueguardrails_exporter, _ = _setup_tracers()

        pipe = Pipeline()
        pipe.add_component("llm", MockEndpointChatGenerator())
        pipe.run({"llm": {"messages": [ChatMessage.from_user("Hi")]}})

        attrs = blueguardrails_exporter.get_finished_spans()[0].attributes
        assert attrs["server.address"] == "llm-proxy.example.com"
        assert attrs["server.port"] == 9443

    def test_runtime_generation_kwargs_and_tools_override_init_config(self):
        """Runtime generation_kwargs/tools overwrite init-time config on the BlueGuardrails span."""
        _, blueguardrails_exporter, _ = _setup_tracers()

        runtime_tool = Tool(
            name="runtime_lookup",
            description="Runtime lookup",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            function=lambda query: query,
        )

        pipe = Pipeline()
        pipe.add_component("llm", MockInitConfigChatGenerator())
        pipe.run(
            {
                "llm": {
                    "messages": [ChatMessage.from_user("Hi")],
                    "generation_kwargs": {"temperature": 0.75, "max_tokens": 5},
                    "tools": [runtime_tool],
                }
            }
        )

        attrs = blueguardrails_exporter.get_finished_spans()[0].attributes
        assert attrs["gen_ai.request.temperature"] == 0.75
        assert attrs["gen_ai.request.max_tokens"] == 5
        assert json.loads(attrs["gen_ai.tool.definitions"])[0]["name"] == "runtime_lookup"

    def test_blueguardrails_span_does_not_hijack_otel_context(self):
        """BlueGuardrails spans must not become current in the OTel context."""
        user_exporter, blueguardrails_exporter, user_provider = _setup_tracers()

        # Simulate: pipeline span → component span → auto-instrumented child
        lib_tracer = user_provider.get_tracer("openai.instrumentation")

        with tracing.tracer.trace("haystack.pipeline.run"):
            with tracing.tracer.trace(
                "haystack.component.run",
                tags={"haystack.component.type": "MockChatGenerator", "haystack.component.name": "llm"},
            ):
                current = opentelemetry.trace.get_current_span()
                assert not isinstance(current, opentelemetry.trace.NonRecordingSpan)

                with lib_tracer.start_as_current_span("openai.chat") as child:
                    child.set_attribute("test", "auto-instrumented")

        # Auto-instrumented span is a child of user's component span, not BlueGuardrails span
        user_spans = user_exporter.get_finished_spans()
        child_spans = [s for s in user_spans if s.name == "openai.chat"]
        component_spans = [s for s in user_spans if s.name == "haystack.component.run"]
        assert len(child_spans) == 1
        assert len(component_spans) == 1
        assert child_spans[0].parent.span_id == component_spans[0].context.span_id

        # BlueGuardrails spans are roots
        blueguardrails_spans = blueguardrails_exporter.get_finished_spans()
        assert len(blueguardrails_spans) == 1
        assert blueguardrails_spans[0].parent is None

    def test_survives_later_enable_tracing(self):
        """If someone calls enable_tracing() after BlueGuardrails, BlueGuardrails keeps working."""
        _, blueguardrails_exporter, _ = _setup_tracers()

        # Simulate a later tracer installation (e.g., Langfuse)
        new_user_tracer, new_user_exporter, _ = _make_user_otel_tracer()
        tracing.enable_tracing(new_user_tracer)

        pipe = Pipeline()
        pipe.add_component("llm", MockChatGenerator())
        pipe.run({"llm": {"messages": [ChatMessage.from_user("Hi")]}})

        # BlueGuardrails still captured the generator span
        blueguardrails_spans = blueguardrails_exporter.get_finished_spans()
        assert len(blueguardrails_spans) == 1
        assert blueguardrails_spans[0].attributes["gen_ai.operation.name"] == "chat"

        # New user tracer got the Haystack spans
        new_user_spans = new_user_exporter.get_finished_spans()
        assert len(new_user_spans) >= 2

    def test_disable_tracing_disables_blueguardrails_sidecar(self):
        """disable_tracing() must suspend BlueGuardrails as well as the user tracer."""
        _, blueguardrails_exporter, _ = _setup_tracers()

        tracing.disable_tracing()
        assert not tracing.is_tracing_enabled()

        with tracing.tracer.trace(
            "haystack.component.run",
            tags={"haystack.component.type": "MockChatGenerator", "haystack.component.name": "llm"},
        ) as span:
            span.set_content_tag("haystack.component.input", {"messages": [ChatMessage.from_user("Hi")]})

        assert len(blueguardrails_exporter.get_finished_spans()) == 0


# ---------------------------------------------------------------------------
# Datadog-style tracer simulation
# ---------------------------------------------------------------------------


class FakeDatadogSpan(Span):
    """Mimics DatadogSpan: stores tags, ignores content tracing flag."""

    def __init__(self, operation_name: str) -> None:
        self.operation_name = operation_name
        self.tags: dict[str, Any] = {}

    def set_tag(self, key: str, value: Any) -> None:
        self.tags[key] = tracing_utils.coerce_tag_value(value)

    def raw_span(self) -> Any:
        return self

    def get_correlation_data_for_logs(self) -> dict[str, Any]:
        return {"dd.trace_id": "fake-123"}


class FakeDatadogTracer(Tracer):
    """Mimics DatadogTracer: ignores parent_span, manages own span stack."""

    def __init__(self) -> None:
        self.spans: list[FakeDatadogSpan] = []
        self._current: FakeDatadogSpan | None = None

    @contextlib.contextmanager
    def trace(
        self,
        operation_name: str,
        tags: dict[str, Any] | None = None,
        parent_span: Span | None = None,  # noqa: ARG002 — ddtrace ignores this
    ) -> Iterator[Span]:
        span = FakeDatadogSpan(operation_name)
        if tags:
            span.set_tags(tags)
        prev = self._current
        self._current = span
        try:
            yield span
        finally:
            self.spans.append(span)
            self._current = prev

    def current_span(self) -> Span | None:
        return self._current


def _setup_datadog_tracers():
    """Install a Datadog-style tracer and BlueGuardrails tracing.

    Returns:
        Datadog-style tracer and BlueGuardrails exporter.
    """
    dd_tracer = FakeDatadogTracer()
    tracing.enable_tracing(dd_tracer)

    blueguardrails_tracer, blueguardrails_exporter = _make_blueguardrails_tracer()
    configure_blueguardrails_tracer(blueguardrails_tracer)

    return dd_tracer, blueguardrails_exporter


class TestDatadogCoexistence:
    """Verify BlueGuardrails tracer works alongside a Datadog-style tracer."""

    def setup_method(self):
        os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"
        reset_haystack_tracing_state()

    def teardown_method(self):
        reset_haystack_tracing_state()
        os.environ.pop("HAYSTACK_CONTENT_TRACING_ENABLED", None)

    def test_datadog_gets_all_spans_blueguardrails_gets_only_generator(self):
        dd_tracer, blueguardrails_exporter = _setup_datadog_tracers()

        pipe = Pipeline()
        pipe.add_component("prompt_builder", ChatPromptBuilder())
        pipe.add_component("llm", MockChatGenerator())
        pipe.connect("prompt_builder.prompt", "llm.messages")

        messages = [ChatMessage.from_user("Tell me about {{topic}}")]
        pipe.run(
            {
                "prompt_builder": {
                    "template": messages,
                    "template_variables": {"topic": "Berlin"},
                }
            }
        )

        dd_span_ops = [s.operation_name for s in dd_tracer.spans]
        assert any("pipeline" in op for op in dd_span_ops), f"Missing pipeline span in DD: {dd_span_ops}"
        assert len(dd_tracer.spans) >= 3, f"Expected >=3 DD spans, got: {dd_span_ops}"

        blueguardrails_spans = blueguardrails_exporter.get_finished_spans()
        assert len(blueguardrails_spans) == 1
        assert blueguardrails_spans[0].attributes["gen_ai.operation.name"] == "chat"
        assert blueguardrails_spans[0].attributes["gen_ai.response.model"] == "gpt-4o-mock"

    def test_datadog_spans_have_haystack_tags_not_genai(self):
        """DD spans get standard Haystack tags; GenAI semconv stays in BlueGuardrails."""
        dd_tracer, _ = _setup_datadog_tracers()

        pipe = Pipeline()
        pipe.add_component("llm", MockChatGenerator())
        pipe.run({"llm": {"messages": [ChatMessage.from_user("Hi")]}})

        llm_spans = [s for s in dd_tracer.spans if s.tags.get("haystack.component.type") == "MockChatGenerator"]
        assert len(llm_spans) == 1

        llm_span = llm_spans[0]
        assert llm_span.tags["haystack.component.name"] == "llm"
        assert not any(k.startswith("gen_ai.") for k in llm_span.tags)

    def test_datadog_correlation_data_preserved(self):
        """CompositeSpan returns DD's correlation data, not BlueGuardrails's."""
        _setup_datadog_tracers()

        with tracing.tracer.trace(
            "haystack.component.run", tags={"haystack.component.type": "MockChatGenerator"}
        ) as span:
            corr = span.get_correlation_data_for_logs()
            assert corr == {"dd.trace_id": "fake-123"}

    def test_datadog_raw_span_not_wrapped(self):
        """CompositeSpan.raw_span() returns the DD span, not the BlueGuardrails OTel span."""
        _setup_datadog_tracers()

        with tracing.tracer.trace(
            "haystack.component.run", tags={"haystack.component.type": "MockChatGenerator"}
        ) as span:
            raw = span.raw_span()
            assert isinstance(raw, FakeDatadogSpan)
