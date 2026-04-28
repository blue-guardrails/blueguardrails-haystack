# SPDX-FileCopyrightText: 2025-present Blue Guardrails
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import json
from typing import Any

import pytest
from conftest import reset_haystack_tracing_state
from haystack import tracing
from haystack.dataclasses import ChatMessage
from haystack.tracing import Span, Tracer
from haystack.tracing.tracer import NullSpan, NullTracer, ProxyTracer
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import blueguardrails_haystack.tracer as tracer_module
from blueguardrails_haystack.component_config import extract_component_config
from blueguardrails_haystack.proxy import _BlueGuardrailsSidecarProxy, configure_blueguardrails_tracer
from blueguardrails_haystack.span import BlueGuardrailsSpan, CompositeSpan
from blueguardrails_haystack.tracer import BlueGuardrailsTracer

# --- Helpers ---


def _make_provider_and_exporter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource.create({"service.name": "test"}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


class FakeSpan(Span):
    """Minimal Span implementation for testing."""

    def __init__(self):
        self.tags: dict[str, Any] = {}
        self.content_tags: dict[str, Any] = {}

    def set_tag(self, key: str, value: Any) -> None:
        self.tags[key] = value

    def set_content_tag(self, key: str, value: Any) -> None:
        self.content_tags[key] = value

    def raw_span(self) -> Any:
        return "fake_raw"

    def get_correlation_data_for_logs(self) -> dict[str, Any]:
        return {"fake": True}


class FakeTracer(Tracer):
    """Minimal Tracer that yields FakeSpan."""

    def __init__(self):
        self._span = FakeSpan()

    def trace(self, operation_name, tags=None, parent_span=None):
        import contextlib

        @contextlib.contextmanager
        def _ctx():
            if tags:
                self._span.set_tags(tags)
            yield self._span

        return _ctx()

    def current_span(self):
        return self._span


class ExplodingOtelSpan:
    def set_attribute(self, key: str, value: Any) -> None:
        raise RuntimeError("otel boom")

    def update_name(self, name: str) -> None:
        raise RuntimeError("otel boom")


class ExplodingBlueGuardrailsTracer(Tracer):
    def __init__(self, phase: str) -> None:
        self.phase = phase

    @contextlib.contextmanager
    def trace(self, operation_name, tags=None, parent_span=None):
        if self.phase == "enter":
            raise RuntimeError("blueguardrails enter boom")

        try:
            yield FakeSpan()
        finally:
            if self.phase == "exit":
                raise RuntimeError("blueguardrails exit boom")

    def current_span(self):
        return None


class RecordingOTLPSpanExporter(SpanExporter):
    """OTLP exporter test double that records spans instead of exporting over HTTP."""

    instances: list["RecordingOTLPSpanExporter"] = []

    def __init__(self, endpoint: str, headers: dict[str, str] | None = None, **_: object) -> None:
        self.endpoint = endpoint
        self.headers = headers or {}
        self.spans = []
        self.instances.append(self)

    def export(self, spans):
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


class TestComponentConfigExtraction:
    def test_extracts_server_from_explicit_generator_base_url(self):
        class FakeGenerator:
            model = "test-model"
            api_base_url = "https://proxy.example.com:8443/v1"

        config = extract_component_config(FakeGenerator())

        assert config["server"] == {"address": "proxy.example.com", "port": 8443}

    def test_extracts_server_from_nested_client_base_url_with_default_port(self):
        class FakeClient:
            base_url = "https://api.openai.com/v1/"

        class FakeGenerator:
            model = "test-model"
            client = FakeClient()

        config = extract_component_config(FakeGenerator())

        assert config["server"] == {"address": "api.openai.com", "port": 443}

    def test_extracts_server_from_bedrock_client_meta_endpoint_url(self):
        class FakeMeta:
            endpoint_url = "https://bedrock-runtime.us-east-1.amazonaws.com"

        class FakeClient:
            meta = FakeMeta()

        class FakeGenerator:
            model = "test-model"
            client = FakeClient()

        config = extract_component_config(FakeGenerator())

        assert config["server"] == {"address": "bedrock-runtime.us-east-1.amazonaws.com", "port": 443}

    def test_extracts_server_from_google_genai_http_options(self):
        class FakeHttpOptions:
            base_url = "https://generativelanguage.googleapis.com/"

        class FakeApiClient:
            _http_options = FakeHttpOptions()

        class FakeClient:
            _api_client = FakeApiClient()

        class FakeGenerator:
            model = "test-model"
            _client = FakeClient()

        config = extract_component_config(FakeGenerator())

        assert config["server"] == {"address": "generativelanguage.googleapis.com", "port": 443}


# --- BlueGuardrailsTracer tests ---


class TestBlueGuardrailsTracer:
    def test_creates_span_for_chat_generator(self):
        provider, exporter = _make_provider_and_exporter()
        tracer = BlueGuardrailsTracer(provider)

        tags = {
            "haystack.component.name": "llm",
            "haystack.component.type": "OpenAIChatGenerator",
        }
        with tracer.trace("haystack.component.run", tags=tags) as span:
            assert isinstance(span, BlueGuardrailsSpan)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "chat llm"
        assert spans[0].attributes["gen_ai.operation.name"] == "chat"
        assert spans[0].attributes["gen_ai.provider.name"] == "openai"

    def test_creates_span_for_plain_generator(self):
        provider, exporter = _make_provider_and_exporter()
        tracer = BlueGuardrailsTracer(provider)

        tags = {
            "haystack.component.name": "gen",
            "haystack.component.type": "OpenAIGenerator",
        }
        with tracer.trace("haystack.component.run", tags=tags) as span:
            assert isinstance(span, BlueGuardrailsSpan)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes["gen_ai.operation.name"] == "text_completion"

    def test_yields_null_span_for_non_generator(self):
        provider, exporter = _make_provider_and_exporter()
        tracer = BlueGuardrailsTracer(provider)

        tags = {
            "haystack.component.name": "retriever",
            "haystack.component.type": "InMemoryBM25Retriever",
        }
        with tracer.trace("haystack.component.run", tags=tags) as span:
            assert isinstance(span, NullSpan)

        assert len(exporter.get_finished_spans()) == 0

    def test_yields_null_span_for_pipeline_run(self):
        provider, exporter = _make_provider_and_exporter()
        tracer = BlueGuardrailsTracer(provider)

        with tracer.trace("haystack.pipeline.run", tags={}) as span:
            assert isinstance(span, NullSpan)

        assert len(exporter.get_finished_spans()) == 0

    def test_pipeline_run_id_is_propagated_as_agent_run_tag(self):
        provider, exporter = _make_provider_and_exporter()
        tracer = BlueGuardrailsTracer(provider)

        with tracer.trace("haystack.pipeline.run", tags={}):
            tags = {
                "haystack.component.name": "llm",
                "haystack.component.type": "OpenAIChatGenerator",
            }
            with tracer.trace("haystack.component.run", tags=tags):
                pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert "gen_ai.agent.run.tags.pipeline_run_id" in spans[0].attributes
        assert "haystack.pipeline.run_id" not in spans[0].attributes

    def test_conversation_tags_are_propagated(self):
        provider, exporter = _make_provider_and_exporter()
        tracer = BlueGuardrailsTracer(provider, conversation_tags={"env": "test", "customer": "acme"})

        tags = {
            "haystack.component.name": "llm",
            "haystack.component.type": "OpenAIChatGenerator",
        }
        with tracer.trace("haystack.component.run", tags=tags):
            pass

        attrs = exporter.get_finished_spans()[0].attributes
        assert attrs["gen_ai.conversation.tags.env"] == "test"
        assert attrs["gen_ai.conversation.tags.customer"] == "acme"

    def test_haystack_component_name_conversation_tag_is_propagated(self):
        provider, exporter = _make_provider_and_exporter()
        tracer = BlueGuardrailsTracer(provider)

        tags = {
            "haystack.component.name": "llm",
            "haystack.component.type": "OpenAIChatGenerator",
        }
        with tracer.trace("haystack.component.run", tags=tags):
            pass

        attrs = exporter.get_finished_spans()[0].attributes
        assert attrs["gen_ai.conversation.tags.haystack_component_name"] == "llm"


class TestBlueGuardrailsSpanContentMapping:
    def test_set_tag_fail_open(self):
        span = BlueGuardrailsSpan(ExplodingOtelSpan(), is_chat=True)

        span.set_tag("haystack.component.name", "llm")

    def test_chat_input_messages(self):
        provider, exporter = _make_provider_and_exporter()
        tracer = BlueGuardrailsTracer(provider)

        tags = {
            "haystack.component.name": "llm",
            "haystack.component.type": "OpenAIChatGenerator",
        }
        with tracer.trace("haystack.component.run", tags=tags) as span:
            messages = [ChatMessage.from_user("Hello")]
            span.set_content_tag("haystack.component.input", {"messages": messages})

        otel_span = exporter.get_finished_spans()[0]
        input_msgs = json.loads(otel_span.attributes["gen_ai.input.messages"])
        assert input_msgs[0]["role"] == "user"
        assert input_msgs[0]["parts"][0]["content"] == "Hello"

    def test_chat_input_conversion_fail_open(self, monkeypatch):
        provider, exporter = _make_provider_and_exporter()
        tracer = BlueGuardrailsTracer(provider)

        def _explode(messages: list[ChatMessage]) -> str:
            raise RuntimeError("convert boom")

        monkeypatch.setattr("blueguardrails_haystack.span.convert_input_messages", _explode)

        tags = {
            "haystack.component.name": "llm",
            "haystack.component.type": "OpenAIChatGenerator",
        }
        with tracer.trace("haystack.component.run", tags=tags) as span:
            span.set_content_tag("haystack.component.input", {"messages": [ChatMessage.from_user("Hello")]})

        otel_span = exporter.get_finished_spans()[0]
        assert "gen_ai.input.messages" not in otel_span.attributes

    def test_chat_output_messages_with_meta(self):
        provider, exporter = _make_provider_and_exporter()
        tracer = BlueGuardrailsTracer(provider)

        tags = {
            "haystack.component.name": "llm",
            "haystack.component.type": "OpenAIChatGenerator",
        }
        with tracer.trace("haystack.component.run", tags=tags) as span:
            reply = ChatMessage.from_assistant(
                "Hi!",
                meta={
                    "model": "gpt-4o",
                    "finish_reason": "stop",
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                },
            )
            span.set_content_tag("haystack.component.output", {"replies": [reply]})

        otel_span = exporter.get_finished_spans()[0]
        assert otel_span.attributes["gen_ai.response.model"] == "gpt-4o"
        assert otel_span.attributes["gen_ai.usage.input_tokens"] == 10
        assert otel_span.attributes["gen_ai.usage.output_tokens"] == 5
        assert otel_span.attributes["gen_ai.response.finish_reasons"] == ("stop",)

        output_msgs = json.loads(otel_span.attributes["gen_ai.output.messages"])
        assert output_msgs[0]["finish_reason"] == "stop"

    def test_plain_generator_input_output(self):
        provider, exporter = _make_provider_and_exporter()
        tracer = BlueGuardrailsTracer(provider)

        tags = {
            "haystack.component.name": "gen",
            "haystack.component.type": "OpenAIGenerator",
        }
        with tracer.trace("haystack.component.run", tags=tags) as span:
            span.set_content_tag("haystack.component.input", {"prompt": "Tell me a joke"})
            span.set_content_tag(
                "haystack.component.output",
                {
                    "replies": ["Why did the chicken..."],
                    "meta": [{"model": "gpt-4o", "usage": {"prompt_tokens": 5, "completion_tokens": 10}}],
                },
            )

        otel_span = exporter.get_finished_spans()[0]
        input_msgs = json.loads(otel_span.attributes["gen_ai.input.messages"])
        assert input_msgs[0]["parts"][0]["content"] == "Tell me a joke"

        output_msgs = json.loads(otel_span.attributes["gen_ai.output.messages"])
        assert output_msgs[0]["parts"][0]["content"] == "Why did the chicken..."

        assert otel_span.attributes["gen_ai.response.model"] == "gpt-4o"
        assert otel_span.attributes["gen_ai.usage.input_tokens"] == 5
        assert otel_span.attributes["gen_ai.usage.output_tokens"] == 10

    def test_openai_nested_cache_tokens_are_reported_without_double_counting(self):
        provider, exporter = _make_provider_and_exporter()
        tracer = BlueGuardrailsTracer(provider)

        tags = {
            "haystack.component.name": "llm",
            "haystack.component.type": "OpenAIChatGenerator",
        }
        with tracer.trace("haystack.component.run", tags=tags) as span:
            reply = ChatMessage.from_assistant(
                "OK",
                meta={
                    "model": "gpt-4o-mini",
                    "usage": {
                        "prompt_tokens": 5215,
                        "prompt_tokens_details": {"cached_tokens": 5120},
                        "completion_tokens": 1,
                    },
                },
            )
            span.set_content_tag("haystack.component.output", {"replies": [reply]})

        attrs = exporter.get_finished_spans()[0].attributes
        assert attrs["gen_ai.usage.input_tokens"] == 5215
        assert attrs["gen_ai.usage.cache_read.input_tokens"] == 5120
        assert attrs["gen_ai.usage.output_tokens"] == 1
        assert attrs["gen_ai.usage.details.cache_read_tokens"] == 5120

    def test_openai_responses_nested_cache_tokens_are_reported_without_double_counting(self):
        provider, exporter = _make_provider_and_exporter()
        tracer = BlueGuardrailsTracer(provider)

        tags = {
            "haystack.component.name": "llm",
            "haystack.component.type": "OpenAIResponsesChatGenerator",
        }
        with tracer.trace("haystack.component.run", tags=tags) as span:
            reply = ChatMessage.from_assistant(
                "OK",
                meta={
                    "model": "gpt-5-mini",
                    "usage": {
                        "input_tokens": 6013,
                        "input_tokens_details": {"cached_tokens": 5888},
                        "output_tokens": 58,
                    },
                },
            )
            span.set_content_tag("haystack.component.output", {"replies": [reply]})

        attrs = exporter.get_finished_spans()[0].attributes
        assert attrs["gen_ai.usage.input_tokens"] == 6013
        assert attrs["gen_ai.usage.cache_read.input_tokens"] == 5888
        assert attrs["gen_ai.usage.output_tokens"] == 58
        assert attrs["gen_ai.usage.details.cache_read_tokens"] == 5888

    def test_openai_usage_detail_attributes_include_nested_details(self):
        provider, exporter = _make_provider_and_exporter()
        tracer = BlueGuardrailsTracer(provider)

        tags = {
            "haystack.component.name": "llm",
            "haystack.component.type": "OpenAIResponsesChatGenerator",
        }
        with tracer.trace("haystack.component.run", tags=tags) as span:
            reply = ChatMessage.from_assistant(
                "OK",
                meta={
                    "model": "gpt-5-mini",
                    "usage": {
                        "input_tokens": 20,
                        "input_tokens_details": {"cached_tokens": 12, "audio_tokens": 3},
                        "output_tokens": 7,
                        "output_tokens_details": {"reasoning_tokens": 4, "audio_tokens": 2},
                    },
                },
            )
            span.set_content_tag("haystack.component.output", {"replies": [reply]})

        attrs = exporter.get_finished_spans()[0].attributes
        assert attrs["gen_ai.usage.input_tokens"] == 20
        assert attrs["gen_ai.usage.output_tokens"] == 7
        assert attrs["gen_ai.usage.cache_read.input_tokens"] == 12
        assert attrs["gen_ai.usage.details.cache_read_tokens"] == 12
        assert attrs["gen_ai.usage.details.input_audio_tokens"] == 3
        assert attrs["gen_ai.usage.details.output_audio_tokens"] == 2
        assert attrs["gen_ai.usage.details.reasoning_tokens"] == 4

    def test_deepseek_openai_compatible_cache_hit_tokens_are_reported_without_double_counting(self):
        provider, exporter = _make_provider_and_exporter()
        tracer = BlueGuardrailsTracer(provider)

        tags = {
            "haystack.component.name": "llm",
            "haystack.component.type": "OpenAIChatGenerator",
        }
        with tracer.trace("haystack.component.run", tags=tags) as span:
            reply = ChatMessage.from_assistant(
                "OK",
                meta={
                    "model": "deepseek-v4-pro",
                    "usage": {
                        "prompt_tokens": 5007,
                        "prompt_cache_hit_tokens": 4992,
                        "prompt_cache_miss_tokens": 15,
                        "completion_tokens": 8,
                    },
                },
            )
            span.set_content_tag("haystack.component.output", {"replies": [reply]})

        attrs = exporter.get_finished_spans()[0].attributes
        assert attrs["gen_ai.usage.input_tokens"] == 5007
        assert attrs["gen_ai.usage.cache_read.input_tokens"] == 4992
        assert attrs["gen_ai.usage.output_tokens"] == 8
        assert attrs["gen_ai.usage.details.cache_read_tokens"] == 4992

    def test_anthropic_cache_tokens_are_added_to_input_tokens(self):
        provider, exporter = _make_provider_and_exporter()
        tracer = BlueGuardrailsTracer(provider)

        tags = {
            "haystack.component.name": "llm",
            "haystack.component.type": "AnthropicChatGenerator",
        }
        with tracer.trace("haystack.component.run", tags=tags) as span:
            reply = ChatMessage.from_assistant(
                "OK",
                meta={
                    "model": "claude-haiku-4-5",
                    "usage": {
                        "prompt_tokens": 9,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 7000,
                        "completion_tokens": 4,
                    },
                },
            )
            span.set_content_tag("haystack.component.output", {"replies": [reply]})

        attrs = exporter.get_finished_spans()[0].attributes
        assert attrs["gen_ai.usage.input_tokens"] == 7009
        assert attrs["gen_ai.usage.cache_read.input_tokens"] == 7000
        assert attrs["gen_ai.usage.cache_creation.input_tokens"] == 0
        assert attrs["gen_ai.usage.output_tokens"] == 4
        assert attrs["gen_ai.usage.details.cache_read_tokens"] == 7000
        assert "gen_ai.usage.details.cache_write_tokens" not in attrs

    def test_bedrock_cache_tokens_are_added_to_input_tokens(self):
        provider, exporter = _make_provider_and_exporter()
        tracer = BlueGuardrailsTracer(provider)

        tags = {
            "haystack.component.name": "llm",
            "haystack.component.type": "AmazonBedrockChatGenerator",
        }
        with tracer.trace("haystack.component.run", tags=tags) as span:
            reply = ChatMessage.from_assistant(
                "OK",
                meta={
                    "model": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
                    "usage": {
                        "prompt_tokens": 41,
                        "cache_read_input_tokens": 35013,
                        "cache_write_input_tokens": 0,
                        "completion_tokens": 7,
                        "total_tokens": 35061,
                    },
                },
            )
            span.set_content_tag("haystack.component.output", {"replies": [reply]})

        attrs = exporter.get_finished_spans()[0].attributes
        assert attrs["gen_ai.usage.input_tokens"] == 35054
        assert attrs["gen_ai.usage.cache_read.input_tokens"] == 35013
        assert attrs["gen_ai.usage.cache_creation.input_tokens"] == 0
        assert attrs["gen_ai.usage.output_tokens"] == 7
        assert attrs["gen_ai.usage.details.cache_read_tokens"] == 35013
        assert "gen_ai.usage.details.cache_write_tokens" not in attrs

    def test_google_cached_content_is_not_double_counted_and_thoughts_are_output_tokens(self):
        provider, exporter = _make_provider_and_exporter()
        tracer = BlueGuardrailsTracer(provider)

        tags = {
            "haystack.component.name": "llm",
            "haystack.component.type": "GoogleGenAIChatGenerator",
        }
        with tracer.trace("haystack.component.run", tags=tags) as span:
            reply = ChatMessage.from_assistant(
                "OK",
                meta={
                    "model": "gemini-2.5-flash",
                    "usage": {
                        "prompt_token_count": 7210,
                        "cached_content_token_count": 7201,
                        "candidates_token_count": 1,
                        "thoughts_token_count": 58,
                    },
                },
            )
            span.set_content_tag("haystack.component.output", {"replies": [reply]})

        attrs = exporter.get_finished_spans()[0].attributes
        assert attrs["gen_ai.usage.input_tokens"] == 7210
        assert attrs["gen_ai.usage.cache_read.input_tokens"] == 7201
        assert attrs["gen_ai.usage.output_tokens"] == 59
        assert attrs["gen_ai.usage.details.cache_read_tokens"] == 7201
        assert attrs["gen_ai.usage.details.cached_content_tokens"] == 7201
        assert attrs["gen_ai.usage.details.thoughts_tokens"] == 58


# --- CompositeSpan tests ---


class TestCompositeSpan:
    def test_set_tag_forwards_to_both(self):
        orig = FakeSpan()
        blueguardrails = FakeSpan()
        composite = CompositeSpan(orig, blueguardrails)

        composite.set_tag("key", "value")
        assert orig.tags["key"] == "value"
        assert blueguardrails.tags["key"] == "value"

    def test_set_content_tag_forwards_to_both(self):
        orig = FakeSpan()
        blueguardrails = FakeSpan()
        composite = CompositeSpan(orig, blueguardrails)

        composite.set_content_tag("haystack.component.input", {"messages": []})
        assert orig.content_tags["haystack.component.input"] == {"messages": []}
        assert blueguardrails.content_tags["haystack.component.input"] == {"messages": []}

    def test_raw_span_returns_original(self):
        orig = FakeSpan()
        blueguardrails = FakeSpan()
        composite = CompositeSpan(orig, blueguardrails)
        assert composite.raw_span() == "fake_raw"

    def test_correlation_data_from_original(self):
        orig = FakeSpan()
        blueguardrails = FakeSpan()
        composite = CompositeSpan(orig, blueguardrails)
        assert composite.get_correlation_data_for_logs() == {"fake": True}


# --- Sidecar proxy tests ---


class TestBlueGuardrailsSidecarProxy:
    def _make_proxy(self):
        """Create a proxy with Blue Guardrails and a user tracer.

        Returns:
            Proxy and in-memory span exporter.
        """
        provider, exporter = _make_provider_and_exporter()
        proxy = ProxyTracer(provided_tracer=FakeTracer())
        proxy.__class__ = _BlueGuardrailsSidecarProxy
        proxy._blueguardrails_tracer = BlueGuardrailsTracer(provider)
        proxy._blueguardrails_enabled = True
        return proxy, exporter

    def _make_blueguardrails_only_proxy(self):
        """Create a proxy with Blue Guardrails and no user tracer.

        Returns:
            Proxy and in-memory span exporter.
        """
        provider, exporter = _make_provider_and_exporter()
        proxy = ProxyTracer(provided_tracer=NullTracer())
        proxy.__class__ = _BlueGuardrailsSidecarProxy
        proxy._blueguardrails_tracer = BlueGuardrailsTracer(provider)
        proxy._blueguardrails_enabled = True
        return proxy, exporter

    # --- Sidecar mode (user tracer + Blue Guardrails) ---

    def test_yields_composite_span_for_generator(self):
        proxy, _ = self._make_proxy()
        with proxy.trace("haystack.component.run", tags={"haystack.component.type": "OpenAIChatGenerator"}) as span:
            assert isinstance(span, CompositeSpan)

    def test_yields_composite_span_for_non_generator(self):
        """Even non-generator spans are composite (Blue Guardrails side is NullSpan)."""
        proxy, _ = self._make_proxy()
        with proxy.trace("haystack.component.run", tags={"haystack.component.type": "PromptBuilder"}) as span:
            assert isinstance(span, CompositeSpan)

    def test_current_span_delegates_to_actual_tracer(self):
        proxy, _ = self._make_proxy()
        assert proxy.current_span() is proxy.actual_tracer._span

    def test_reads_actual_tracer_at_call_time(self):
        """Changing actual_tracer after install is picked up on next trace."""
        proxy, _ = self._make_proxy()

        new_tracer = FakeTracer()
        proxy.actual_tracer = new_tracer

        with proxy.trace("haystack.component.run", tags={"haystack.component.type": "OpenAIChatGenerator"}) as span:
            # The user span should come from new_tracer
            assert span._original is new_tracer._span

    def test_parent_span_unwrapping(self):
        proxy, _ = self._make_proxy()

        with proxy.trace("haystack.pipeline.run") as parent_span:
            assert isinstance(parent_span, CompositeSpan)
            with proxy.trace(
                "haystack.component.run",
                tags={"haystack.component.type": "OpenAIChatGenerator", "haystack.component.name": "llm"},
                parent_span=parent_span,
            ) as child_span:
                assert isinstance(child_span, CompositeSpan)

    def test_non_generator_does_not_create_blueguardrails_span(self):
        proxy, exporter = self._make_proxy()
        with proxy.trace(
            "haystack.component.run",
            tags={"haystack.component.type": "PromptBuilder", "haystack.component.name": "pb"},
        ):
            pass

        assert len(exporter.get_finished_spans()) == 0

    def test_blueguardrails_enter_failure_falls_back_to_user_span(self):
        proxy, _ = self._make_proxy()
        proxy._blueguardrails_tracer = ExplodingBlueGuardrailsTracer("enter")

        with proxy.trace("haystack.component.run", tags={"haystack.component.type": "OpenAIChatGenerator"}) as span:
            assert span is proxy.actual_tracer._span

    def test_blueguardrails_exit_failure_is_swallowed(self):
        proxy, _ = self._make_proxy()
        proxy._blueguardrails_tracer = ExplodingBlueGuardrailsTracer("exit")

        with proxy.trace("haystack.component.run", tags={"haystack.component.type": "OpenAIChatGenerator"}) as span:
            assert isinstance(span, CompositeSpan)

    def test_blueguardrails_exit_failure_does_not_hide_user_exception(self):
        proxy, _ = self._make_proxy()
        proxy._blueguardrails_tracer = ExplodingBlueGuardrailsTracer("exit")

        with pytest.raises(RuntimeError, match="user boom"):
            with proxy.trace("haystack.component.run", tags={"haystack.component.type": "OpenAIChatGenerator"}):
                raise RuntimeError("user boom")

    # --- Blue Guardrails-only mode (no user tracer) ---

    def test_blueguardrails_only_yields_blueguardrails_span_for_generator(self):
        proxy, exporter = self._make_blueguardrails_only_proxy()
        with proxy.trace(
            "haystack.component.run",
            tags={"haystack.component.type": "OpenAIChatGenerator", "haystack.component.name": "llm"},
        ) as span:
            assert isinstance(span, BlueGuardrailsSpan)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes["gen_ai.operation.name"] == "chat"

    def test_blueguardrails_only_yields_null_span_for_non_generator(self):
        proxy, exporter = self._make_blueguardrails_only_proxy()
        with proxy.trace(
            "haystack.component.run",
            tags={"haystack.component.type": "PromptBuilder", "haystack.component.name": "pb"},
        ) as span:
            assert isinstance(span, NullSpan)

        assert len(exporter.get_finished_spans()) == 0

    def test_blueguardrails_only_transitions_to_sidecar_on_enable_tracing(self):
        """When a user tracer is installed after Blue Guardrails, switch to composite mode."""
        proxy, exporter = self._make_blueguardrails_only_proxy()

        # Install a user tracer
        user_tracer = FakeTracer()
        proxy.actual_tracer = user_tracer

        with proxy.trace(
            "haystack.component.run",
            tags={"haystack.component.type": "OpenAIChatGenerator", "haystack.component.name": "llm"},
        ) as span:
            assert isinstance(span, CompositeSpan)
            assert span._original is user_tracer._span

    def test_blueguardrails_only_enter_failure_yields_null_span(self):
        proxy, _ = self._make_blueguardrails_only_proxy()
        proxy._blueguardrails_tracer = ExplodingBlueGuardrailsTracer("enter")

        with proxy.trace("haystack.component.run", tags={"haystack.component.type": "OpenAIChatGenerator"}) as span:
            assert isinstance(span, NullSpan)

    def test_blueguardrails_only_exit_failure_does_not_hide_user_exception(self):
        proxy, _ = self._make_blueguardrails_only_proxy()
        proxy._blueguardrails_tracer = ExplodingBlueGuardrailsTracer("exit")

        with pytest.raises(RuntimeError, match="user boom"):
            with proxy.trace("haystack.component.run", tags={"haystack.component.type": "OpenAIChatGenerator"}):
                raise RuntimeError("user boom")


class TestConfigureBlueGuardrailsTracer:
    def setup_method(self):
        reset_haystack_tracing_state()

    def teardown_method(self):
        reset_haystack_tracing_state()

    def test_blueguardrails_only_creates_spans_without_user_tracer(self):
        """Blue Guardrails works in standalone mode when no user tracer is installed."""
        provider, exporter = _make_provider_and_exporter()
        configure_blueguardrails_tracer(BlueGuardrailsTracer(provider))

        with tracing.tracer.trace(
            "haystack.component.run",
            tags={"haystack.component.type": "OpenAIChatGenerator", "haystack.component.name": "llm"},
        ) as span:
            span.set_content_tag("haystack.component.input", {"messages": [ChatMessage.from_user("Hi")]})

        assert len(exporter.get_finished_spans()) == 1

    def test_default_configure_reads_blueguardrails_api_key(self, monkeypatch):
        RecordingOTLPSpanExporter.instances = []
        monkeypatch.setenv("BLUEGUARDRAILS_API_KEY", "fake-key")
        monkeypatch.setattr(tracer_module, "OTLPSpanExporter", RecordingOTLPSpanExporter)
        monkeypatch.setattr(tracer_module, "BatchSpanProcessor", SimpleSpanProcessor)

        blueguardrails_tracer = configure_blueguardrails_tracer(name="support-agent", tags={"env": "test"})

        assert isinstance(blueguardrails_tracer, BlueGuardrailsTracer)
        exporter = RecordingOTLPSpanExporter.instances[-1]
        assert exporter.endpoint == "https://api.blueguardrails.com/v1/traces"
        assert exporter.headers == {"Authorization": "Bearer fake-key"}

        with tracing.tracer.trace(
            "haystack.component.run",
            tags={"haystack.component.type": "OpenAIChatGenerator", "haystack.component.name": "llm"},
        ):
            pass

        assert len(exporter.spans) == 1
        span = exporter.spans[0]
        assert span.resource.attributes["haystack.pipeline.name"] == "support-agent"
        assert span.attributes["gen_ai.conversation.tags.env"] == "test"

    def test_default_configure_accepts_explicit_api_key(self, monkeypatch):
        RecordingOTLPSpanExporter.instances = []
        monkeypatch.delenv("BLUEGUARDRAILS_API_KEY", raising=False)
        monkeypatch.setattr(tracer_module, "OTLPSpanExporter", RecordingOTLPSpanExporter)
        monkeypatch.setattr(tracer_module, "BatchSpanProcessor", SimpleSpanProcessor)

        configure_blueguardrails_tracer(name="support-agent", api_key="explicit-key")

        exporter = RecordingOTLPSpanExporter.instances[-1]
        assert exporter.headers == {"Authorization": "Bearer explicit-key"}

    def test_default_configure_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("BLUEGUARDRAILS_API_KEY", raising=False)

        with pytest.raises(ValueError, match="BLUEGUARDRAILS_API_KEY"):
            configure_blueguardrails_tracer()

    def test_default_configure_validates_sample_rate(self, monkeypatch):
        monkeypatch.setenv("BLUEGUARDRAILS_API_KEY", "fake-key")

        with pytest.raises(ValueError, match="sample_rate"):
            configure_blueguardrails_tracer(sample_rate=1.1)

    def test_disable_tracing_suspends_blueguardrails(self):
        provider, exporter = _make_provider_and_exporter()
        configure_blueguardrails_tracer(BlueGuardrailsTracer(provider))

        tracing.disable_tracing()

        with tracing.tracer.trace(
            "haystack.component.run",
            tags={"haystack.component.type": "OpenAIChatGenerator", "haystack.component.name": "llm"},
        ) as span:
            span.set_content_tag("haystack.component.input", {"messages": [ChatMessage.from_user("Hi")]})

        assert len(exporter.get_finished_spans()) == 0

    def test_enabling_null_tracer_suspends_blueguardrails(self):
        provider, exporter = _make_provider_and_exporter()
        configure_blueguardrails_tracer(BlueGuardrailsTracer(provider))

        tracing.enable_tracing(NullTracer())

        with tracing.tracer.trace(
            "haystack.component.run",
            tags={"haystack.component.type": "OpenAIChatGenerator", "haystack.component.name": "llm"},
        ) as span:
            span.set_content_tag("haystack.component.input", {"messages": [ChatMessage.from_user("Hi")]})

        assert len(exporter.get_finished_spans()) == 0
