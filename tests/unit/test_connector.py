# SPDX-FileCopyrightText: 2025-present Blue Guardrails
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from copy import deepcopy
from unittest.mock import patch

import pytest
from conftest import reset_haystack_tracing_state
from haystack import Pipeline, component, default_from_dict, default_to_dict, tracing
from haystack.dataclasses import ChatMessage
from haystack.tracing.tracer import NullTracer
from haystack.utils import Secret
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

import blueguardrails_haystack.components.connector as connector_module
from blueguardrails_haystack import BlueGuardrailsConnector
from blueguardrails_haystack.component_config import extract_component_config
from blueguardrails_haystack.proxy import _BlueGuardrailsSidecarProxy


@component
class SerdeChatGenerator:
    """Serializable test ChatGenerator used to exercise pipeline serde."""

    def __init__(
        self,
        model: str = "serde-chat-model",
        api_base_url: str = "https://serde.example.com:8443/v1",
        generation_kwargs: dict | None = None,
    ) -> None:
        self.model = model
        self.api_base_url = api_base_url
        self.generation_kwargs = generation_kwargs or {
            "temperature": 0.2,
            "max_tokens": 11,
            "stop": ["END"],
        }

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage]) -> dict[str, list[ChatMessage]]:
        return {
            "replies": [
                ChatMessage.from_assistant(
                    "serde response",
                    meta={
                        "finish_reason": "stop",
                        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                    },
                )
            ]
        }

    def to_dict(self) -> dict:
        return default_to_dict(
            self,
            model=self.model,
            api_base_url=self.api_base_url,
            generation_kwargs=self.generation_kwargs,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "SerdeChatGenerator":
        return default_from_dict(cls, data)


class RecordingOTLPSpanExporter(SpanExporter):
    """OTLPSpanExporter test double that records spans instead of exporting over HTTP."""

    instances: list["RecordingOTLPSpanExporter"] = []

    def __init__(self, endpoint: str, headers: dict[str, str] | None = None, **_: object) -> None:
        self.endpoint = endpoint
        self.headers = headers or {}
        self.spans = []
        self.shutdown_called = False
        self.instances.append(self)

    def export(self, spans):
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        self.shutdown_called = True

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


@pytest.fixture
def recording_connector_exporter(monkeypatch):
    RecordingOTLPSpanExporter.instances = []
    monkeypatch.setattr(connector_module, "OTLPSpanExporter", RecordingOTLPSpanExporter)
    monkeypatch.setattr(connector_module, "BatchSpanProcessor", SimpleSpanProcessor)
    return RecordingOTLPSpanExporter


def _make_serde_pipeline(model: str = "serde-chat-model") -> Pipeline:
    pipe = Pipeline()
    pipe.add_component(
        "blueguardrails",
        BlueGuardrailsConnector(
            name="serde-pipeline",
            endpoint="http://localhost:4318/v1/traces",
            api_key=Secret.from_env_var("BLUE_GUARDRAILS_API_KEY", strict=False),
            tags={"suite": "serde"},
        ),
    )
    pipe.add_component("llm", SerdeChatGenerator(model=model))
    return pipe


def _latest_exported_spans(exporter_cls: type[RecordingOTLPSpanExporter]):
    assert exporter_cls.instances, "expected connector deserialization to initialize an exporter"
    return exporter_cls.instances[-1].spans


def _assert_loaded_component_config(pipe: Pipeline, *, model: str = "serde-chat-model") -> None:
    llm = pipe.get_component("llm")

    assert extract_component_config(llm) == {
        "model": model,
        "server": {"address": "serde.example.com", "port": 8443},
        "request_options": {"temperature": 0.2, "max_tokens": 11, "stop": ["END"]},
    }


def _assert_loaded_pipeline_traces_chat_generator(
    pipe: Pipeline,
    exporter_cls: type[RecordingOTLPSpanExporter],
    *,
    model: str = "serde-chat-model",
) -> None:
    result = pipe.run({"llm": {"messages": [ChatMessage.from_user("Hi from serde")]}})

    assert result["llm"]["replies"][0].text == "serde response"

    spans = _latest_exported_spans(exporter_cls)
    assert len(spans) == 1

    span = spans[0]
    attrs = span.attributes
    assert span.name == f"chat {model}"
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.provider.name"] == "serde"
    assert attrs["gen_ai.request.model"] == model
    assert attrs["gen_ai.response.model"] == model
    assert attrs["server.address"] == "serde.example.com"
    assert attrs["server.port"] == 8443
    assert attrs["gen_ai.request.temperature"] == 0.2
    assert attrs["gen_ai.request.max_tokens"] == 11
    assert attrs["gen_ai.request.stop_sequences"] == ("END",)
    assert attrs["gen_ai.conversation.tags.suite"] == "serde"
    assert attrs["gen_ai.conversation.tags.haystack_component_name"] == "llm"

    input_messages = json.loads(attrs["gen_ai.input.messages"])
    assert input_messages[0]["role"] == "user"
    assert input_messages[0]["parts"][0]["content"] == "Hi from serde"

    output_messages = json.loads(attrs["gen_ai.output.messages"])
    assert output_messages[0]["role"] == "assistant"
    assert output_messages[0]["parts"][0]["content"] == "serde response"
    assert attrs["gen_ai.usage.input_tokens"] == 3
    assert attrs["gen_ai.usage.output_tokens"] == 2


class TestBlueGuardrailsConnector:
    def setup_method(self):
        reset_haystack_tracing_state()

    def teardown_method(self):
        reset_haystack_tracing_state()

    def test_installs_sidecar_on_proxy(self):
        BlueGuardrailsConnector(
            name="test",
            api_key=Secret.from_token("test-key"),
            endpoint="http://localhost:4318/v1/traces",
        )
        assert tracing.tracer.__class__ is _BlueGuardrailsSidecarProxy
        assert tracing.tracer._blueguardrails_tracer is not None
        assert tracing.tracer._blueguardrails_enabled

    def test_does_not_touch_actual_tracer(self):
        """Blue Guardrails installs as a sidecar — actual_tracer stays user-owned."""
        original = NullTracer()
        tracing.enable_tracing(original)

        BlueGuardrailsConnector(
            name="test",
            api_key=Secret.from_token("test-key"),
            endpoint="http://localhost:4318/v1/traces",
        )

        # actual_tracer is still the user's tracer, not wrapped
        assert tracing.tracer.actual_tracer is original

    def test_survives_later_enable_tracing(self):
        """If someone calls enable_tracing() after Blue Guardrails, Blue Guardrails keeps working."""
        BlueGuardrailsConnector(
            name="test",
            api_key=Secret.from_token("test-key"),
            endpoint="http://localhost:4318/v1/traces",
        )
        blueguardrails_tracer = tracing.tracer._blueguardrails_tracer

        # Simulate a later tracer installation (e.g., Langfuse)
        new_tracer = NullTracer()
        tracing.enable_tracing(new_tracer)

        # Blue Guardrails sidecar is still installed, actual_tracer changed
        assert tracing.tracer.__class__ is _BlueGuardrailsSidecarProxy
        assert tracing.tracer._blueguardrails_tracer is blueguardrails_tracer
        assert tracing.tracer.actual_tracer is new_tracer

    def test_idempotent_double_init(self):
        BlueGuardrailsConnector(
            name="test",
            api_key=Secret.from_token("test-key"),
            endpoint="http://localhost:4318/v1/traces",
        )
        first_blueguardrails = tracing.tracer._blueguardrails_tracer

        BlueGuardrailsConnector(
            name="test",
            api_key=Secret.from_token("test-key"),
            endpoint="http://localhost:4318/v1/traces",
        )

        # Class swap is idempotent and identical config reuses the existing tracer
        assert tracing.tracer.__class__ is _BlueGuardrailsSidecarProxy
        assert tracing.tracer._blueguardrails_tracer is first_blueguardrails

    def test_run_returns_name(self):
        connector = BlueGuardrailsConnector(
            name="my-pipeline",
            api_key=Secret.from_token("test-key"),
            endpoint="http://localhost:4318/v1/traces",
        )
        result = connector.run()
        assert result == {"name": "my-pipeline"}

    def test_serialization_roundtrip(self):
        with patch.dict(os.environ, {"BLUE_GUARDRAILS_API_KEY": "fake-key"}):
            connector = BlueGuardrailsConnector(
                name="test",
                endpoint="http://localhost:4318/v1/traces",
                api_key=Secret.from_env_var("BLUE_GUARDRAILS_API_KEY"),
                sample_rate=0.5,
                tags={"env": "test"},
            )
        data = connector.to_dict()

        assert data["init_parameters"]["name"] == "test"
        assert data["init_parameters"]["sample_rate"] == 0.5
        assert data["init_parameters"]["tags"] == {"env": "test"}
        assert data["init_parameters"]["api_key"] == {
            "type": "env_var",
            "env_vars": ["BLUE_GUARDRAILS_API_KEY"],
            "strict": True,
        }

    def test_deserialization(self):
        data = {
            "type": "blueguardrails_haystack.components.connector.BlueGuardrailsConnector",
            "init_parameters": {
                "name": "test",
                "endpoint": "http://localhost:4318/v1/traces",
                "api_key": {"type": "env_var", "env_vars": ["BLUE_GUARDRAILS_API_KEY"], "strict": False},
                "sample_rate": 0.1,
                "tags": None,
            },
        }
        with patch.dict(os.environ, {"BLUE_GUARDRAILS_API_KEY": "fake-key"}):
            connector = BlueGuardrailsConnector.from_dict(data)

        assert connector.name == "test"
        assert connector.sample_rate == 0.1

    def test_connector_to_dict_from_dict_roundtrip_initializes_tracing(self, recording_connector_exporter, monkeypatch):
        monkeypatch.setenv("BLUE_GUARDRAILS_API_KEY", "fake-key")
        connector = BlueGuardrailsConnector(
            name="serde-connector",
            endpoint="http://localhost:4318/v1/traces",
            api_key=Secret.from_env_var("BLUE_GUARDRAILS_API_KEY"),
            sample_rate=1.0,
            tags={"suite": "serde"},
        )
        data = connector.to_dict()

        assert data == {
            "type": "blueguardrails_haystack.components.connector.BlueGuardrailsConnector",
            "init_parameters": {
                "name": "serde-connector",
                "endpoint": "http://localhost:4318/v1/traces",
                "api_key": {"type": "env_var", "env_vars": ["BLUE_GUARDRAILS_API_KEY"], "strict": True},
                "sample_rate": 1.0,
                "tags": {"suite": "serde"},
            },
        }

        reset_haystack_tracing_state()
        round_tripped = BlueGuardrailsConnector.from_dict(deepcopy(data))

        assert round_tripped.name == "serde-connector"
        assert round_tripped.endpoint == "http://localhost:4318/v1/traces"
        assert round_tripped.sample_rate == 1.0
        assert round_tripped.tags == {"suite": "serde"}
        assert round_tripped.api_key.to_dict() == {
            "type": "env_var",
            "env_vars": ["BLUE_GUARDRAILS_API_KEY"],
            "strict": True,
        }
        assert tracing.tracer.__class__ is _BlueGuardrailsSidecarProxy
        assert tracing.tracer._blueguardrails_tracer is not None
        assert tracing.tracer._blueguardrails_enabled

        exporter = recording_connector_exporter.instances[-1]
        assert exporter.endpoint == "http://localhost:4318/v1/traces"
        assert exporter.headers == {"Authorization": "Bearer fake-key"}

        with tracing.tracer.trace(
            "haystack.component.run",
            tags={"haystack.component.name": "llm", "haystack.component.type": "SerdeChatGenerator"},
        ) as span:
            span.set_content_tag("haystack.component.input", {"messages": [ChatMessage.from_user("Hello")]})

        spans = _latest_exported_spans(recording_connector_exporter)
        assert len(spans) == 1
        assert spans[0].attributes["gen_ai.operation.name"] == "chat"
        assert spans[0].attributes["gen_ai.conversation.tags.suite"] == "serde"

    def test_pipeline_with_connector_and_chat_generator_to_dict_from_dict(
        self, recording_connector_exporter, monkeypatch
    ):
        monkeypatch.setenv("HAYSTACK_CONTENT_TRACING_ENABLED", "true")
        pipe = _make_serde_pipeline(model="from-dict-model")
        data = pipe.to_dict()

        assert data["components"]["blueguardrails"]["type"] == (
            "blueguardrails_haystack.components.connector.BlueGuardrailsConnector"
        )
        assert data["components"]["llm"]["type"].endswith("SerdeChatGenerator")

        reset_haystack_tracing_state()
        loaded = Pipeline.from_dict(deepcopy(data))

        assert isinstance(loaded.get_component("blueguardrails"), BlueGuardrailsConnector)
        assert isinstance(loaded.get_component("llm"), SerdeChatGenerator)
        _assert_loaded_component_config(loaded, model="from-dict-model")
        assert tracing.tracer.__class__ is _BlueGuardrailsSidecarProxy
        assert tracing.tracer._blueguardrails_tracer is not None
        assert tracing.tracer._blueguardrails_enabled

        _assert_loaded_pipeline_traces_chat_generator(
            loaded,
            recording_connector_exporter,
            model="from-dict-model",
        )

    def test_pipeline_with_connector_yaml_roundtrip_initializes_tracer_and_component_patch(
        self, recording_connector_exporter, monkeypatch
    ):
        monkeypatch.setenv("HAYSTACK_CONTENT_TRACING_ENABLED", "true")
        pipe = _make_serde_pipeline(model="yaml-model")
        yaml_data = pipe.dumps()

        assert "BlueGuardrailsConnector" in yaml_data
        assert "SerdeChatGenerator" in yaml_data

        reset_haystack_tracing_state()
        loaded = Pipeline.loads(yaml_data)

        assert isinstance(loaded.get_component("blueguardrails"), BlueGuardrailsConnector)
        assert isinstance(loaded.get_component("llm"), SerdeChatGenerator)
        _assert_loaded_component_config(loaded, model="yaml-model")
        assert tracing.tracer.__class__ is _BlueGuardrailsSidecarProxy
        assert tracing.tracer._blueguardrails_tracer is not None
        assert tracing.tracer._blueguardrails_enabled

        _assert_loaded_pipeline_traces_chat_generator(
            loaded,
            recording_connector_exporter,
            model="yaml-model",
        )
