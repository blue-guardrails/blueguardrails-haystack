# SPDX-FileCopyrightText: 2025-present Blue Guardrails
#
# SPDX-License-Identifier: Apache-2.0

"""Real LLM integration tests for Blue Guardrails GenAI tracing.

These tests make live provider calls. Select them with pytest's integration marker:

    uv run --extra integration pytest -m integration tests/integration/test_real_generators.py

Set provider credentials in the usual Haystack environment variables or in a .env
file loaded by python-dotenv. For Bedrock, `AWS_BEARER_TOKEN_BEDROCK` is also
accepted for API key auth when a region is set. To also export the captured spans
to Blue Guardrails, set BG_API_KEY and add --bg-send-traces:

    uv run --extra integration pytest -m integration --bg-send-traces tests/integration/test_real_generators.py

You can also set BG_SEND_TRACES=1 in .env instead of passing the flag.

To persist raw Haystack inputs/outputs and captured semconv attributes as JSON fixtures, add:

    BG_RECORD_LLM_FIXTURES=1 BG_LLM_FIXTURE_DIR=tests/fixtures/llm_io ...
"""

from __future__ import annotations

import base64
import importlib
import json
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pytest
from conftest import reset_haystack_tracing_state
from haystack import Pipeline
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool
from haystack.utils import Secret
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import blueguardrails_haystack.components.connector as connector_module
from blueguardrails_haystack import BlueGuardrailsConnector
from blueguardrails_haystack.tracer import BGTracer, install_bg_tracer

pytestmark = pytest.mark.integration

PROMPT = "Reply with the exact phrase: blueguardrails trace ok"
SYSTEM_PROMPT = "You are a concise integration-test assistant. Follow the user's instruction exactly."
AGENT_TOOL_VALUE = "blueguardrails agent trace ok"
AGENT_PROMPT = (
    "Use the blueguardrails_noop tool exactly once with value "
    f"'{AGENT_TOOL_VALUE}'. Do not answer before calling the tool. "
    "After the tool result is available, send a final assistant response that includes the exact tool result."
)
TWO_GENERATOR_PIPELINE_PROMPTS = {
    "first_llm": "Reply with the exact phrase: blueguardrails first pipeline trace ok",
    "second_llm": "Reply with the exact phrase: blueguardrails second pipeline trace ok",
}
MAX_TOKENS = int(os.getenv("BG_LLM_MAX_TOKENS", "32"))
AGENT_MAX_TOKENS = int(os.getenv("BG_LLM_AGENT_MAX_TOKENS", str(max(MAX_TOKENS, 64))))

OPENAI_MODEL = "gpt-5.4-nano"
ANTHROPIC_MODEL = "claude-haiku-4-5"
GOOGLE_MODEL = "gemini-3.1-flash-lite-preview"
BEDROCK_MODEL = "global.anthropic.claude-haiku-4-5-20251001-v1:0"


@dataclass(frozen=True)
class RealGeneratorCase:
    id: str
    class_path: str
    kind: Literal["chat", "text"]
    provider: str
    model_env: str
    default_model: str
    init_kwargs: Callable[[str], dict[str, Any]]
    generation_kwargs: dict[str, Any]
    credentials_available: Callable[[], tuple[bool, str]]
    init_accepts_generation_kwargs: bool = True
    expect_usage: bool = True
    expect_finish_reasons: bool = True

    @property
    def operation_name(self) -> str:
        return "chat" if self.kind == "chat" else "text_completion"


@dataclass(frozen=True)
class RealGeneratorRunVariant:
    id: str
    streaming: bool


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").lower() in {"1", "true", "yes", "on"}


def _require_env(*names: str) -> tuple[bool, str]:
    missing = [name for name in names if not os.getenv(name)]
    if missing:
        return False, f"missing environment variable(s): {', '.join(missing)}"
    return True, ""


def _openai_credentials() -> tuple[bool, str]:
    return _require_env("OPENAI_API_KEY")


def _azure_credentials() -> tuple[bool, str]:
    has_token = bool(os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_AD_TOKEN"))
    if not has_token:
        return False, "missing AZURE_OPENAI_API_KEY or AZURE_OPENAI_AD_TOKEN"
    return _require_env("AZURE_OPENAI_ENDPOINT")


def _azure_api_key_credentials() -> tuple[bool, str]:
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        return False, "missing AZURE_OPENAI_API_KEY"
    return _require_env("AZURE_OPENAI_ENDPOINT")


def _anthropic_credentials() -> tuple[bool, str]:
    return _require_env("ANTHROPIC_API_KEY")


def _google_credentials() -> tuple[bool, str]:
    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        return True, ""
    return False, "missing GOOGLE_API_KEY or GEMINI_API_KEY"


def _bedrock_credentials() -> tuple[bool, str]:
    if not (os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION")):
        return False, "missing AWS_DEFAULT_REGION or AWS_REGION"

    if _truthy_env("BG_AWS_CREDENTIALS_CONFIGURED"):
        return True, ""

    has_bearer_token = bool(os.getenv("AWS_BEARER_TOKEN_BEDROCK"))
    has_static_keys = bool(os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"))
    has_profile = bool(os.getenv("AWS_PROFILE"))
    has_web_identity = bool(os.getenv("AWS_WEB_IDENTITY_TOKEN_FILE") and os.getenv("AWS_ROLE_ARN"))
    has_container_creds = bool(
        os.getenv("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI") or os.getenv("AWS_CONTAINER_CREDENTIALS_FULL_URI")
    )
    if has_bearer_token or has_static_keys or has_profile or has_web_identity or has_container_creds:
        return True, ""
    return False, (
        "missing AWS credentials; set AWS_BEARER_TOKEN_BEDROCK, AWS credentials, or "
        "BG_AWS_CREDENTIALS_CONFIGURED=1 to rely on ambient AWS metadata"
    )


def _openai_init(model: str) -> dict[str, Any]:
    return {"model": model}


def _azure_init(model: str) -> dict[str, Any]:
    return {
        "azure_deployment": model,
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    }


def _azure_responses_init(model: str) -> dict[str, Any]:
    return {"azure_deployment": model}


def _model_init(model: str) -> dict[str, Any]:
    return {"model": model}


def _bedrock_auth_init_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "aws_region_name": Secret.from_env_var(["AWS_DEFAULT_REGION", "AWS_REGION"], strict=False),
    }

    if os.getenv("AWS_BEARER_TOKEN_BEDROCK"):
        # Botocore still resolves AWS config/credentials while creating the client, even though requests will use the
        # Bedrock bearer token. Supplying harmless explicit credentials and overriding AWS_PROFILE avoids failures from
        # unrelated local AWS profiles/configuration; botocore ignores these values once it selects httpBearerAuth.
        kwargs.update(
            {
                "aws_access_key_id": Secret.from_token("unused-for-bedrock-bearer-auth"),
                "aws_secret_access_key": Secret.from_token("unused-for-bedrock-bearer-auth"),
                "aws_session_token": None,
                "aws_profile_name": Secret.from_token("default"),
            }
        )

    return kwargs


def _bedrock_init(model: str) -> dict[str, Any]:
    return {"model": model, **_bedrock_auth_init_kwargs()}


def _bedrock_text_init(model: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        **_bedrock_auth_init_kwargs(),
        "max_length": MAX_TOKENS,
    }
    model_family = os.getenv("BG_BEDROCK_TEXT_MODEL_FAMILY", "anthropic.claude")
    if model_family:
        kwargs["model_family"] = model_family
    return kwargs


def _noop_tool(value: str) -> str:
    return value


def _integration_test_tool() -> Tool:
    return Tool(
        name="blueguardrails_noop",
        description="Returns the provided value. Do not use unless explicitly asked to call a tool.",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        function=_noop_tool,
    )


CHAT_CASES = [
    RealGeneratorCase(
        id="openai_chat",
        class_path="haystack.components.generators.chat.openai:OpenAIChatGenerator",
        kind="chat",
        provider="openai",
        model_env="BG_OPENAI_MODEL",
        default_model=OPENAI_MODEL,
        init_kwargs=_openai_init,
        generation_kwargs={"max_completion_tokens": MAX_TOKENS},
        credentials_available=_openai_credentials,
    ),
    RealGeneratorCase(
        id="azure_openai_chat",
        class_path="haystack.components.generators.chat.azure:AzureOpenAIChatGenerator",
        kind="chat",
        provider="azure.ai.openai",
        model_env="BG_AZURE_OPENAI_DEPLOYMENT",
        default_model=OPENAI_MODEL,
        init_kwargs=_azure_init,
        generation_kwargs={"max_completion_tokens": MAX_TOKENS},
        credentials_available=_azure_credentials,
    ),
    RealGeneratorCase(
        id="openai_responses_chat",
        class_path="haystack.components.generators.chat.openai_responses:OpenAIResponsesChatGenerator",
        kind="chat",
        provider="openai",
        model_env="BG_OPENAI_RESPONSES_MODEL",
        default_model=OPENAI_MODEL,
        init_kwargs=_openai_init,
        generation_kwargs={"max_output_tokens": MAX_TOKENS},
        credentials_available=_openai_credentials,
    ),
    RealGeneratorCase(
        id="azure_openai_responses_chat",
        class_path="haystack.components.generators.chat.azure_responses:AzureOpenAIResponsesChatGenerator",
        kind="chat",
        provider="azure.ai.openai",
        model_env="BG_AZURE_OPENAI_RESPONSES_DEPLOYMENT",
        default_model=OPENAI_MODEL,
        init_kwargs=_azure_responses_init,
        generation_kwargs={"max_output_tokens": MAX_TOKENS},
        credentials_available=_azure_api_key_credentials,
    ),
    RealGeneratorCase(
        id="anthropic_chat",
        class_path="haystack_integrations.components.generators.anthropic:AnthropicChatGenerator",
        kind="chat",
        provider="anthropic",
        model_env="BG_ANTHROPIC_MODEL",
        default_model=ANTHROPIC_MODEL,
        init_kwargs=_model_init,
        generation_kwargs={"max_tokens": MAX_TOKENS},
        credentials_available=_anthropic_credentials,
    ),
    RealGeneratorCase(
        id="google_genai_chat",
        class_path="haystack_integrations.components.generators.google_genai:GoogleGenAIChatGenerator",
        kind="chat",
        provider="gcp.gemini",
        model_env="BG_GOOGLE_MODEL",
        default_model=GOOGLE_MODEL,
        init_kwargs=_model_init,
        generation_kwargs={"max_output_tokens": MAX_TOKENS},
        credentials_available=_google_credentials,
    ),
    RealGeneratorCase(
        id="amazon_bedrock_chat",
        class_path="haystack_integrations.components.generators.amazon_bedrock:AmazonBedrockChatGenerator",
        kind="chat",
        provider="aws.bedrock",
        model_env="BG_BEDROCK_MODEL",
        default_model=BEDROCK_MODEL,
        init_kwargs=_bedrock_init,
        generation_kwargs={"maxTokens": MAX_TOKENS},
        credentials_available=_bedrock_credentials,
    ),
]

TEXT_CASES = [
    RealGeneratorCase(
        id="openai_text",
        class_path="haystack.components.generators.openai:OpenAIGenerator",
        kind="text",
        provider="openai",
        model_env="BG_OPENAI_MODEL",
        default_model=OPENAI_MODEL,
        init_kwargs=_openai_init,
        generation_kwargs={"max_completion_tokens": MAX_TOKENS},
        credentials_available=_openai_credentials,
    ),
    RealGeneratorCase(
        id="azure_openai_text",
        class_path="haystack.components.generators.azure:AzureOpenAIGenerator",
        kind="text",
        provider="azure.ai.openai",
        model_env="BG_AZURE_OPENAI_DEPLOYMENT",
        default_model=OPENAI_MODEL,
        init_kwargs=_azure_init,
        generation_kwargs={"max_completion_tokens": MAX_TOKENS},
        credentials_available=_azure_credentials,
    ),
    RealGeneratorCase(
        id="anthropic_text",
        class_path="haystack_integrations.components.generators.anthropic:AnthropicGenerator",
        kind="text",
        provider="anthropic",
        model_env="BG_ANTHROPIC_MODEL",
        default_model=ANTHROPIC_MODEL,
        init_kwargs=_model_init,
        generation_kwargs={"max_tokens": MAX_TOKENS},
        credentials_available=_anthropic_credentials,
    ),
    RealGeneratorCase(
        id="amazon_bedrock_text",
        class_path="haystack_integrations.components.generators.amazon_bedrock:AmazonBedrockGenerator",
        kind="text",
        provider="aws.bedrock",
        model_env="BG_BEDROCK_TEXT_MODEL",
        default_model=BEDROCK_MODEL,
        init_kwargs=_bedrock_text_init,
        generation_kwargs={"max_tokens": MAX_TOKENS},
        credentials_available=_bedrock_credentials,
        init_accepts_generation_kwargs=False,
        # The legacy InvokeModel component returns AWS ResponseMetadata, not model usage metadata.
        expect_usage=False,
        expect_finish_reasons=False,
    ),
]

# Keep targeted end-to-end tests to a single actual provider to avoid multiplying
# live LLM calls while still exercising real ChatGenerator behavior.
OPENAI_CHAT_CASES = [case for case in CHAT_CASES if case.id == "openai_chat"]
AGENT_CASES = OPENAI_CHAT_CASES

GENERATOR_RUN_VARIANTS = [
    RealGeneratorRunVariant(id="non_streaming", streaming=False),
    RealGeneratorRunVariant(id="streaming", streaming=True),
]


@pytest.fixture(autouse=True)
def reset_haystack_tracing() -> None:
    reset_haystack_tracing_state()
    yield
    reset_haystack_tracing_state()


def _import_class(class_path: str) -> type[Any]:
    module_name, class_name = class_path.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        pytest.skip(f"missing optional integration package for {class_path}: {error}")
    return getattr(module, class_name)


def _agent_generation_kwargs(case: RealGeneratorCase) -> dict[str, Any]:
    """Return generation kwargs with enough tokens for agent tool use.

    Args:
        case: Generator test case.

    Returns:
        Generation kwargs for agent tests.
    """
    generation_kwargs = dict(case.generation_kwargs)
    for key in ("max_tokens", "max_completion_tokens", "max_output_tokens", "maxTokens"):
        if key in generation_kwargs:
            generation_kwargs[key] = max(int(generation_kwargs[key]), AGENT_MAX_TOKENS)
    return generation_kwargs


_OPENAI_CHAT_COMPLETIONS_STREAM_USAGE_CASE_IDS = {
    "openai_chat",
    "azure_openai_chat",
    "openai_text",
    "azure_openai_text",
}


def _streaming_generation_kwargs(case: RealGeneratorCase) -> dict[str, Any]:
    """Return runtime kwargs needed for complete streaming metadata.

    Args:
        case: Generator test case.

    Returns:
        Provider-specific streaming kwargs.
    """
    if case.id in _OPENAI_CHAT_COMPLETIONS_STREAM_USAGE_CASE_IDS:
        # OpenAI Chat Completions streams include token usage only when this option is set.
        # Haystack's OpenAIGenerator/AzureOpenAIGenerator use Chat Completions under the hood, too.
        return {"stream_options": {"include_usage": True}}
    return {}


def _make_bg_exporter(bg_live_export_config: Any | None) -> tuple[InMemorySpanExporter, TracerProvider]:
    exporter = InMemorySpanExporter()
    resource_attributes = {
        "service.name": "bg-real-llm-tests",
        "haystack.pipeline.name": "blueguardrails-haystack-integration-tests",
        "test.suite": "real-llm-generators",
    }
    if bg_live_export_config is not None:
        resource_attributes["blueguardrails.test.live_export"] = True

    provider = TracerProvider(resource=Resource.create(resource_attributes))
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    if bg_live_export_config is not None:
        provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(
                    endpoint=bg_live_export_config.endpoint,
                    headers={"Authorization": f"Bearer {bg_live_export_config.api_key}"},
                )
            )
        )

    install_bg_tracer(BGTracer(provider))
    return exporter, provider


def _run_case(
    case: RealGeneratorCase, bg_live_export_config: Any | None, variant: RealGeneratorRunVariant
) -> tuple[dict[str, Any], Any, str]:
    available, reason = case.credentials_available()
    if not available:
        pytest.skip(reason)

    generator_cls = _import_class(case.class_path)
    model = os.getenv(case.model_env, case.default_model)
    init_kwargs = case.init_kwargs(model)
    if case.init_accepts_generation_kwargs:
        init_kwargs["generation_kwargs"] = dict(case.generation_kwargs)
    if case.kind == "chat":
        init_kwargs.setdefault("tools", [_integration_test_tool()])
    generator = generator_cls(**init_kwargs)

    run_input: dict[str, Any]
    if case.kind == "chat":
        run_input = {"messages": [ChatMessage.from_system(SYSTEM_PROMPT), ChatMessage.from_user(PROMPT)]}
    else:
        run_input = {"prompt": PROMPT}

    streaming_chunks: list[Any] = []
    if variant.streaming:

        def _streaming_callback(chunk: Any) -> None:
            streaming_chunks.append(chunk)

        run_input["streaming_callback"] = _streaming_callback
        streaming_generation_kwargs = _streaming_generation_kwargs(case)
        if streaming_generation_kwargs:
            run_input["generation_kwargs"] = streaming_generation_kwargs

    exporter, provider = _make_bg_exporter(bg_live_export_config)
    try:
        pipe = Pipeline()
        pipe.add_component("llm", generator)
        pipeline_output = pipe.run({"llm": run_input})
        component_output = pipeline_output["llm"]

        if variant.streaming:
            assert streaming_chunks, "streaming callback should receive at least one chunk"

        flushed = provider.force_flush()
        if bg_live_export_config is not None and not flushed:
            pytest.fail("timed out flushing spans to Blue Guardrails")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1, f"expected exactly one BG span, got {[span.name for span in spans]}"
        span = spans[0]
        _assert_semconv_span(case, span, model, streaming=variant.streaming)
        _record_fixture(case, model, init_kwargs, run_input, component_output, span, variant)
        return run_input, component_output, model
    finally:
        provider.shutdown()


def _text_from_semconv_message(message: dict[str, Any]) -> str:
    return "".join(str(part.get("content", "")) for part in message.get("parts", []) if part.get("type") == "text")


def _assert_semconv_span(
    case: RealGeneratorCase, span: Any, model: str, expected_prompt: str = PROMPT, streaming: bool = False
) -> None:
    attrs = span.attributes

    assert span.kind.name == "CLIENT"
    assert attrs["gen_ai.operation.name"] == case.operation_name
    assert attrs["gen_ai.provider.name"] == case.provider
    assert attrs["gen_ai.request.model"] == model
    assert attrs["gen_ai.response.model"]
    assert span.name == f"{case.operation_name} {model}"
    assert attrs["gen_ai.request.max_tokens"] == MAX_TOKENS
    if streaming:
        assert attrs["gen_ai.request.stream"] is True
    assert "gen_ai.agent.run.tags.pipeline_run_id" in attrs
    assert "haystack.pipeline.run_id" not in attrs

    input_messages = json.loads(attrs["gen_ai.input.messages"])
    assert input_messages
    assert input_messages[-1]["role"] == "user"
    assert expected_prompt in _text_from_semconv_message(input_messages[-1])

    output_messages = json.loads(attrs["gen_ai.output.messages"])
    assert output_messages
    assert output_messages[0]["role"] == "assistant"
    assert output_messages[0]["parts"], "assistant output must include captured content parts"

    if case.kind == "chat":
        tool_defs = json.loads(attrs["gen_ai.tool.definitions"])
        assert tool_defs[0]["type"] == "function"
        assert tool_defs[0]["name"] == "blueguardrails_noop"
        assert tool_defs[0]["parameters"]["properties"]["value"]["type"] == "string"

    finish_reasons_present = "gen_ai.response.finish_reasons" in attrs
    if case.expect_finish_reasons and (not streaming or finish_reasons_present):
        assert attrs["gen_ai.response.finish_reasons"]

    if case.expect_usage:
        assert attrs["gen_ai.usage.input_tokens"] > 0
        assert attrs["gen_ai.usage.output_tokens"] >= 0


def _tool_call_arguments(part: dict[str, Any]) -> dict[str, Any]:
    arguments = part.get("arguments")
    if isinstance(arguments, str):
        return json.loads(arguments)
    assert isinstance(arguments, dict)
    return arguments


def _assert_agent_semconv_spans(case: RealGeneratorCase, spans: list[Any], model: str) -> None:
    assert len(spans) >= 2, f"expected at least one Agent tool loop, got {[span.name for span in spans]}"

    pipeline_run_ids = set()
    for span in spans:
        attrs = span.attributes
        assert span.kind.name == "CLIENT"
        assert attrs["gen_ai.operation.name"] == "chat"
        assert attrs["gen_ai.provider.name"] == case.provider
        assert attrs["gen_ai.request.model"] == model
        assert attrs["gen_ai.response.model"]
        assert span.name == f"chat {model}"
        pipeline_run_ids.add(attrs["gen_ai.agent.run.tags.pipeline_run_id"])

        tool_defs = json.loads(attrs["gen_ai.tool.definitions"])
        assert tool_defs[0]["type"] == "function"
        assert tool_defs[0]["name"] == "blueguardrails_noop"

    assert len(pipeline_run_ids) == 1

    first_span = spans[0]
    first_input_messages = json.loads(first_span.attributes["gen_ai.input.messages"])
    assert first_input_messages[-1]["role"] == "user"
    assert AGENT_PROMPT in _text_from_semconv_message(first_input_messages[-1])

    first_output_messages = json.loads(first_span.attributes["gen_ai.output.messages"])
    assert first_output_messages[0]["role"] == "assistant"
    tool_call_parts = [part for part in first_output_messages[0]["parts"] if part.get("type") == "tool_call"]
    assert tool_call_parts
    assert tool_call_parts[0]["name"] == "blueguardrails_noop"
    assert _tool_call_arguments(tool_call_parts[0])["value"] == AGENT_TOOL_VALUE

    final_span = spans[-1]
    final_input_messages = json.loads(final_span.attributes["gen_ai.input.messages"])
    tool_messages = [message for message in final_input_messages if message.get("role") == "tool"]
    assert tool_messages
    tool_response_parts = [
        part
        for message in tool_messages
        for part in message.get("parts", [])
        if part.get("type") == "tool_call_response"
    ]
    assert tool_response_parts
    assert tool_response_parts[0]["response"] == AGENT_TOOL_VALUE

    final_output_messages = json.loads(final_span.attributes["gen_ai.output.messages"])
    assert final_output_messages[0]["role"] == "assistant"
    assert not any(part.get("type") == "tool_call" for part in final_output_messages[0]["parts"])
    assert AGENT_TOOL_VALUE in _text_from_semconv_message(final_output_messages[0])


def _jsonable(value: Any) -> Any:
    if isinstance(value, ChatMessage):
        return _jsonable(value.to_dict())
    tool_spec = getattr(value, "tool_spec", None)
    if isinstance(tool_spec, dict):
        return _jsonable(tool_spec)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bytes):
        return {"__bytes_base64__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if callable(value):
        module = getattr(value, "__module__", type(value).__module__)
        name = getattr(value, "__qualname__", type(value).__name__)
        return f"{module}.{name}"
    return str(value)


def _record_fixture(
    case: RealGeneratorCase,
    model: str,
    init_kwargs: dict[str, Any],
    run_input: dict[str, Any],
    output: Any,
    span: Any,
    variant: RealGeneratorRunVariant,
) -> None:
    if not _truthy_env("BG_RECORD_LLM_FIXTURES"):
        return

    fixture_dir = Path(os.getenv("BG_LLM_FIXTURE_DIR", "tests/fixtures/llm_io"))
    fixture_dir.mkdir(parents=True, exist_ok=True)

    attrs = {key: _jsonable(value) for key, value in span.attributes.items()}
    semconv_payloads = {}
    for key in ("gen_ai.input.messages", "gen_ai.output.messages", "gen_ai.tool.definitions"):
        if isinstance(attrs.get(key), str):
            semconv_payloads[key] = json.loads(attrs[key])

    fixture = {
        "case_id": case.id,
        "class_path": case.class_path,
        "kind": case.kind,
        "provider": case.provider,
        "model": model,
        "variant": {"id": variant.id, "streaming": variant.streaming},
        "generated_at": datetime.now(UTC).isoformat(),
        "raw_init": _jsonable(init_kwargs),
        "raw_input": _jsonable(run_input),
        "raw_output": _jsonable(output),
        "span": {
            "name": span.name,
            "kind": span.kind.name,
            "attributes": attrs,
        },
        "semconv": semconv_payloads,
    }

    fixture_name = case.id if not variant.streaming else f"{case.id}_streaming"
    path = fixture_dir / f"{fixture_name}.json"
    path.write_text(json.dumps(fixture, indent=2, sort_keys=True), encoding="utf-8")


@pytest.mark.parametrize("case", CHAT_CASES, ids=[case.id for case in CHAT_CASES])
@pytest.mark.parametrize("variant", GENERATOR_RUN_VARIANTS, ids=[variant.id for variant in GENERATOR_RUN_VARIANTS])
def test_real_chat_generator_tracing(
    case: RealGeneratorCase, variant: RealGeneratorRunVariant, bg_live_export_config: Any | None
) -> None:
    """Verify each supported chat generator emits a complete GenAI span.

    Args:
        case: Generator test case.
        variant: Streaming or non-streaming run variant.
        bg_live_export_config: Optional live export configuration.
    """
    _run_case(case, bg_live_export_config, variant)


@pytest.mark.parametrize("case", TEXT_CASES, ids=[case.id for case in TEXT_CASES])
@pytest.mark.parametrize("variant", GENERATOR_RUN_VARIANTS, ids=[variant.id for variant in GENERATOR_RUN_VARIANTS])
def test_real_text_generator_tracing(
    case: RealGeneratorCase, variant: RealGeneratorRunVariant, bg_live_export_config: Any | None
) -> None:
    """Verify each supported text generator emits a complete GenAI span.

    Args:
        case: Generator test case.
        variant: Streaming or non-streaming run variant.
        bg_live_export_config: Optional live export configuration.
    """
    _run_case(case, bg_live_export_config, variant)


@pytest.mark.parametrize("case", OPENAI_CHAT_CASES, ids=[case.id for case in OPENAI_CHAT_CASES])
def test_real_pipeline_with_two_chat_generators(case: RealGeneratorCase, bg_live_export_config: Any | None) -> None:
    """Verify a two-generator pipeline emits correlated Blue Guardrails spans.

    Args:
        case: Generator test case.
        bg_live_export_config: Optional live export configuration.
    """
    available, reason = case.credentials_available()
    if not available:
        pytest.skip(reason)

    generator_cls = _import_class(case.class_path)
    model = os.getenv(case.model_env, case.default_model)

    def _make_generator() -> Any:
        init_kwargs = case.init_kwargs(model)
        if case.init_accepts_generation_kwargs:
            init_kwargs["generation_kwargs"] = dict(case.generation_kwargs)
        init_kwargs.setdefault("tools", [_integration_test_tool()])
        return generator_cls(**init_kwargs)

    exporter, provider = _make_bg_exporter(bg_live_export_config)
    try:
        pipe = Pipeline()
        pipe.add_component("first_llm", _make_generator())
        pipe.add_component("second_llm", _make_generator())
        pipeline_output = pipe.run(
            {
                component_name: {"messages": [ChatMessage.from_system(SYSTEM_PROMPT), ChatMessage.from_user(prompt)]}
                for component_name, prompt in TWO_GENERATOR_PIPELINE_PROMPTS.items()
            }
        )

        assert set(pipeline_output) == set(TWO_GENERATOR_PIPELINE_PROMPTS)
        for component_output in pipeline_output.values():
            assert component_output["replies"], "each generator should produce at least one reply"

        flushed = provider.force_flush()
        if bg_live_export_config is not None and not flushed:
            pytest.fail("timed out flushing two-generator Pipeline spans to Blue Guardrails")

        spans = exporter.get_finished_spans()
        assert len(spans) == 2, f"expected two BG spans, got {[span.name for span in spans]}"

        component_names = {span.attributes.get("haystack.component.name") for span in spans}
        assert component_names == set(TWO_GENERATOR_PIPELINE_PROMPTS)

        pipeline_run_ids = {span.attributes["gen_ai.agent.run.tags.pipeline_run_id"] for span in spans}
        assert len(pipeline_run_ids) == 1
        assert uuid.UUID(str(next(iter(pipeline_run_ids))))

        for span in spans:
            component_name = span.attributes["haystack.component.name"]
            assert span.attributes["gen_ai.conversation.tags.haystack_component_name"] == component_name
            _assert_semconv_span(case, span, model, TWO_GENERATOR_PIPELINE_PROMPTS[component_name])
    finally:
        provider.shutdown()


@pytest.mark.parametrize("case", OPENAI_CHAT_CASES, ids=[case.id for case in OPENAI_CHAT_CASES])
def test_real_chat_generator_pipeline_run_id_is_agent_run_tag(
    case: RealGeneratorCase, bg_live_export_config: Any | None
) -> None:
    """Verify a chat generator span includes the pipeline run tag.

    Args:
        case: Generator test case.
        bg_live_export_config: Optional live export configuration.
    """
    available, reason = case.credentials_available()
    if not available:
        pytest.skip(reason)

    generator_cls = _import_class(case.class_path)
    model = os.getenv(case.model_env, case.default_model)
    init_kwargs = case.init_kwargs(model)
    if case.init_accepts_generation_kwargs:
        init_kwargs["generation_kwargs"] = case.generation_kwargs
    generator = generator_cls(**init_kwargs)

    exporter, provider = _make_bg_exporter(bg_live_export_config)
    try:
        pipe = Pipeline()
        pipe.add_component("llm", generator)
        pipe.run({"llm": {"messages": [ChatMessage.from_user(PROMPT)]}})

        flushed = provider.force_flush()
        if bg_live_export_config is not None and not flushed:
            pytest.fail("timed out flushing run-tag span to Blue Guardrails")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = spans[0].attributes
        pipeline_run_id = attrs["gen_ai.agent.run.tags.pipeline_run_id"]
        assert uuid.UUID(str(pipeline_run_id))
        assert "haystack.pipeline.run_id" not in attrs
    finally:
        provider.shutdown()


@pytest.mark.parametrize("case", OPENAI_CHAT_CASES, ids=[case.id for case in OPENAI_CHAT_CASES])
def test_real_connector_tags_are_conversation_tags(
    case: RealGeneratorCase, monkeypatch: pytest.MonkeyPatch, bg_live_export_config: Any | None
) -> None:
    """Verify connector tags become GenAI conversation tag attributes.

    Args:
        case: Generator test case.
        monkeypatch: Pytest monkeypatch fixture.
        bg_live_export_config: Optional live export configuration.
    """
    available, reason = case.credentials_available()
    if not available:
        pytest.skip(reason)

    generator_cls = _import_class(case.class_path)
    model = os.getenv(case.model_env, case.default_model)
    init_kwargs = case.init_kwargs(model)
    if case.init_accepts_generation_kwargs:
        init_kwargs["generation_kwargs"] = case.generation_kwargs
    generator = generator_cls(**init_kwargs)

    class RecordingExporter:
        def __init__(self, real_exporter: Any | None = None) -> None:
            self.memory_exporter = InMemorySpanExporter()
            self.real_exporter = real_exporter

        def export(self, spans: Any) -> Any:
            result = self.memory_exporter.export(spans)
            if self.real_exporter is not None:
                return self.real_exporter.export(spans)
            return result

        def shutdown(self) -> None:
            if self.real_exporter is not None:
                self.real_exporter.shutdown()
            self.memory_exporter.shutdown()

        def force_flush(self, timeout_millis: int = 30000) -> bool:
            memory_flushed = self.memory_exporter.force_flush(timeout_millis)
            if self.real_exporter is None:
                return memory_flushed
            real_force_flush = getattr(self.real_exporter, "force_flush", None)
            real_flushed = bool(real_force_flush(timeout_millis)) if callable(real_force_flush) else True
            return real_flushed and memory_flushed

    recording_exporter: RecordingExporter | None = None

    def _make_recording_exporter(endpoint: str, headers: dict[str, str]) -> RecordingExporter:
        nonlocal recording_exporter
        real_exporter = (
            OTLPSpanExporter(endpoint=endpoint, headers=headers) if bg_live_export_config is not None else None
        )
        recording_exporter = RecordingExporter(real_exporter)
        return recording_exporter

    monkeypatch.setattr(connector_module, "OTLPSpanExporter", _make_recording_exporter)

    tags = {"env": "integration", "customer": "acme", "team": "growth"}
    connector = BlueGuardrailsConnector(
        name="real-generator-conversation-tags",
        endpoint=(
            bg_live_export_config.endpoint if bg_live_export_config is not None else "http://localhost:4318/v1/traces"
        ),
        api_key=Secret.from_token(bg_live_export_config.api_key if bg_live_export_config is not None else "test-key"),
        tags=tags,
    )
    try:
        pipe = Pipeline()
        pipe.add_component("llm", generator)
        pipe.run({"llm": {"messages": [ChatMessage.from_user(PROMPT)]}})

        assert connector._provider.force_flush()
        assert recording_exporter is not None
        spans = recording_exporter.memory_exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = spans[0].attributes
        for key, value in tags.items():
            assert attrs[f"gen_ai.conversation.tags.{key}"] == value
            assert key not in attrs
            assert key not in spans[0].resource.attributes
    finally:
        connector._provider.shutdown()


@pytest.mark.parametrize("case", AGENT_CASES, ids=[case.id for case in AGENT_CASES])
def test_real_haystack_agent_tracing(case: RealGeneratorCase, bg_live_export_config: Any | None) -> None:
    """Verify a Haystack Agent emits GenAI spans for LLM calls.

    Args:
        case: Generator test case.
        bg_live_export_config: Optional live export configuration.
    """
    available, reason = case.credentials_available()
    if not available:
        pytest.skip(reason)

    generator_cls = _import_class(case.class_path)
    model = os.getenv(case.model_env, case.default_model)
    init_kwargs = case.init_kwargs(model)
    if case.init_accepts_generation_kwargs:
        init_kwargs["generation_kwargs"] = _agent_generation_kwargs(case)
    generator = generator_cls(**init_kwargs)

    tool = _integration_test_tool()
    agent = Agent(
        chat_generator=generator,
        tools=[tool],
        system_prompt=SYSTEM_PROMPT,
        exit_conditions=["text"],
        max_agent_steps=3,
    )

    exporter, provider = _make_bg_exporter(bg_live_export_config)
    try:
        result = agent.run(messages=[ChatMessage.from_user(AGENT_PROMPT)])

        flushed = provider.force_flush()
        if bg_live_export_config is not None and not flushed:
            pytest.fail("timed out flushing Agent spans to Blue Guardrails")

        tool_results = [message.tool_call_result for message in result["messages"] if message.tool_call_result]
        assert tool_results
        assert tool_results[0].result == AGENT_TOOL_VALUE
        assert tool_results[0].error is False

        last_message = result["last_message"]
        assert last_message.tool_call_result is None
        assert last_message.text is not None
        assert AGENT_TOOL_VALUE in last_message.text

        spans = exporter.get_finished_spans()
        _assert_agent_semconv_spans(case, spans, model)
    finally:
        provider.shutdown()
