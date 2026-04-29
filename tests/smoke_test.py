# SPDX-FileCopyrightText: 2025-present Blue Guardrails
#
# SPDX-License-Identifier: Apache-2.0

"""Release smoke test for built distributions.

Run this script against an installed wheel or source distribution, not the local
source tree. It validates that package metadata includes the runtime
dependencies, the public API imports, and a minimal trace can be produced without
contacting external services.
"""

from haystack.dataclasses import ChatMessage
from haystack.tracing.tracer import tracer
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from blueguardrails_haystack import BlueGuardrailsConnector, BlueGuardrailsTracer, configure_blueguardrails_tracer


def assert_equal(actual: object, expected: object, message: str) -> None:
    """Raise a useful smoke-test failure when values differ."""
    if actual != expected:
        raise RuntimeError(f"{message}: expected {expected!r}, got {actual!r}")


def main() -> None:
    """Exercise the installed package through its public tracing API."""
    if BlueGuardrailsConnector is None:
        raise RuntimeError("BlueGuardrailsConnector was not imported")

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    configure_blueguardrails_tracer(
        BlueGuardrailsTracer(provider, conversation_tags={"smoke": "true"}),
        replace=True,
    )

    with tracer.trace(
        "haystack.component.run",
        tags={
            "haystack.component.name": "release_smoke_generator",
            "haystack.component.type": "OpenAIChatGenerator",
            "haystack.component.model": "gpt-4o-mini",
        },
    ) as span:
        span.set_content_tag(
            "haystack.component.input",
            {"messages": [ChatMessage.from_user("Hello from the release smoke test.")]},
        )
        span.set_content_tag(
            "haystack.component.output",
            {
                "replies": [
                    ChatMessage.from_assistant(
                        "Smoke test response.",
                        meta={
                            "model": "gpt-4o-mini",
                            "finish_reason": "stop",
                            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
                        },
                    )
                ]
            },
        )

    finished_spans = exporter.get_finished_spans()
    assert_equal(len(finished_spans), 1, "expected exactly one exported smoke-test span")

    otel_span = finished_spans[0]
    assert_equal(otel_span.name, "chat gpt-4o-mini", "unexpected span name")

    attributes = otel_span.attributes
    assert_equal(attributes.get("gen_ai.operation.name"), "chat", "unexpected operation name")
    assert_equal(attributes.get("gen_ai.provider.name"), "openai", "unexpected provider name")
    assert_equal(attributes.get("gen_ai.request.model"), "gpt-4o-mini", "unexpected request model")
    assert_equal(attributes.get("gen_ai.response.model"), "gpt-4o-mini", "unexpected response model")
    assert_equal(attributes.get("gen_ai.usage.input_tokens"), 5, "unexpected input token count")
    assert_equal(attributes.get("gen_ai.usage.output_tokens"), 3, "unexpected output token count")
    assert_equal(attributes.get("gen_ai.conversation.tags.smoke"), "true", "conversation tag was not exported")

    print("Release smoke test succeeded")


if __name__ == "__main__":
    main()
