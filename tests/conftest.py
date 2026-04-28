# SPDX-FileCopyrightText: 2025-present Blue Guardrails
#
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest configuration for Blue Guardrails tests."""

import os
from dataclasses import dataclass

import pytest
from haystack import tracing
from haystack.tracing.tracer import ProxyTracer

import blueguardrails_haystack.proxy as blueguardrails_proxy

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

_DEFAULT_BLUEGUARDRAILS_ENDPOINT = "https://app.blueguardrails.com/v1/traces"


@dataclass(frozen=True)
class BlueGuardrailsLiveExportConfig:
    endpoint: str
    api_key: str


def reset_haystack_tracing_state() -> None:
    """Reset Haystack's global tracing proxy to its vanilla state."""
    tracing.disable_tracing()
    tracing.tracer.__class__ = ProxyTracer
    tracing.tracer._blueguardrails_tracer = None
    tracing.tracer._blueguardrails_enabled = False

    if blueguardrails_proxy._pipeline_span_patched and blueguardrails_proxy._original_create_component_span is not None:
        from haystack.core.pipeline.base import PipelineBase

        PipelineBase._create_component_span = staticmethod(blueguardrails_proxy._original_create_component_span)

    blueguardrails_proxy._pipeline_span_patched = False
    blueguardrails_proxy._original_create_component_span = None


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").lower() in {"1", "true", "yes", "on"}


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("blueguardrails")
    group.addoption(
        "--blueguardrails-send-traces",
        action="store_true",
        default=False,
        help=(
            "Export integration/benchmark test spans to Blue Guardrails. "
            "Can also be enabled with BLUEGUARDRAILS_SEND_TRACES=1. Requires BLUEGUARDRAILS_API_KEY."
        ),
    )
    group.addoption(
        "--blueguardrails-endpoint",
        action="store",
        default=None,
        help=(
            "Blue Guardrails OTLP HTTP traces endpoint to use with --blueguardrails-send-traces. "
            "Defaults to BLUEGUARDRAILS_ENDPOINT or the production endpoint."
        ),
    )


@pytest.fixture
def blueguardrails_live_export_config(request: pytest.FixtureRequest) -> BlueGuardrailsLiveExportConfig | None:
    """Return live Blue Guardrails export settings when explicitly enabled.

    Args:
        request: Pytest request with command-line option access.

    Returns:
        Live export configuration, or ``None`` when export is disabled.
    """
    send_traces = request.config.getoption("--blueguardrails-send-traces") or _truthy_env("BLUEGUARDRAILS_SEND_TRACES")
    if not send_traces:
        return None

    api_key = os.getenv("BLUEGUARDRAILS_API_KEY")
    if not api_key:
        pytest.fail("--blueguardrails-send-traces/BLUEGUARDRAILS_SEND_TRACES requires BLUEGUARDRAILS_API_KEY")

    endpoint = (
        request.config.getoption("--blueguardrails-endpoint")
        or os.getenv("BLUEGUARDRAILS_ENDPOINT")
        or _DEFAULT_BLUEGUARDRAILS_ENDPOINT
    )
    return BlueGuardrailsLiveExportConfig(endpoint=endpoint, api_key=api_key)
