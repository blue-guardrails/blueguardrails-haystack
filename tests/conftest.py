# SPDX-FileCopyrightText: 2025-present Blue Guardrails
#
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest configuration for Blue Guardrails tests."""

import os
from dataclasses import dataclass

import pytest
from haystack import tracing
from haystack.tracing.tracer import ProxyTracer

import blueguardrails_haystack.proxy as bg_proxy

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

_DEFAULT_BG_ENDPOINT = "https://app.blueguardrails.com/v1/traces"


@dataclass(frozen=True)
class BGLiveExportConfig:
    endpoint: str
    api_key: str


def reset_haystack_tracing_state() -> None:
    """Reset Haystack's global tracing proxy to its vanilla state."""
    tracing.disable_tracing()
    tracing.tracer.__class__ = ProxyTracer
    tracing.tracer._bg_tracer = None
    tracing.tracer._bg_enabled = False

    if bg_proxy._pipeline_span_patched and bg_proxy._original_create_component_span is not None:
        from haystack.core.pipeline.base import PipelineBase

        PipelineBase._create_component_span = staticmethod(bg_proxy._original_create_component_span)

    bg_proxy._pipeline_span_patched = False
    bg_proxy._original_create_component_span = None


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").lower() in {"1", "true", "yes", "on"}


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("blueguardrails")
    group.addoption(
        "--bg-send-traces",
        action="store_true",
        default=False,
        help=(
            "Export integration/benchmark test spans to Blue Guardrails. "
            "Can also be enabled with BG_SEND_TRACES=1. Requires BG_API_KEY."
        ),
    )
    group.addoption(
        "--bg-endpoint",
        action="store",
        default=None,
        help=(
            "Blue Guardrails OTLP HTTP traces endpoint to use with --bg-send-traces. "
            "Defaults to BG_ENDPOINT or the production endpoint."
        ),
    )


@pytest.fixture
def bg_live_export_config(request: pytest.FixtureRequest) -> BGLiveExportConfig | None:
    """Return live Blue Guardrails export settings when explicitly enabled.

    Args:
        request: Pytest request with command-line option access.

    Returns:
        Live export configuration, or ``None`` when export is disabled.
    """
    send_traces = request.config.getoption("--bg-send-traces") or _truthy_env("BG_SEND_TRACES")
    if not send_traces:
        return None

    api_key = os.getenv("BG_API_KEY")
    if not api_key:
        pytest.fail("--bg-send-traces/BG_SEND_TRACES requires BG_API_KEY")

    endpoint = request.config.getoption("--bg-endpoint") or os.getenv("BG_ENDPOINT") or _DEFAULT_BG_ENDPOINT
    return BGLiveExportConfig(endpoint=endpoint, api_key=api_key)
