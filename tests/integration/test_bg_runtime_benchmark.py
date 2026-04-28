# SPDX-FileCopyrightText: 2025-present Blue Guardrails
#
# SPDX-License-Identifier: Apache-2.0

"""Runtime benchmark for Haystack pipeline runs with and without live BG tracing.

This benchmark uses a mock chat generator so LLM latency is deterministic and the
measured delta is the tracing/export path rather than provider latency. By default
it sends 100k+ character input and output messages on every measured run. It only
runs when selected explicitly with pytest's benchmark marker and Blue Guardrails
live export is enabled, for example:

    uv run --extra integration pytest -m benchmark --bg-send-traces tests/integration/test_bg_runtime_benchmark.py
"""

from __future__ import annotations

import json
import os
import statistics
import time
from typing import Any

import pytest
from conftest import reset_haystack_tracing_state
from haystack import Pipeline, component
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

from blueguardrails_haystack import BlueGuardrailsConnector

pytestmark = pytest.mark.benchmark

BENCHMARK_PROMPT = "Reply with exactly: blueguardrails benchmark ok"
BENCHMARK_RESPONSE = "blueguardrails benchmark ok"


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        pytest.fail(f"{name} must be an integer, got {value!r}")


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        pytest.fail(f"{name} must be a float, got {value!r}")


def _reset_haystack_tracing() -> None:
    reset_haystack_tracing_state()


@component
class BenchmarkMockChatGenerator:
    """Deterministic chat generator for runtime measurements."""

    def __init__(self, latency_seconds: float, response_text: str) -> None:
        self.model = "benchmark-mock-model"
        self.latency_seconds = latency_seconds
        self.response_text = response_text

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage]) -> dict[str, list[ChatMessage]]:  # noqa: ARG002
        if self.latency_seconds > 0:
            time.sleep(self.latency_seconds)
        return {
            "replies": [
                ChatMessage.from_assistant(
                    self.response_text,
                    meta={
                        "model": self.model,
                        "finish_reason": "stop",
                        "usage": {"prompt_tokens": 8, "completion_tokens": 4},
                    },
                )
            ]
        }


def _make_large_text(label: str, target_chars: int) -> str:
    if target_chars <= 0:
        return ""
    prefix = f"{label} chars={target_chars}\n"
    chunk = (
        "Blue Guardrails tracing benchmark payload. "
        "This deterministic text makes serialization/export overhead visible. "
    )
    repeats = max(0, (target_chars - len(prefix) + len(chunk) - 1) // len(chunk))
    return (prefix + (chunk * repeats))[:target_chars]


def _make_pipeline(latency_seconds: float, response_text: str) -> Pipeline:
    pipe = Pipeline()
    pipe.add_component(
        "llm",
        BenchmarkMockChatGenerator(latency_seconds=latency_seconds, response_text=response_text),
    )
    return pipe


def _run_pipeline_once(pipe: Pipeline, prompt_text: str) -> None:
    pipe.run({"llm": {"messages": [ChatMessage.from_user(prompt_text)]}})


def _measure_pipeline_runtime(pipe: Pipeline, runs: int, prompt_text: str) -> list[float]:
    durations = []
    for _ in range(runs):
        started = time.perf_counter()
        _run_pipeline_once(pipe, prompt_text)
        durations.append(time.perf_counter() - started)
    return durations


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile for an empty list")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((percentile / 100) * (len(ordered) - 1))))
    return ordered[index]


def _duration_stats(values: list[float]) -> dict[str, float]:
    return {
        "min_ms": min(values) * 1000,
        "median_ms": statistics.median(values) * 1000,
        "mean_ms": statistics.fmean(values) * 1000,
        "p95_ms": _percentile(values, 95) * 1000,
        "max_ms": max(values) * 1000,
    }


def test_pipeline_runtime_with_and_without_live_bg_tracing(
    bg_live_export_config: Any | None, record_property: pytest.RecordProperty
) -> None:
    """Measure pipeline runtime with and without live Blue Guardrails export.

    Args:
        bg_live_export_config: Live export configuration.
        record_property: Pytest fixture for recording benchmark metrics.
    """
    if bg_live_export_config is None:
        pytest.skip("enable with --bg-send-traces or BG_SEND_TRACES=1 to benchmark live Blue Guardrails export")

    runs = _env_int("BG_BENCHMARK_RUNS", 30)
    warmup_runs = _env_int("BG_BENCHMARK_WARMUP_RUNS", 5)
    latency_seconds = _env_float("BG_BENCHMARK_MOCK_LATENCY_MS", 1.0) / 1000
    input_chars = _env_int("BG_BENCHMARK_INPUT_CHARS", 100_000)
    output_chars = _env_int("BG_BENCHMARK_OUTPUT_CHARS", 100_000)
    if runs < 2:
        pytest.fail("BG_BENCHMARK_RUNS must be at least 2")
    if warmup_runs < 0:
        pytest.fail("BG_BENCHMARK_WARMUP_RUNS must be non-negative")
    if input_chars < 0 or output_chars < 0:
        pytest.fail("BG_BENCHMARK_INPUT_CHARS and BG_BENCHMARK_OUTPUT_CHARS must be non-negative")

    prompt_text = _make_large_text("benchmark input", input_chars) or BENCHMARK_PROMPT
    response_text = _make_large_text("benchmark output", output_chars) or BENCHMARK_RESPONSE

    _reset_haystack_tracing()
    baseline_pipe = _make_pipeline(latency_seconds, response_text)
    for _ in range(warmup_runs):
        _run_pipeline_once(baseline_pipe, prompt_text)
    baseline_durations = _measure_pipeline_runtime(baseline_pipe, runs, prompt_text)

    _reset_haystack_tracing()
    connector = BlueGuardrailsConnector(
        name="pipeline-runtime-benchmark",
        endpoint=bg_live_export_config.endpoint,
        api_key=Secret.from_token(bg_live_export_config.api_key),
        tags={"benchmark": "pipeline_runtime", "generator": "mock_chat"},
    )
    try:
        traced_pipe = _make_pipeline(latency_seconds, response_text)
        for _ in range(warmup_runs):
            _run_pipeline_once(traced_pipe, prompt_text)
        assert connector._provider.force_flush(), "timed out flushing warmup spans to Blue Guardrails"

        traced_durations = _measure_pipeline_runtime(traced_pipe, runs, prompt_text)
        flush_started = time.perf_counter()
        assert connector._provider.force_flush(), "timed out flushing measured spans to Blue Guardrails"
        flush_seconds = time.perf_counter() - flush_started
    finally:
        connector._provider.shutdown()
        _reset_haystack_tracing()

    baseline_stats = _duration_stats(baseline_durations)
    traced_stats = _duration_stats(traced_durations)
    median_overhead_ms = traced_stats["median_ms"] - baseline_stats["median_ms"]
    mean_overhead_ms = traced_stats["mean_ms"] - baseline_stats["mean_ms"]
    median_ratio = traced_stats["median_ms"] / baseline_stats["median_ms"]
    mean_ratio = traced_stats["mean_ms"] / baseline_stats["mean_ms"]

    summary = {
        "runs": runs,
        "warmup_runs": warmup_runs,
        "mock_latency_ms": latency_seconds * 1000,
        "input_chars": len(prompt_text),
        "output_chars": len(response_text),
        "payload_chars_per_run": len(prompt_text) + len(response_text),
        "without_tracing": baseline_stats,
        "with_bg_tracing": traced_stats,
        "median_overhead_ms": median_overhead_ms,
        "mean_overhead_ms": mean_overhead_ms,
        "median_overhead_ratio": median_ratio,
        "mean_overhead_ratio": mean_ratio,
        "bg_force_flush_ms": flush_seconds * 1000,
    }

    for key, value in {
        "bg_benchmark_runs": runs,
        "bg_benchmark_warmup_runs": warmup_runs,
        "bg_benchmark_mock_latency_ms": latency_seconds * 1000,
        "bg_benchmark_input_chars": len(prompt_text),
        "bg_benchmark_output_chars": len(response_text),
        "bg_benchmark_payload_chars_per_run": len(prompt_text) + len(response_text),
        "bg_benchmark_without_tracing_median_ms": baseline_stats["median_ms"],
        "bg_benchmark_with_tracing_median_ms": traced_stats["median_ms"],
        "bg_benchmark_median_overhead_ms": median_overhead_ms,
        "bg_benchmark_median_overhead_ratio": median_ratio,
        "bg_benchmark_force_flush_ms": flush_seconds * 1000,
    }.items():
        record_property(key, value)

    print("\nBlue Guardrails pipeline runtime benchmark:")
    print(json.dumps(summary, indent=2, sort_keys=True))

    max_median_ratio = os.getenv("BG_BENCHMARK_MAX_MEDIAN_OVERHEAD_RATIO")
    if max_median_ratio is not None:
        assert median_ratio <= float(max_median_ratio)
