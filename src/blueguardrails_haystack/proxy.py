# SPDX-FileCopyrightText: 2025-present Blue Guardrails
#
# SPDX-License-Identifier: Apache-2.0

"""Sidecar proxy that fans out Haystack trace calls to Blue Guardrails."""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

from haystack import logging
from haystack.tracing.tracer import NullSpan, NullTracer, ProxyTracer, Span

from blueguardrails_haystack.component_config import extract_component_config
from blueguardrails_haystack.span import BlueGuardrailsSpan, CompositeSpan

if TYPE_CHECKING:
    from haystack.utils import Secret

    from blueguardrails_haystack.tracer import BlueGuardrailsTracer

logger = logging.getLogger(__name__)

_pipeline_span_patched = False
_original_create_component_span: Any | None = None


def _blueguardrails_span(span: Span) -> BlueGuardrailsSpan | None:
    """Return the Blue Guardrails span inside a Haystack span, if present."""
    if isinstance(span, CompositeSpan) and isinstance(span.blueguardrails, BlueGuardrailsSpan):
        return span.blueguardrails
    if isinstance(span, BlueGuardrailsSpan):
        return span
    return None


def _set_blueguardrails_component_config(span: Span, config: dict[str, Any]) -> None:
    """Attach component init config only to the Blue Guardrails span."""
    if not config:
        return
    try:
        if blueguardrails_span := _blueguardrails_span(span):
            blueguardrails_span.set_component_config(config)
    except Exception as error:
        logger.warning("Blue Guardrails tracer skipped component config", error=repr(error))


def _instrument_streaming_callback(span: Span, inputs: dict[str, Any]) -> Any:
    """Wrap a component streaming callback long enough to record first-chunk latency."""
    blueguardrails_span = _blueguardrails_span(span)
    if blueguardrails_span is None:
        return None

    streaming_callback = inputs.get("streaming_callback")
    if callable(streaming_callback):
        wrapped_callback = blueguardrails_span.wrap_streaming_callback(streaming_callback)
        inputs["streaming_callback"] = wrapped_callback
        return "input", streaming_callback, wrapped_callback

    return None


def _restore_streaming_callback(inputs: dict[str, Any], callback_state: Any) -> None:
    """Restore the user's original streaming callback after the component run."""
    if callback_state is None:
        return
    location, original_callback, wrapped_callback = callback_state
    if location == "input" and inputs.get("streaming_callback") is wrapped_callback:
        inputs["streaming_callback"] = original_callback


def _patch_pipeline_component_span_for_models() -> None:
    """Patch Haystack component spans to expose generator init config to Blue Guardrails."""
    global _original_create_component_span, _pipeline_span_patched

    if _pipeline_span_patched:
        return

    try:
        from haystack.core.pipeline.base import PipelineBase
    except Exception as error:
        logger.warning("Blue Guardrails tracer could not patch Haystack pipeline spans", error=repr(error))
        return

    original_create_component_span = PipelineBase._create_component_span
    _original_create_component_span = original_create_component_span

    @staticmethod
    @contextlib.contextmanager
    def _create_component_span_with_blueguardrails_model(
        component_name: str, instance: Any, inputs: dict[str, Any], parent_span: Span | None = None
    ) -> Generator[Span]:
        with original_create_component_span(component_name, instance, inputs, parent_span) as span:
            _set_blueguardrails_component_config(span, extract_component_config(instance))
            callback_state = _instrument_streaming_callback(span, inputs)
            try:
                yield span
            finally:
                _restore_streaming_callback(inputs, callback_state)

    PipelineBase._create_component_span = _create_component_span_with_blueguardrails_model
    _pipeline_span_patched = True


def _enter_blueguardrails_trace(
    blueguardrails_tracer: BlueGuardrailsTracer,
    operation_name: str,
    tags: dict[str, Any] | None,
    parent_span: Span | None = None,
) -> tuple[Any, Span]:
    """Start a Blue Guardrails trace without propagating failures."""
    try:
        ctx = blueguardrails_tracer.trace(operation_name, tags=tags, parent_span=parent_span)
        span = ctx.__enter__()
        return ctx, span
    except Exception as error:
        logger.warning("Blue Guardrails tracer failed to start span", operation_name=operation_name, error=repr(error))
        return None, NullSpan()


def _exit_blueguardrails_trace(blueguardrails_ctx: Any | None, exc_info: tuple[Any, ...] | None = None) -> None:
    """End a Blue Guardrails trace without propagating failures."""
    if blueguardrails_ctx is None:
        return
    try:
        blueguardrails_ctx.__exit__(*exc_info) if exc_info else blueguardrails_ctx.__exit__(None, None, None)
    except Exception as error:
        logger.warning("Blue Guardrails tracer failed while closing span", error=repr(error))


class _BlueGuardrailsSidecarProxy(ProxyTracer):
    """Fan out Haystack trace calls to the user tracer and Blue Guardrails."""

    _blueguardrails_tracer: BlueGuardrailsTracer | None = None
    _blueguardrails_enabled: bool = False

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)

        if name != "actual_tracer":
            return
        if getattr(self, "_blueguardrails_tracer", None) is None:
            return

        super().__setattr__("_blueguardrails_enabled", not isinstance(value, NullTracer))

    @contextlib.contextmanager
    def trace(
        self, operation_name: str, tags: dict[str, Any] | None = None, parent_span: Span | None = None
    ) -> Generator[Span]:
        """Trace through the user tracer and Blue Guardrails when enabled."""
        if self._blueguardrails_tracer is None or not self._blueguardrails_enabled:
            with self.actual_tracer.trace(operation_name, tags=tags, parent_span=parent_span) as span:
                yield span
            return

        has_user_tracer = not isinstance(self.actual_tracer, NullTracer)
        if not has_user_tracer:
            blueguardrails_ctx, blueguardrails_span = _enter_blueguardrails_trace(
                self._blueguardrails_tracer, operation_name, tags, parent_span
            )
            try:
                yield blueguardrails_span
            except Exception:
                _exit_blueguardrails_trace(blueguardrails_ctx, sys.exc_info())
                raise
            else:
                _exit_blueguardrails_trace(blueguardrails_ctx)
            return

        user_parent = parent_span
        blueguardrails_parent = parent_span
        if isinstance(parent_span, CompositeSpan):
            user_parent = parent_span.original
            blueguardrails_parent = parent_span.blueguardrails

        with self.actual_tracer.trace(operation_name, tags=tags, parent_span=user_parent) as user_span:
            blueguardrails_ctx, blueguardrails_span = _enter_blueguardrails_trace(
                self._blueguardrails_tracer, operation_name, tags, blueguardrails_parent
            )
            try:
                yield CompositeSpan(user_span, blueguardrails_span) if blueguardrails_ctx is not None else user_span
            except Exception:
                _exit_blueguardrails_trace(blueguardrails_ctx, sys.exc_info())
                raise
            else:
                _exit_blueguardrails_trace(blueguardrails_ctx)

    def current_span(self) -> Span | None:
        return self.actual_tracer.current_span()


def configure_blueguardrails_tracer(
    blueguardrails_tracer: BlueGuardrailsTracer | None = None,
    *,
    name: str | None = None,
    endpoint: str | None = None,
    api_key: str | Secret | None = None,
    sample_rate: float = 1.0,
    tags: dict[str, str] | None = None,
    replace: bool = False,
) -> BlueGuardrailsTracer:
    """Configure Blue Guardrails on the global Haystack tracing proxy.

    If ``blueguardrails_tracer`` is not provided, this function creates one with the default
    Blue Guardrails OTLP exporter. In that mode, it reads ``BLUE_GUARDRAILS_API_KEY`` unless
    ``api_key`` is provided explicitly.

    Args:
        blueguardrails_tracer: Existing Blue Guardrails tracer to use. If omitted, a tracer
            with the default Blue Guardrails exporter is created.
        name: Trace name shown in Blue Guardrails when creating a default tracer.
        endpoint: Blue Guardrails OTLP trace endpoint when creating a default tracer.
        api_key: API key used to authorize trace export. Defaults to ``BLUE_GUARDRAILS_API_KEY``.
        sample_rate: Fraction of generator calls to trace, from 0.0 to 1.0.
        tags: Conversation tags attached to exported spans.
        replace: Replace an already-installed Blue Guardrails tracer. Defaults
            to ``False`` to preserve existing idempotent connector behavior.

    Returns:
        The configured Blue Guardrails tracer.

    Raises:
        ValueError: If a default tracer is created and the API key is missing.
    """
    from haystack.tracing.tracer import tracer as proxy

    existing_blueguardrails_tracer: BlueGuardrailsTracer | None = getattr(proxy, "_blueguardrails_tracer", None)
    if existing_blueguardrails_tracer is not None and not replace:
        tracer_to_configure = existing_blueguardrails_tracer
    elif blueguardrails_tracer is None:
        from blueguardrails_haystack.tracer import _DEFAULT_ENDPOINT, _DEFAULT_TRACE_NAME, create_blueguardrails_tracer

        tracer_to_configure = create_blueguardrails_tracer(
            name=name or _DEFAULT_TRACE_NAME,
            endpoint=endpoint or _DEFAULT_ENDPOINT,
            api_key=api_key,
            sample_rate=sample_rate,
            tags=tags,
        )
    else:
        tracer_to_configure = blueguardrails_tracer

    _patch_pipeline_component_span_for_models()

    proxy.__class__ = _BlueGuardrailsSidecarProxy
    if not isinstance(proxy, _BlueGuardrailsSidecarProxy):
        raise TypeError("Haystack tracing proxy could not be upgraded for Blue Guardrails")

    proxy._blueguardrails_tracer = tracer_to_configure
    proxy._blueguardrails_enabled = True
    return tracer_to_configure
