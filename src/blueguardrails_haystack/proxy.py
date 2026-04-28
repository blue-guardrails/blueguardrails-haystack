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
from blueguardrails_haystack.span import BGSpan, CompositeSpan

if TYPE_CHECKING:
    from blueguardrails_haystack.tracer import BGTracer

logger = logging.getLogger(__name__)

_pipeline_span_patched = False
_original_create_component_span: Any | None = None


def _set_bg_component_config(span: Span, config: dict[str, Any]) -> None:
    """Attach component init config only to the Blue Guardrails span."""
    if not config:
        return
    try:
        if isinstance(span, CompositeSpan) and isinstance(span.bg, BGSpan):
            span.bg.set_component_config(config)
        elif isinstance(span, BGSpan):
            span.set_component_config(config)
    except Exception as error:
        logger.warning("Blue Guardrails tracer skipped component config", error=repr(error))


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
    def _create_component_span_with_bg_model(
        component_name: str, instance: Any, inputs: dict[str, Any], parent_span: Span | None = None
    ) -> Generator[Span]:
        with original_create_component_span(component_name, instance, inputs, parent_span) as span:
            _set_bg_component_config(span, extract_component_config(instance))
            yield span

    PipelineBase._create_component_span = _create_component_span_with_bg_model
    _pipeline_span_patched = True


def _enter_bg_trace(
    bg_tracer: BGTracer, operation_name: str, tags: dict[str, Any] | None, parent_span: Span | None = None
) -> tuple[Any, Span]:
    """Start a Blue Guardrails trace without propagating failures."""
    try:
        ctx = bg_tracer.trace(operation_name, tags=tags, parent_span=parent_span)
        span = ctx.__enter__()
        return ctx, span
    except Exception as error:
        logger.warning("Blue Guardrails tracer failed to start span", operation_name=operation_name, error=repr(error))
        return None, NullSpan()


def _exit_bg_trace(bg_ctx: Any | None, exc_info: tuple[Any, ...] | None = None) -> None:
    """End a Blue Guardrails trace without propagating failures."""
    if bg_ctx is None:
        return
    try:
        bg_ctx.__exit__(*exc_info) if exc_info else bg_ctx.__exit__(None, None, None)
    except Exception as error:
        logger.warning("Blue Guardrails tracer failed while closing span", error=repr(error))


class _BGSidecarProxy(ProxyTracer):
    """Fan out Haystack trace calls to the user tracer and Blue Guardrails."""

    _bg_tracer: BGTracer | None = None
    _bg_enabled: bool = False

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)

        if name != "actual_tracer":
            return
        if getattr(self, "_bg_tracer", None) is None:
            return

        super().__setattr__("_bg_enabled", not isinstance(value, NullTracer))

    @contextlib.contextmanager
    def trace(
        self, operation_name: str, tags: dict[str, Any] | None = None, parent_span: Span | None = None
    ) -> Generator[Span]:
        """Trace through the user tracer and Blue Guardrails when enabled."""
        if self._bg_tracer is None or not self._bg_enabled:
            with self.actual_tracer.trace(operation_name, tags=tags, parent_span=parent_span) as span:
                yield span
            return

        has_user_tracer = not isinstance(self.actual_tracer, NullTracer)
        if not has_user_tracer:
            bg_ctx, bg_span = _enter_bg_trace(self._bg_tracer, operation_name, tags, parent_span)
            try:
                yield bg_span
            except Exception:
                _exit_bg_trace(bg_ctx, sys.exc_info())
                raise
            else:
                _exit_bg_trace(bg_ctx)
            return

        user_parent = parent_span
        bg_parent = parent_span
        if isinstance(parent_span, CompositeSpan):
            user_parent = parent_span.original
            bg_parent = parent_span.bg

        with self.actual_tracer.trace(operation_name, tags=tags, parent_span=user_parent) as user_span:
            bg_ctx, bg_span = _enter_bg_trace(self._bg_tracer, operation_name, tags, bg_parent)
            try:
                yield CompositeSpan(user_span, bg_span) if bg_ctx is not None else user_span
            except Exception:
                _exit_bg_trace(bg_ctx, sys.exc_info())
                raise
            else:
                _exit_bg_trace(bg_ctx)

    def current_span(self) -> Span | None:
        return self.actual_tracer.current_span()


def install_bg_tracer(bg_tracer: BGTracer, *, replace: bool = False) -> None:
    """Install Blue Guardrails on the global Haystack tracing proxy.

    Args:
        bg_tracer: Blue Guardrails tracer to install.
        replace: Replace an already-installed Blue Guardrails tracer. Defaults
            to ``False`` to preserve existing idempotent connector behavior.
    """
    from haystack.tracing.tracer import tracer as proxy

    _patch_pipeline_component_span_for_models()

    proxy.__class__ = _BGSidecarProxy
    if not isinstance(proxy, _BGSidecarProxy):
        raise TypeError("Haystack tracing proxy could not be upgraded for Blue Guardrails")

    if replace or proxy._bg_tracer is None:
        proxy._bg_tracer = bg_tracer
    proxy._bg_enabled = True
