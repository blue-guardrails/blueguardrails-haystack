# SPDX-FileCopyrightText: 2025-present BlueGuardrails
#
# SPDX-License-Identifier: Apache-2.0

"""Haystack component that enables BlueGuardrails LLM tracing."""

from typing import Any

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.utils import Secret, deserialize_secrets_inplace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

from blueguardrails_haystack.proxy import configure_blueguardrails_tracer
from blueguardrails_haystack.tracer import BlueGuardrailsTracer

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "https://api.blueguardrails.com/v1/traces"


@component
class BlueGuardrailsConnector:
    """Send Haystack generator traces to BlueGuardrails.

    Add this component to a pipeline without connecting it. It installs a sidecar
    tracer that exports GenAI spans while leaving the user's tracer in place.

    Examples:
        Add the connector when constructing a pipeline::

            from haystack import Pipeline
            from haystack.components.generators.chat import OpenAIChatGenerator
            from haystack.utils import Secret
            from blueguardrails_haystack import BlueGuardrailsConnector

            pipe = Pipeline()
            pipe.add_component(
                "blueguardrails",
                BlueGuardrailsConnector(
                    name="my-pipeline",
                    api_key=Secret.from_env_var("BLUEGUARDRAILS_API_KEY"),
                ),
            )
            pipe.add_component("llm", OpenAIChatGenerator())
    """

    def __init__(
        self,
        name: str,
        *,
        endpoint: str = _DEFAULT_ENDPOINT,
        api_key: Secret | None = Secret.from_env_var("BLUEGUARDRAILS_API_KEY"),  # noqa: B008
        sample_rate: float = 1.0,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Initialize the connector.

        Args:
            name: Pipeline trace name shown in BlueGuardrails.
            endpoint: BlueGuardrails OTLP trace endpoint.
            api_key: API key used to authorize trace export.
            sample_rate: Fraction of generator calls to trace, from 0.0 to 1.0.
            tags: Conversation tags attached to exported spans.

        Raises:
            ValueError: If ``sample_rate`` is invalid or ``api_key`` cannot be resolved.
        """
        self.name = name
        self.endpoint = endpoint
        self.api_key = api_key
        self.sample_rate = sample_rate
        self.tags = tags

        headers: dict[str, str] = {}
        if api_key:
            resolved_key = api_key.resolve_value()
            if resolved_key:
                headers["Authorization"] = f"Bearer {resolved_key}"

        # Use an isolated provider so existing OTel configuration is not affected.
        resource = Resource.create(
            {
                "service.name": "blueguardrails-haystack",
                "haystack.pipeline.name": name,
            }
        )
        provider = TracerProvider(
            sampler=TraceIdRatioBased(sample_rate),
            resource=resource,
        )
        exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers)
        provider.add_span_processor(BatchSpanProcessor(exporter))

        configure_blueguardrails_tracer(BlueGuardrailsTracer(provider, conversation_tags=tags))

        self._provider = provider

    @component.output_types(name=str)
    def run(self) -> dict[str, str]:
        """Return the configured pipeline trace name.

        Returns:
            Mapping containing the connector name.
        """
        return {"name": self.name}

    def to_dict(self) -> dict[str, Any]:
        """Serialize the connector configuration.

        Returns:
            Haystack-compatible serialized component data.
        """
        return default_to_dict(
            self,
            name=self.name,
            endpoint=self.endpoint,
            api_key=self.api_key.to_dict() if self.api_key else None,
            sample_rate=self.sample_rate,
            tags=self.tags,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BlueGuardrailsConnector":
        """Deserialize the connector configuration.

        Args:
            data: Haystack-compatible serialized component data.

        Returns:
            Deserialized connector instance.

        Raises:
            KeyError: If required serialized fields are missing.
        """
        init_params = data["init_parameters"]
        deserialize_secrets_inplace(init_params, keys=["api_key"])
        return default_from_dict(cls, data)
