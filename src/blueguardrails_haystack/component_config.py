# SPDX-FileCopyrightText: 2025-present Blue Guardrails
#
# SPDX-License-Identifier: Apache-2.0

"""Extract generator initialization config for Blue Guardrails spans."""

from typing import Any
from urllib.parse import urlparse

from blueguardrails_haystack._utils import is_mapping, to_int
from blueguardrails_haystack.request_options import component_request_options

_DEFAULT_PORTS = {"http": 80, "https": 443}
_SERVER_URL_ATTRS = (
    "api_base_url",
    "base_url",
    "_base_url",
    "azure_endpoint",
    "_azure_endpoint",
    "endpoint_url",
    "api_endpoint",
    "host",
)
_SERVER_NESTED_ATTRS = (
    "client",
    "async_client",
    "_client",
    "_api_client",
    "api_client",
    "meta",
    "_endpoint",
    "_http_options",
    "http_options",
    "http_client_kwargs",
    "boto3_config",
)


def _get_value(value: Any, key: str) -> Any:
    if is_mapping(value):
        return value.get(key)
    try:
        return getattr(value, key, None)
    except Exception:
        return None


def _server_attrs_from_url(value: Any) -> dict[str, Any]:
    """Extract server host and port from a URL-like value."""
    if value is None:
        return {}

    host = getattr(value, "host", None) or getattr(value, "hostname", None)
    scheme = getattr(value, "scheme", None)
    port = getattr(value, "port", None)
    if callable(host):
        host = None
    if callable(scheme):
        scheme = None
    if callable(port):
        port = None
    if host:
        result: dict[str, Any] = {"address": str(host)}
        port_int = to_int(port)
        if port_int is None and isinstance(scheme, str):
            port_int = _DEFAULT_PORTS.get(scheme.lower())
        if port_int is not None:
            result["port"] = port_int
        return result

    if not isinstance(value, str):
        return {}

    text = value.strip()
    if not text:
        return {}

    # urlparse treats "example.com:443" as a path unless we add a scheme-relative prefix.
    parse_text = text if "://" in text else f"//{text}"
    parsed = urlparse(parse_text)
    if not parsed.hostname:
        return {}

    result = {"address": parsed.hostname}
    try:
        port_int = parsed.port
    except ValueError:
        port_int = None
    if port_int is None and parsed.scheme:
        port_int = _DEFAULT_PORTS.get(parsed.scheme.lower())
    if port_int is not None:
        result["port"] = port_int
    return result


def _extract_server_from_object(value: Any, max_depth: int = 4, seen: set[int] | None = None) -> dict[str, Any]:
    """Extract server host and port from generator or client state.

    Only known endpoint-bearing and nested client attributes are inspected; the
    traversal is bounded and cycle-safe.
    """
    attrs = _server_attrs_from_url(value)
    if attrs:
        return attrs
    if value is None or isinstance(value, (str, bytes, int, float, bool)) or max_depth <= 0:
        return {}

    seen = seen or set()
    value_id = id(value)
    if value_id in seen:
        return {}
    seen.add(value_id)

    for key in _SERVER_URL_ATTRS:
        attrs = _server_attrs_from_url(_get_value(value, key))
        if attrs:
            return attrs

    for key in _SERVER_NESTED_ATTRS:
        attrs = _extract_server_from_object(_get_value(value, key), max_depth=max_depth - 1, seen=seen)
        if attrs:
            return attrs

    return {}


def _extract_component_model(instance: Any) -> str | None:
    """Extract the configured model from a generator instance."""
    for attr in ("model", "model_name", "_model", "_model_name", "azure_deployment", "_azure_deployment"):
        value = getattr(instance, attr, None)
        if isinstance(value, str) and value:
            return value
        nested_model = getattr(value, "model_name", None)
        if isinstance(nested_model, str) and nested_model:
            return nested_model
    return None


def _extract_component_server(instance: Any) -> dict[str, Any]:
    """Extract endpoint host and port from a generator instance."""
    return _extract_server_from_object(instance)


def _extract_component_tools(instance: Any) -> Any:
    """Extract initialization-time tools from a generator instance."""
    for attr in ("tools", "_tools"):
        if hasattr(instance, attr):
            tools = getattr(instance, attr)
            if tools is not None:
                return tools
    return None


def extract_component_config(instance: Any) -> dict[str, Any]:
    """Extract generator configuration needed for Blue Guardrails spans."""
    config: dict[str, Any] = {}

    if model := _extract_component_model(instance):
        config["model"] = model

    if server := _extract_component_server(instance):
        config["server"] = server

    if request_options := component_request_options(instance):
        config["request_options"] = request_options

    tools = _extract_component_tools(instance)
    if tools is not None:
        config["tools"] = tools

    return config
