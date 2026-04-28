# SPDX-FileCopyrightText: 2025-present Blue Guardrails
#
# SPDX-License-Identifier: Apache-2.0

"""Convert Haystack/provider tool definitions to OTel GenAI JSON."""

import json
from collections.abc import Iterable
from typing import Any


def _iter_tool_items(tools: Any) -> list[Any]:
    """Flatten tool containers into individual tool-like objects."""
    if tools is None:
        return []
    if isinstance(tools, dict):
        return [tools]

    toolset_tools = getattr(tools, "tools", None)
    if toolset_tools is not None and not callable(toolset_tools):
        return _iter_tool_items(toolset_tools)

    if isinstance(tools, Iterable) and not isinstance(tools, (str, bytes)):
        result: list[Any] = []
        for item in tools:
            result.extend(_iter_tool_items(item))
        return result

    return [tools]


def _tool_parameters_from_dict(data: dict[str, Any]) -> Any:
    parameters = data.get("parameters") or data.get("input_schema") or data.get("inputSchema")
    if isinstance(parameters, dict) and "json" in parameters:
        return parameters["json"]
    return parameters


def _tool_definition(
    name: Any,
    *,
    tool_type: str = "function",
    description: Any = None,
    parameters: Any = None,
) -> dict[str, Any] | None:
    if not name:
        return None

    result: dict[str, Any] = {"type": str(tool_type), "name": str(name)}
    if description is not None:
        result["description"] = description
    if parameters is not None:
        result["parameters"] = parameters
    return result


def _tool_dict_to_semconv(tool: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a tool dictionary to a GenAI tool definition."""
    function = tool.get("function")
    if isinstance(function, dict):
        return _tool_definition(
            function.get("name"),
            description=function.get("description"),
            parameters=function.get("parameters"),
        )

    spec = tool.get("toolSpec")
    if isinstance(spec, dict):
        return _tool_definition(
            spec.get("name"),
            description=spec.get("description"),
            parameters=_tool_parameters_from_dict(spec),
        )

    return _tool_definition(
        tool.get("name"),
        tool_type=tool.get("type") or "function",
        description=tool.get("description"),
        parameters=_tool_parameters_from_dict(tool),
    )


def _tool_to_semconv(tool: Any) -> dict[str, Any] | None:
    """Convert a tool-like object to a GenAI tool definition."""
    if isinstance(tool, dict):
        return _tool_dict_to_semconv(tool)

    spec = getattr(tool, "tool_spec", None)
    if isinstance(spec, dict):
        return _tool_dict_to_semconv(spec)

    return _tool_definition(
        getattr(tool, "name", None),
        description=str(description) if (description := getattr(tool, "description", None)) is not None else None,
        parameters=getattr(tool, "parameters", None),
    )


def convert_tool_definitions(tools: Any) -> str:
    """Convert tool definitions to GenAI JSON."""
    definitions = [converted for tool in _iter_tool_items(tools) if (converted := _tool_to_semconv(tool))]
    return json.dumps(definitions, default=str)
