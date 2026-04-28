# SPDX-FileCopyrightText: 2025-present Blue Guardrails
#
# SPDX-License-Identifier: Apache-2.0

"""Convert Haystack/provider tool definitions to OTel GenAI JSON."""

import json
from typing import Any

from blueguardrails_haystack._utils import AnyMapping, is_mapping, is_non_string_iterable


def _iter_tool_items(tools: Any) -> list[Any]:
    """Flatten tool containers into individual tool-like objects."""
    if tools is None:
        return []
    if is_mapping(tools):
        return [tools]

    toolset_tools = getattr(tools, "tools", None)
    if toolset_tools is not None and not callable(toolset_tools):
        return _iter_tool_items(toolset_tools)

    if is_non_string_iterable(tools):
        result: list[Any] = []
        for item in tools:
            result.extend(_iter_tool_items(item))
        return result

    return [tools]


def _tool_parameters_from_dict(data: AnyMapping) -> Any:
    parameters = data.get("parameters") or data.get("input_schema") or data.get("inputSchema")
    if is_mapping(parameters) and "json" in parameters:
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


def _tool_dict_to_semconv(tool: AnyMapping) -> dict[str, Any] | None:
    """Convert a tool dictionary to a GenAI tool definition."""
    function = tool.get("function")
    if is_mapping(function):
        return _tool_definition(
            function.get("name"),
            description=function.get("description"),
            parameters=function.get("parameters"),
        )

    spec = tool.get("toolSpec")
    if is_mapping(spec):
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
    if is_mapping(tool):
        return _tool_dict_to_semconv(tool)

    spec = getattr(tool, "tool_spec", None)
    if is_mapping(spec):
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
