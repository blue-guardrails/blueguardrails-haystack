# SPDX-FileCopyrightText: 2025-present Blue Guardrails
#
# SPDX-License-Identifier: Apache-2.0

"""Small shared normalization helpers."""

from collections.abc import Iterator
from typing import Any


def first_present(mapping: dict[str, Any], *keys: str) -> Any:
    """Return the first non-``None`` value for the given keys."""
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def nested_first_present(mapping: dict[str, Any], *paths: tuple[str, ...]) -> Any:
    """Return the first non-``None`` value found at a nested dictionary path."""
    for path in paths:
        current: Any = mapping
        for key in path:
            if not isinstance(current, dict) or key not in current or current[key] is None:
                break
            current = current[key]
        else:
            return current
    return None


def to_int(value: Any) -> int | None:
    """Convert a provider usage value to an integer when possible."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def snake_to_lower_camel(key: str) -> str:
    """Convert a snake_case key to lowerCamelCase."""
    head, *tail = key.split("_")
    return head + "".join(part.capitalize() for part in tail)


def mapping_numeric_items(value: Any) -> Iterator[tuple[str, Any]]:
    """Yield key/value pairs from a mapping whose values can be converted to integers."""
    if not isinstance(value, dict):
        return
    for key, item_value in value.items():
        if to_int(item_value) is not None:
            yield str(key), item_value
