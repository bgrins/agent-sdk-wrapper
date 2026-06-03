"""Tests for tool-callable introspection."""

from __future__ import annotations

from agent_sdk_wrapper.tools import (
    TOOL_DESCRIPTION_ATTR,
    TOOL_NAME_ATTR,
    json_schema_for,
    tool_description,
    tool_name,
)


def add(a: int, b: int = 2) -> int:
    """Add two numbers.

    Longer description that should be ignored.
    """
    return a + b


def test_tool_metadata():
    assert tool_name(add) == "add"
    assert tool_description(add) == "Add two numbers."


def test_tool_metadata_overrides():
    def fn() -> str:
        return "ok"

    setattr(fn, TOOL_NAME_ATTR, "custom_name")
    setattr(fn, TOOL_DESCRIPTION_ATTR, "Custom description.")

    assert tool_name(fn) == "custom_name"
    assert tool_description(fn) == "Custom description."


def test_json_schema_basic_types():
    def fn(s: str, i: int, f: float, b: bool) -> str:
        return s

    schema = json_schema_for(fn)
    assert schema["type"] == "object"
    assert schema["properties"] == {
        "s": {"type": "string"},
        "i": {"type": "integer"},
        "f": {"type": "number"},
        "b": {"type": "boolean"},
    }
    assert set(schema["required"]) == {"s", "i", "f", "b"}


def test_json_schema_default_makes_optional():
    schema = json_schema_for(add)
    assert schema["properties"]["a"] == {"type": "integer"}
    assert schema["properties"]["b"] == {"type": "integer"}
    assert schema["required"] == ["a"]


def test_json_schema_optional_unwrap():
    def fn(x: int | None) -> int:
        return x or 0

    schema = json_schema_for(fn)
    assert schema["properties"]["x"] == {"type": "integer"}
