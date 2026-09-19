"""Tests for tool-callable introspection."""

from __future__ import annotations

import enum
from typing import Literal

import pytest
from pydantic import BaseModel

from agent_sdk_wrapper import ConfigError
from agent_sdk_wrapper.tools import (
    TOOL_DESCRIPTION_ATTR,
    TOOL_NAME_ATTR,
    json_schema_for,
    tool_description,
    tool_name,
    validate_tool_names,
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


def test_tool_names_must_be_valid_and_unique():
    def first() -> None: ...

    def other() -> None: ...

    setattr(other, TOOL_NAME_ATTR, "first")
    with pytest.raises(ConfigError, match="duplicate tool name 'first'"):
        validate_tool_names([first, other])
    with pytest.raises(ConfigError, match="<lambda>"):
        validate_tool_names([lambda: None])


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
    assert schema["properties"]["b"] == {"type": "integer", "default": 2}
    assert schema["required"] == ["a"]


def _call(fn, args):
    import asyncio

    from agent_sdk_wrapper.tools import _make_anthropic_handler

    return asyncio.run(_make_anthropic_handler(fn)(args))


def test_optional_parameter_without_default_accepts_null():
    def fn(x: int | None) -> str:
        return repr(x)

    schema = json_schema_for(fn)
    assert schema["properties"]["x"] == {"anyOf": [{"type": "integer"}, {"type": "null"}]}
    assert schema["required"] == ["x"]
    assert _call(fn, {"x": None})["content"][0]["text"] == "None"


class Color(enum.Enum):
    RED = "red"


class Point(BaseModel):
    x: int


def test_literal_enum_nested_model_and_list_items_are_validated_and_coerced():
    def fn(level: Literal[1, 2, 3], color: Color, point: Point, ids: list[int]) -> str:
        return f"{level!r} {color!r} {point!r} {ids!r}"

    schema = json_schema_for(fn)
    assert schema["properties"]["level"]["enum"] == [1, 2, 3]
    assert schema["properties"]["ids"] == {"type": "array", "items": {"type": "integer"}}
    out = _call(fn, {"level": 2, "color": "red", "point": {"x": "3"}, "ids": ["1", 2]})
    assert out["content"][0]["text"] == "2 <Color.RED: 'red'> Point(x=3) [1, 2]"


def test_unresolvable_hint_only_loses_its_own_parameter():
    def fn(n: int, other: MissingType) -> str:  # noqa: F821
        return repr(n)

    schema = json_schema_for(fn)
    assert schema["properties"]["n"] == {"type": "integer"}
    assert schema["properties"]["other"] == {}
    assert _call(fn, {"n": "5", "other": 1})["content"][0]["text"] == "5"


def test_invalid_arguments_return_a_tool_error_without_calling():
    calls = []

    def fn(n: int) -> int:
        calls.append(n)
        return n

    out = _call(fn, {"n": "not a number"})
    assert out["is_error"] is True
    assert "invalid arguments" in out["content"][0]["text"]
    assert calls == []


async def test_sync_tool_runs_off_the_event_loop():
    import asyncio
    import time

    from agent_sdk_wrapper.tools import _make_anthropic_handler

    def slow() -> str:
        time.sleep(0.3)
        return "done"

    ticks = 0

    async def tick():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    ticker = asyncio.create_task(tick())
    out = await _make_anthropic_handler(slow)({})
    ticker.cancel()
    assert out["content"][0]["text"] == "done"
    assert ticks >= 10


async def test_codex_tool_server_script_completes_an_mcp_handshake(tmp_path, monkeypatch):
    """Check the generated MCP server with a real offline handshake and tool calls."""
    import asyncio
    import importlib
    import json
    import sys

    from agent_sdk_wrapper.providers.openai_provider import _tool_manifest, _tool_server_script

    # Importable only through the parent's sys.path; uses a module global.
    modules = tmp_path / "modules"
    modules.mkdir()
    (modules / "handshake_tools.py").write_text(
        "OFFSET = 10\n\n\n"
        "def shift(value: int) -> int:\n"
        '    """Shift a value."""\n'
        "    return value + OFFSET\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(modules))
    shift = importlib.import_module("handshake_tools").shift

    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    server_dir = tmp_path / "server"
    server_dir.mkdir()
    script = server_dir / "server.py"
    script.write_text(_tool_server_script(), encoding="utf-8")
    (server_dir / "tools.json").write_text(
        json.dumps(_tool_manifest([add, shift]), ensure_ascii=False), encoding="utf-8"
    )

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        str(script),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=server_dir,
    )
    assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
    stderr_task = asyncio.create_task(proc.stderr.read())

    async def send(request):
        proc.stdin.write((json.dumps(request) + "\n").encode())
        await proc.stdin.drain()

    async def receive(request_id):
        while line := await proc.stdout.readline():
            response = json.loads(line)
            if response.get("id") == request_id:
                assert "error" not in response, response
                return response
        stderr = (await stderr_task).decode(errors="replace")
        raise AssertionError(f"server closed before response {request_id}; stderr:\n{stderr}")

    async def call(request_id, name, arguments):
        await send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }
        )
        return (await receive(request_id))["result"]

    try:
        async with asyncio.timeout(60):
            # Wait for initialization; keep stdin open until the last response.
            await send(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "test", "version": "1"},
                    },
                }
            )
            initialized = await receive(1)
            assert "tools" in initialized["result"]["capabilities"]
            await send({"jsonrpc": "2.0", "method": "notifications/initialized"})
            await send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
            listed = await receive(2)
            added = await call(3, "add", {"a": 2, "b": 3})
            shifted = await call(4, "shift", {"value": 1})
    finally:
        proc.stdin.close()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except TimeoutError:
            proc.kill()
            await proc.wait()
        stderr = (await stderr_task).decode(errors="replace")

    assert proc.returncode == 0, stderr
    assert [tool["name"] for tool in listed["result"]["tools"]] == ["add", "shift"]
    assert (added.get("isError"), added["content"][0]["text"]) == (False, "5")
    assert (shifted.get("isError"), shifted["content"][0]["text"]) == (False, "11")
