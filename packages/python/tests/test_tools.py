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
    import threading

    from agent_sdk_wrapper.tools import _make_anthropic_handler

    loop_ran = threading.Event()

    def blocks_until_the_loop_runs() -> str:
        # On the event loop thread this would deadlock: the loop could never set the event.
        return "done" if loop_ran.wait(timeout=5) else "loop blocked"

    async def mark_loop_running():
        await asyncio.sleep(0)
        loop_ran.set()

    marker = asyncio.create_task(mark_loop_running())
    out = await _make_anthropic_handler(blocks_until_the_loop_runs)({})
    await marker
    assert out["content"][0]["text"] == "done"


class _ToolServer:
    """Run the generated Codex tool server as a subprocess and talk JSON-RPC to it."""

    def __init__(self, directory, tools):
        import json

        from agent_sdk_wrapper.providers.openai_provider import _tool_manifest, _tool_server_script

        directory.mkdir()
        self.script = directory / "server.py"
        self.script.write_text(_tool_server_script(), encoding="utf-8")
        (directory / "tools.json").write_text(
            json.dumps(_tool_manifest(tools), ensure_ascii=False), encoding="utf-8"
        )
        self.stderr = ""
        self._next_id = 0

    async def __aenter__(self):
        import asyncio
        import sys

        self.proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(self.script),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.script.parent,
        )
        self._stderr_task = asyncio.create_task(self.proc.stderr.read())
        return self

    async def __aexit__(self, *exc_info):
        import asyncio

        self.proc.stdin.close()
        try:
            await asyncio.wait_for(self.proc.wait(), timeout=5)
        except TimeoutError:
            self.proc.kill()
            await self.proc.wait()
        self.stderr = (await self._stderr_task).decode(errors="replace")

    async def send(self, message):
        import json

        self.proc.stdin.write((json.dumps(message) + "\n").encode())
        await self.proc.stdin.drain()

    async def request(self, method, params):
        """Return the response to one request; keep stdin open for the next."""
        import asyncio
        import json

        self._next_id += 1
        request_id = self._next_id
        await self.send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        async with asyncio.timeout(60):
            while line := await self.proc.stdout.readline():
                response = json.loads(line)
                if response.get("id") == request_id:
                    return response
        raise AssertionError(f"server closed before responding to {method}")

    async def initialize(self):
        response = await self.request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            },
        )
        if "error" not in response:
            await self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return response

    async def call(self, name, arguments):
        response = await self.request("tools/call", {"name": name, "arguments": arguments})
        return response["result"]


async def test_codex_tool_server_script_completes_an_mcp_handshake(tmp_path, monkeypatch):
    """Check the generated MCP server with a real offline handshake and tool calls."""
    import importlib

    # Importable only through the parent's sys.path; uses a module global.
    modules = tmp_path / "modules"
    modules.mkdir()
    (modules / "handshake_tools.py").write_text(
        "OFFSET = 10\n\n\n"
        "def shift(value: int) -> int:\n"
        '    """Shift a value."""\n'
        "    return value + OFFSET\n\n\n"
        "class Thing:\n"
        "    pass\n\n\n"
        "def search(query: str, _page: int = 1, thing: Thing = None, **filters: str) -> str:\n"
        '    """Search."""\n'
        "    return f'{query} {_page} {thing} {filters}'\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(modules))
    handshake_tools = importlib.import_module("handshake_tools")

    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    # In the server, __main__ is the server script, whose globals include `manifest`.
    def manifest(x: int) -> int:
        """Double a value."""
        if x < 0:
            raise ValueError(f"negative: {x}")
        return x * 2

    manifest.__module__ = "__main__"
    tools = [add, handshake_tools.shift, handshake_tools.search, manifest]

    async with _ToolServer(tmp_path / "server", tools) as server:
        initialized = await server.initialize()
        assert "tools" in initialized["result"]["capabilities"]
        listed = (await server.request("tools/list", {}))["result"]["tools"]
        added = await server.call("add", {"a": 2, "b": 3})
        shifted = await server.call("shift", {"value": 1})
        searched = await server.call("search", {"query": "q", "_page": "2", "lang": "en"})
        doubled = await server.call("manifest", {"x": 4})
        failed = await server.call("manifest", {"x": -1})
        invalid = await server.call("add", {"a": "two", "b": 3})

    assert server.proc.returncode == 0, server.stderr
    # The Claude handler advertises the same schemas.
    assert [(t["name"], t["inputSchema"]) for t in listed] == [
        (tool.__name__, json_schema_for(tool)) for tool in tools
    ]
    assert (added.get("isError"), added["content"][0]["text"]) == (False, "5")
    assert (shifted.get("isError"), shifted["content"][0]["text"]) == (False, "11")
    assert searched["content"][0]["text"] == "q 2 None {'lang': 'en'}"
    assert (doubled.get("isError"), doubled["content"][0]["text"]) == (False, "8")
    assert (failed["isError"], failed["content"][0]["text"]) == (True, "Error: negative: -1")
    assert invalid["isError"] is True
    assert invalid["content"][0]["text"].startswith("Error: invalid arguments:")


async def test_codex_tool_server_reports_a_failed_import_at_initialize(tmp_path, monkeypatch):
    import sys
    import types

    # Importable in the parent only: the server has no such module to fall back on.
    ghost_tools = types.ModuleType("ghost_tools")
    exec("def ghost() -> str:\n    return 'boo'\n", ghost_tools.__dict__)
    monkeypatch.setitem(sys.modules, "ghost_tools", ghost_tools)

    async with _ToolServer(tmp_path / "server", [ghost_tools.ghost]) as server:
        initialized = await server.initialize()

    assert initialized["error"]["message"] == (
        "cannot load tool 'ghost': ModuleNotFoundError: No module named 'ghost_tools'"
    )
    assert server.proc.returncode != 0


def test_unsupported_parameter_types_fall_back_to_an_open_schema():
    import sqlite3

    def fn(conn: sqlite3.Connection, n: int) -> str:
        return f"{type(conn).__name__} {n}"

    schema = json_schema_for(fn)
    assert schema["properties"]["conn"] == {}
    assert schema["properties"]["n"] == {"type": "integer"}
    assert _call(fn, {"conn": "db", "n": "2"})["content"][0]["text"] == "str 2"


def test_field_defaults_and_titled_default_values_survive():
    from pydantic import Field

    def fn(limit: int = Field(5, description="Max rows"), meta: dict = {"title": "x"}) -> str:  # noqa: B006
        return f"{limit} {meta}"

    schema = json_schema_for(fn)
    assert schema["properties"]["limit"] == {
        "default": 5,
        "description": "Max rows",
        "type": "integer",
    }
    assert schema["properties"]["meta"]["default"] == {"title": "x"}
    assert _call(fn, {})["content"][0]["text"] == "5 {'title': 'x'}"


def test_keyword_catch_all_tools_receive_unnamed_arguments():
    def search(query: str, **filters: str) -> str:
        return f"{query} {filters}"

    assert json_schema_for(search)["additionalProperties"] is True
    out = _call(search, {"query": "q", "lang": "en"})
    assert out["content"][0]["text"] == "q {'lang': 'en'}"


def test_positional_only_parameters_are_rejected():
    def scale(value: int, /, factor: int = 2) -> int:
        return value * factor

    with pytest.raises(ConfigError, match="positional-only parameter 'value'"):
        json_schema_for(scale)

