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


def test_codex_tool_server_script_completes_an_mcp_handshake(tmp_path):
    """The generated stdio server must actually start under the installed mcp.

    It runs in a subprocess, so an import that no longer resolves (mcp 2.x
    renamed FastMCP to MCPServer) surfaces only as a handshake failure inside a
    live Codex run. Driving the real protocol here catches it offline.
    """
    import json
    import subprocess
    import sys

    from agent_sdk_wrapper.providers.openai_provider import _tool_entry, _tool_server_script

    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    script = tmp_path / "server.py"
    script.write_text(_tool_server_script(), encoding="utf-8")
    (tmp_path / "tools.json").write_text(
        json.dumps([_tool_entry(add)], ensure_ascii=False), encoding="utf-8"
    )

    requests = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    ]
    proc = subprocess.run(
        [sys.executable, str(script)],
        input="\n".join(json.dumps(r) for r in requests) + "\n",
        capture_output=True,
        text=True,
        timeout=60,
        cwd=tmp_path,
    )

    responses = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    assert responses, f"server produced no output; stderr:\n{proc.stderr}"
    listed = next(r for r in responses if r.get("id") == 2)
    assert [tool["name"] for tool in listed["result"]["tools"]] == ["add"]
