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


async def test_codex_tool_server_script_completes_an_mcp_handshake(tmp_path):
    """The generated stdio server must actually start under the installed mcp.

    It runs in a subprocess, so an import that no longer resolves (mcp 2.x
    renamed FastMCP to MCPServer) surfaces only as a handshake failure inside a
    live Codex run. Driving the real protocol here catches it offline.
    """
    import asyncio
    import json
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
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        str(script),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=tmp_path,
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

    try:
        async with asyncio.timeout(60):
            # Await initialization before announcing readiness, and leave stdin
            # open until tools/list completes. Sending everything then EOF races
            # the MCP server's shutdown against its response tasks.
            await send(requests[0])
            initialized = await receive(1)
            assert "tools" in initialized["result"]["capabilities"]
            await send(requests[1])
            await send(requests[2])
            listed = await receive(2)
    finally:
        proc.stdin.close()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except TimeoutError:
            proc.kill()
            await proc.wait()
        stderr = (await stderr_task).decode(errors="replace")

    assert proc.returncode == 0, stderr
    assert [tool["name"] for tool in listed["result"]["tools"]] == ["add"]
