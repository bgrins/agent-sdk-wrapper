"""Drive the real Codex runtime against a local mock Responses API."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import signal
import sys
from pathlib import Path
from typing import Any

import pytest
from conformance.mocks import MockCodex
from conformance.runner import seed_chatgpt_login
from pydantic import BaseModel, Field

from agent_sdk_wrapper import Agent, McpStdioServer, RunResult, SubagentDef, TokenUsage

pytest.importorskip("codex_cli_bin")

MODEL = "gpt-5.4"


@pytest.fixture
def mock_api():
    api = MockCodex().start()
    yield api
    api.stop()


@pytest.fixture
def codex_home(tmp_path: Path) -> Path:
    home = tmp_path / "codex-home"
    home.mkdir()
    return home


def codex_config(api: MockCodex, home: Path, *overrides: str) -> dict[str, Any]:
    """Point Codex at the mock and block every other network destination."""

    dead_proxy = "http://127.0.0.1:9"
    return {
        "config_overrides": (
            'model_provider="mock"',
            'model_providers.mock.name="mock"',
            f'model_providers.mock.base_url="{api.base_url}"',
            'model_providers.mock.wire_api="responses"',
            "model_providers.mock.requires_openai_auth=true",
            "model_providers.mock.request_max_retries=0",
            "model_providers.mock.stream_max_retries=0",
            "model_providers.mock.supports_websockets=false",
            *overrides,
        ),
        "env": {
            "CODEX_HOME": str(home),
            "HOME": str(home),
            "HTTPS_PROXY": dead_proxy,
            "HTTP_PROXY": dead_proxy,
            "ALL_PROXY": dead_proxy,
            # git (run by Codex for plugin checks) prefers the lowercase names.
            "https_proxy": dead_proxy,
            "http_proxy": dead_proxy,
            "all_proxy": dead_proxy,
            "CODEX_ACCESS_TOKEN": "",
            "NO_PROXY": "127.0.0.1,localhost",
            "OPENAI_API_KEY": "",
            "CODEX_API_KEY": "",
        },
    }


def codex_agent(
    api: MockCodex,
    home: Path,
    cwd: Path,
    *overrides: str,
    provider_options: dict[str, Any] | None = None,
    **agent_options: Any,
) -> Agent:
    return Agent(
        provider="codex",
        model=MODEL,
        cwd=cwd,
        timeout=60,
        provider_options={
            "api_key": "sk-mock-key",
            "config": codex_config(api, home, *overrides),
            **(provider_options or {}),
        },
        **agent_options,
    )


def event_types(result: RunResult) -> list[str]:
    return [envelope.event.type for envelope in result.events]


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, list):
        return [text for item in value for text in _strings(item)]
    return []


def login_agent(api: MockCodex, home: Path, cwd: Path, cli_login: str) -> Agent:
    return Agent(
        provider="codex",
        model=MODEL,
        cwd=cwd,
        timeout=60,
        cli_login=cli_login,
        provider_options={"config": codex_config(api, home)},
    )


async def test_api_key_login_leaves_a_chatgpt_login_untouched(mock_api, codex_home, tmp_path):
    seed_chatgpt_login(codex_home)
    auth = codex_home / "auth.json"
    before = auth.read_text(encoding="utf-8")
    mock_api.steps = [{"text": "hello"}]

    result = await codex_agent(mock_api, codex_home, tmp_path).run("hi")

    assert result.ok, result.error
    assert result.final_text == "hello"
    assert [r["headers"].get("authorization") for r in mock_api.requests] == ["Bearer sk-mock-key"]
    assert auth.read_text(encoding="utf-8") == before


SECRET_KEY = "sk-mock-SECRET-4242"


def env_key_agent(api: MockCodex, home: Path, cwd: Path, *overrides: str) -> Agent:
    """An agent whose only API key is OPENAI_API_KEY in the run env."""

    config = codex_config(api, home, *overrides)
    for name in ("OPENAI_API_KEY", "CODEX_API_KEY"):
        del config["env"][name]
    return Agent(
        provider="codex",
        model=MODEL,
        cwd=cwd,
        timeout=60,
        env={"OPENAI_API_KEY": SECRET_KEY},
        provider_options={"config": config},
    )


def files_containing(root: Path, text: str) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file() and text.encode() in p.read_bytes()]


async def test_an_env_api_key_stays_out_of_codex_home_and_commands(mock_api, codex_home, tmp_path):
    mock_api.steps = [{"shell": 'printf "%s" "${OPENAI_API_KEY:-unset}"'}, {"text": "done"}]

    result = await env_key_agent(mock_api, codex_home, tmp_path).run("hi")

    assert result.ok, result.error
    assert [r["headers"].get("authorization") for r in mock_api.requests] == [
        f"Bearer {SECRET_KEY}"
    ] * 2
    [shell] = [e.event for e in result.events if e.event.type == "tool_result"]
    assert shell.output == "unset"
    assert files_containing(codex_home, SECRET_KEY) == []


async def test_a_provider_env_key_keeps_the_key_without_persisting_it(
    mock_api, codex_home, tmp_path
):
    mock_api.steps = [{"shell": "true"}, {"text": "done"}]
    agent = env_key_agent(
        mock_api,
        codex_home,
        tmp_path,
        "model_providers.mock.requires_openai_auth=false",
        'model_providers.mock.env_key="OPENAI_API_KEY"',
    )

    result = await agent.run("hi")

    assert result.ok, result.error
    assert [r["headers"].get("authorization") for r in mock_api.requests] == [
        f"Bearer {SECRET_KEY}"
    ] * 2
    assert list(codex_home.glob("shell_snapshots/*")) == []
    assert files_containing(codex_home, SECRET_KEY) == []


class Detail(BaseModel):
    note: str
    score: int = 0


class Report(BaseModel):
    a: int
    b: str | None = None
    detail: Detail = Field(description="Supporting detail.")


async def test_structured_output_is_sent_in_strict_form(mock_api, codex_home, tmp_path):
    answer = {"a": 1, "b": None, "detail": {"note": "n", "score": None}}
    mock_api.steps = [{"text": json.dumps(answer)}]

    result = await codex_agent(mock_api, codex_home, tmp_path, output_schema=Report).run("go")

    assert result.ok, result.error
    assert result.structured_output == Report(a=1, detail=Detail(note="n"))
    text_format = mock_api.requests[0]["body"]["text"]["format"]
    assert text_format["type"] == "json_schema"
    assert text_format["strict"] is True
    schema = text_format["schema"]
    assert schema["required"] == ["a", "b", "detail"]
    assert schema["additionalProperties"] is False
    detail = schema["properties"]["detail"]
    assert "$ref" not in detail
    assert detail["description"] == "Supporting detail."
    assert detail["required"] == ["note", "score"]
    assert detail["additionalProperties"] is False
    assert "default" not in json.dumps(schema)


ECHO_MCP_SERVER = '''
import os
import sys

from mcp.server.mcpserver import MCPServer as Server


def echo() -> str:
    """Return this server's arguments and GREETING."""
    return "|".join([*sys.argv[1:], os.environ.get("GREETING", "<unset>")])


server = Server("echo")
server.add_tool(echo, name="echo", description="Echo.", structured_output=False)
server.run("stdio")
'''

UNICODE_TEXT = "fox \U0001f98a café del\x7f"


async def test_unicode_config_reaches_codex_intact(mock_api, codex_home, tmp_path):
    script = tmp_path / "echo_server.py"
    script.write_text(ECHO_MCP_SERVER, encoding="utf-8")
    # tool_search returns the deferred spawn_agent tool, which lists subagent descriptions.
    mock_api.steps = [
        {"tool_search": "spawn agent", "tool": {"name": "mcp__echo__echo"}},
        {"text": "done"},
    ]
    agent = codex_agent(
        mock_api,
        codex_home,
        tmp_path,
        mcp_servers=[
            McpStdioServer(
                name="echo",
                command=sys.executable,
                args=[str(script), UNICODE_TEXT],
                env={"GREETING": UNICODE_TEXT},
                default_tools_approval_mode="approve",
                # Codex only waits for required servers before the first model request.
                required=True,
            )
        ],
        subagents={"fox": SubagentDef(description=UNICODE_TEXT, prompt=UNICODE_TEXT)},
    )

    result = await agent.run("hi")

    assert result.ok, result.error
    [tool_result] = [e.event for e in result.events if e.event.type == "tool_result"]
    [content] = json.loads(tool_result.output or "{}")["content"]
    assert content["text"] == f"{UNICODE_TEXT}|{UNICODE_TEXT}"
    followup = _strings(mock_api.requests[-1]["body"]["input"])
    assert any(f"fox: {{\n{UNICODE_TEXT}\n}}" in text for text in followup)


async def test_mcp_env_passthrough_comes_from_the_run_env(mock_api, codex_home, tmp_path):
    script = tmp_path / "echo_server.py"
    script.write_text(ECHO_MCP_SERVER, encoding="utf-8")

    def echo_server(name: str, **options: Any) -> McpStdioServer:
        return McpStdioServer(
            name=name,
            command=sys.executable,
            args=[str(script)],
            env_passthrough=["GREETING"],
            default_tools_approval_mode="approve",
            required=True,
            **options,
        )

    mock_api.steps = [
        {
            "tool": [
                {"name": "mcp__inherits__echo"},
                {"name": "mcp__overrides__echo"},
            ]
        },
        {"text": "done"},
    ]
    servers = [echo_server("inherits"), echo_server("overrides", env={"GREETING": "explicit"})]
    agent = codex_agent(
        mock_api, codex_home, tmp_path, env={"GREETING": "from-run"}, mcp_servers=servers
    )

    result = await agent.run("hi")

    assert result.ok, result.error
    outputs = {
        e.event.name: json.loads(e.event.output or "{}")["content"][0]["text"]
        for e in result.events
        if e.event.type == "tool_result"
    }
    assert outputs == {"inherits.echo": "from-run", "overrides.echo": "explicit"}


async def test_wrapper_tools_see_the_parent_env_and_imports(
    mock_api, codex_home, tmp_path, monkeypatch
):
    modules = tmp_path / "modules"
    modules.mkdir()
    (modules / "runtime_tools.py").write_text(
        "import os\n\nPREFIX = 'token:'\n\n\n"
        "def token() -> str:\n"
        '    """Return the token."""\n'
        "    return PREFIX + os.environ.get('WRAPPER_TOOL_TOKEN', '<unset>')\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(modules))
    token = importlib.import_module("runtime_tools").token
    mock_api.steps = [
        {"tool": {"name": "mcp__agent_sdk_wrapper_tools__token"}},
        {"text": "done"},
    ]
    agent = codex_agent(
        mock_api, codex_home, tmp_path, tools=[token], env={"WRAPPER_TOOL_TOKEN": "t0k3n"}
    )

    result = await agent.run("token?")

    assert result.ok, result.error
    [tool_result] = [e.event for e in result.events if e.event.type == "tool_result"]
    assert not tool_result.is_error, tool_result.output
    [content] = json.loads(tool_result.output or "{}")["content"]
    assert content["text"] == "token:t0k3n"


async def test_wrapper_tools_validate_and_report_like_the_claude_handler(
    mock_api, codex_home, tmp_path, monkeypatch
):
    modules = tmp_path / "modules"
    modules.mkdir()
    (modules / "kwargs_tools.py").write_text(
        "def search(query: str, **filters: str) -> str:\n"
        '    """Search with any filters."""\n'
        "    return f'{query} {sorted(filters.items())}'\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(modules))
    search = importlib.import_module("kwargs_tools").search

    def shout(text: str) -> str:
        """Upper-case text."""
        if not text:
            raise ValueError("nothing to shout")
        return text.upper()

    # A script's tools live in __main__, which the tool server rebuilds from source.
    shout.__module__ = "__main__"
    shout.__qualname__ = "shout"
    prefix = "mcp__agent_sdk_wrapper_tools__"
    mock_api.steps = [
        {
            "tool": [
                {"name": f"{prefix}search", "input": {"query": "q", "lang": "en"}},
                {"name": f"{prefix}shout", "input": {"text": "hi"}},
                {"name": f"{prefix}shout", "input": {"text": ""}},
            ]
        },
        {"text": "done"},
    ]

    result = await codex_agent(mock_api, codex_home, tmp_path, tools=[search, shout]).run("go")

    assert result.ok, result.error
    outputs = {
        e.event.id: (e.event.is_error, json.loads(e.event.output or "{}")["content"][0]["text"])
        for e in result.events
        if e.event.type == "tool_result"
    }
    assert outputs == {
        "call_0_0": (False, "q [('lang', 'en')]"),
        "call_0_1": (False, "HI"),
        "call_0_2": (True, "Error: nothing to shout"),
    }


async def test_a_required_mcp_server_that_exits_is_not_transient(mock_api, codex_home, tmp_path):
    broken = McpStdioServer(name="broken", command="/bin/sh", args=["-c", "exit 3"], required=True)

    result = await codex_agent(mock_api, codex_home, tmp_path, mcp_servers=[broken]).run("hi")

    # "connection closed: initialize response" is not a dropped API connection.
    assert result.error_type == "provider_exception"
    assert "required MCP servers failed to initialize: broken" in (result.error or "")
    assert mock_api.requests == []


async def test_a_tool_the_server_cannot_import_fails_the_run_with_the_import_error(
    mock_api, codex_home, tmp_path, monkeypatch
):
    import types

    ghost_tools = types.ModuleType("ghost_tools")
    exec("def ghost() -> str:\n    return 'boo'\n", ghost_tools.__dict__)
    monkeypatch.setitem(sys.modules, "ghost_tools", ghost_tools)

    result = await codex_agent(mock_api, codex_home, tmp_path, tools=[ghost_tools.ghost]).run("hi")

    assert result.error_type == "provider_exception"
    assert "cannot load tool 'ghost': ModuleNotFoundError: No module named 'ghost_tools'" in (
        result.error or ""
    )
    assert mock_api.requests == []


async def test_session_reports_the_model_and_mcp_startup_failures(mock_api, codex_home, tmp_path):
    broken = McpStdioServer(name="broken", command="/bin/sh", args=["-c", "exit 3"])

    result = await codex_agent(mock_api, codex_home, tmp_path, mcp_servers=[broken]).run("hi")

    assert result.ok, result.error
    events = [envelope.event for envelope in result.events]
    assert [e.model for e in events if e.type == "session_info"] == [MODEL]
    warnings = [e.message for e in events if e.type == "warning"]
    assert any("`broken` failed to start" in message for message in warnings), warnings


async def test_turns_of_a_continued_thread_report_only_their_own_requests(
    mock_api, codex_home, tmp_path
):
    # Codex repeats the thread's unchanged usage before each failed attempt.
    failed = {"stream_error": {"code": "server_error", "message": "boom"}}
    agent = codex_agent(
        mock_api,
        codex_home,
        tmp_path,
        "model_providers.mock.stream_max_retries=1",
        continue_session=True,
    )
    usages = []
    for plan in (
        [{"text": "one", "usage": (100, 10)}],
        [failed, {"text": "two", "usage": (300, 30)}],
        [failed],
    ):
        mock_api.set_steps(plan)
        mock_api.requests.clear()
        usages.append((await agent.run("hi")).usage)

    assert usages == [
        TokenUsage(input_tokens=100, output_tokens=10, total_tokens=110, requests=1),
        TokenUsage(input_tokens=300, output_tokens=30, total_tokens=330, requests=1),
        None,
    ]


async def test_an_exhausted_context_window_reports_no_usage(mock_api, codex_home, tmp_path):
    mock_api.steps = [
        {
            "stream_error": {
                "code": "context_length_exceeded",
                "message": "Input exceeds the window.",
            }
        }
    ]

    result = await codex_agent(mock_api, codex_home, tmp_path).run("hi")

    assert result.error_type == "context_window_exceeded"
    assert result.usage is None


async def test_reasoning_tokens_without_a_reasoning_item_yield_empty_thinking(
    mock_api, codex_home, tmp_path
):
    mock_api.steps = [{"text": "answer", "usage": [100, 10, 7]}]

    result = await codex_agent(mock_api, codex_home, tmp_path).run("hi")

    assert result.ok, result.error
    events = [e.event for e in result.events if e.event.type in ("thinking", "usage")]
    assert [(e.type, getattr(e, "text", None)) for e in events] == [
        ("thinking", ""),
        ("usage", None),
    ]
    assert events[1].usage.reasoning_output_tokens == 7


async def test_rejected_api_key_is_one_authentication_error(mock_api, codex_home, tmp_path):
    mock_api.steps = [
        {
            "status": 401,
            "body": {
                "error": {
                    "message": "Incorrect API key provided: sk-mock.",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key",
                }
            },
        }
    ]

    result = await codex_agent(mock_api, codex_home, tmp_path).run("hi")

    assert not result.ok
    errors = [e.event for e in result.events if e.event.type == "error"]
    assert [e.error_type for e in errors] == ["authentication_failed"]
    assert "401 Unauthorized" in errors[0].message


async def test_non_ascii_error_body_keeps_its_text(mock_api, codex_home, tmp_path):
    mock_api.steps = [{"status": 400, "body": "Offline gateway probe ✓"}]

    result = await codex_agent(mock_api, codex_home, tmp_path).run("hi")

    errors = [e.event for e in result.events if e.event.type == "error"]
    assert [e.message for e in errors] == ["Offline gateway probe ✓"]


async def test_signal_killed_app_server_raises_process_terminated(mock_api, codex_home, tmp_path):
    from openai_codex import AsyncCodex, CodexConfig

    from agent_sdk_wrapper import ProcessTerminatedError, RunRequest
    from agent_sdk_wrapper.providers.openai_provider import OpenAIProvider

    mock_api.steps = [{"hang": 30}]
    config = codex_config(mock_api, codex_home, 'cli_auth_credentials_store="ephemeral"')
    req = RunRequest(provider="openai", prompt="hi", model=MODEL, cwd=tmp_path)

    async with AsyncCodex(config=CodexConfig(**config)) as codex:
        await codex.login_api_key("sk-mock-key")
        pid = codex._client._sync._proc.pid

        async def kill_once_requested() -> None:
            while not mock_api.requests:
                await asyncio.sleep(0.05)
            os.kill(pid, signal.SIGKILL)

        killer = asyncio.create_task(kill_once_requested())
        with pytest.raises(ProcessTerminatedError) as raised:
            async with asyncio.timeout(30):
                async for _ in OpenAIProvider(codex=codex).stream(req):
                    pass
        await killer

    assert raised.value.signal == signal.SIGKILL


async def test_cli_login_require_uses_the_stored_chatgpt_login(mock_api, codex_home, tmp_path):
    access = seed_chatgpt_login(codex_home)
    mock_api.steps = [{"shell": "true"}, {"text": "hello"}]

    result = await login_agent(mock_api, codex_home, tmp_path, "require").run("hi")

    assert result.ok, result.error
    assert [r["headers"].get("authorization") for r in mock_api.requests] == [
        f"Bearer {access}"
    ] * 2
    # The runtime env is not copied into CODEX_HOME.
    assert list(codex_home.glob("shell_snapshots/*")) == []
