"""Drive the real Codex runtime against a local mock Responses API."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import signal
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field

from agent_sdk_wrapper import Agent, McpStdioServer, RunResult, SubagentDef, TokenUsage

pytest.importorskip("codex_cli_bin")

MODEL = "gpt-5.4"


class MockResponses:
    """Serve scripted Responses API turns and record every request."""

    def __init__(self) -> None:
        self.plan: list[dict[str, Any]] = [{"text": "ok"}]
        self.requests: list[dict[str, Any]] = []
        self.release = threading.Event()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._server.daemon_threads = True
        self._server.block_on_close = False
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def posts(self) -> list[dict[str, Any]]:
        return [r for r in self.requests if r["method"] == "POST"]

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self.release.set()
        self._server.shutdown()
        self._server.server_close()

    def _next_step(self) -> tuple[int, dict[str, Any]]:
        index = len(self.posts())
        return index, self.plan[min(index - 1, len(self.plan) - 1)]

    def _handler(self):
        mock = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:
                pass

            def _send(self, status: int, body: bytes, content_type: str) -> None:
                self.send_response(status)
                self.send_header("content-type", content_type)
                self.send_header("content-length", str(len(body)))
                self.send_header("connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                mock.requests.append({"method": "GET", "path": self.path})
                self._send(404, b"{}", "application/json")

            def do_POST(self) -> None:
                raw = self.rfile.read(int(self.headers.get("content-length") or 0))
                mock.requests.append(
                    {
                        "method": "POST",
                        "path": self.path,
                        "authorization": self.headers.get("authorization"),
                        "body": json.loads(raw),
                    }
                )
                index, step = mock._next_step()
                if "hang" in step:
                    mock.release.wait(step["hang"])
                    return
                if "status" in step and "text" in step:
                    body = step["text"].encode()
                    self._send(step["status"], body, "text/plain; charset=utf-8")
                    return
                if "status" in step:
                    body = json.dumps(step["body"]).encode()
                    self._send(step["status"], body, "application/json")
                    return
                self._send(200, _sse(index, step).encode(), "text/event-stream")

        return Handler


def _sse(index: int, step: dict[str, Any]) -> str:
    response_id = f"resp_{index}"
    items: list[dict[str, Any]] = list(step.get("items", []))
    for n, call in enumerate(step.get("calls", [step["call"]] if "call" in step else [])):
        items.append(
            {
                "type": "function_call",
                "id": f"fc_{index}_{n}",
                "call_id": f"call_{index}_{n}",
                "name": call["name"],
                "arguments": json.dumps(call.get("args", {})),
                **({"namespace": call["namespace"]} if "namespace" in call else {}),
            }
        )
    if "shell" in step:
        items.append(
            {
                "type": "function_call",
                "id": f"fc_{index}",
                "call_id": f"call_{index}",
                "name": "exec_command",
                "arguments": json.dumps({"cmd": step["shell"]}),
            }
        )
    if "text" in step:
        items.append(
            {
                "type": "message",
                "role": "assistant",
                "id": f"msg_{index}",
                "content": [{"type": "output_text", "text": step["text"]}],
            }
        )
    input_tokens, output_tokens = step.get("usage", (100, 10))
    completed = {
        "type": "response.completed",
        "response": {
            "id": response_id,
            "usage": {
                "input_tokens": input_tokens,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": output_tokens,
                "output_tokens_details": {"reasoning_tokens": step.get("reasoning", 0)},
                "total_tokens": input_tokens + output_tokens,
            },
        },
    }
    if "failed" in step:
        failed = {"id": response_id, "status": "failed", "error": step["failed"]}
        completed = {"type": "response.failed", "response": failed}
    events = [
        {"type": "response.created", "response": {"id": response_id}},
        *({"type": "response.output_item.done", "item": item} for item in items),
        completed,
    ]
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


@pytest.fixture
def mock_api():
    api = MockResponses()
    api.start()
    yield api
    api.stop()


@pytest.fixture
def codex_home(tmp_path: Path) -> Path:
    home = tmp_path / "codex-home"
    home.mkdir()
    return home


def codex_config(api: MockResponses, home: Path, *overrides: str) -> dict[str, Any]:
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
    api: MockResponses,
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


def seed_chatgpt_login(home: Path) -> str:
    """Write a stored ChatGPT login Codex accepts offline; return its access token.

    Codex needs plan claims in the ID token, and a fresh last_refresh with a
    far-future exp avoids a token refresh.
    """

    import base64
    from datetime import UTC, datetime

    def part(value: dict[str, Any]) -> str:
        return base64.urlsafe_b64encode(json.dumps(value).encode()).rstrip(b"=").decode()

    def jwt(claims: dict[str, Any]) -> str:
        return f"{part({'alg': 'RS256'})}.{part(claims)}.c2ln"

    claims = {
        "email": "a@b.c",
        "exp": 4102444800,
        "https://api.openai.com/auth": {
            "chatgpt_plan_type": "pro",
            "chatgpt_account_id": "acct_1",
            "chatgpt_user_id": "user_1",
        },
    }
    access = jwt({**claims, "sub": "access"})
    (home / "auth.json").write_text(
        json.dumps(
            {
                "OPENAI_API_KEY": None,
                "tokens": {
                    "id_token": jwt(claims),
                    "access_token": access,
                    "refresh_token": "chatgpt-refresh",
                    "account_id": "acct_1",
                },
                "last_refresh": datetime.now(UTC).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    return access


def login_agent(api: MockResponses, home: Path, cwd: Path, cli_login: str) -> Agent:
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
    mock_api.plan = [{"text": "hello"}]

    result = await codex_agent(mock_api, codex_home, tmp_path).run("hi")

    assert result.ok, result.error
    assert result.final_text == "hello"
    assert [r["authorization"] for r in mock_api.posts()] == ["Bearer sk-mock-key"]
    assert auth.read_text(encoding="utf-8") == before


SECRET_KEY = "sk-mock-SECRET-4242"


def env_key_agent(api: MockResponses, home: Path, cwd: Path, *overrides: str) -> Agent:
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


async def test_an_env_api_key_stays_out_of_codex_home_and_commands(
    mock_api, codex_home, tmp_path
):
    mock_api.plan = [{"shell": 'printf "%s" "${OPENAI_API_KEY:-unset}"'}, {"text": "done"}]

    result = await env_key_agent(mock_api, codex_home, tmp_path).run("hi")

    assert result.ok, result.error
    assert [r["authorization"] for r in mock_api.posts()] == [f"Bearer {SECRET_KEY}"] * 2
    [shell] = [e.event for e in result.events if e.event.type == "tool_result"]
    assert shell.output == "unset"
    assert list(codex_home.glob("shell_snapshots/*"))
    assert files_containing(codex_home, SECRET_KEY) == []


async def test_a_provider_env_key_keeps_the_key_but_no_shell_snapshot(
    mock_api, codex_home, tmp_path
):
    mock_api.plan = [{"shell": "true"}, {"text": "done"}]
    agent = env_key_agent(
        mock_api,
        codex_home,
        tmp_path,
        "model_providers.mock.requires_openai_auth=false",
        'model_providers.mock.env_key="OPENAI_API_KEY"',
    )

    result = await agent.run("hi")

    assert result.ok, result.error
    assert [r["authorization"] for r in mock_api.posts()] == [f"Bearer {SECRET_KEY}"] * 2
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
    mock_api.plan = [{"text": json.dumps(answer)}]

    result = await codex_agent(mock_api, codex_home, tmp_path, output_schema=Report).run("go")

    assert result.ok, result.error
    assert result.structured_output == Report(a=1, detail=Detail(note="n"))
    text_format = mock_api.posts()[0]["body"]["text"]["format"]
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


def _web_search_tools(api: MockResponses) -> list[dict[str, Any]]:
    return [t for t in api.posts()[-1]["body"]["tools"] if t.get("type") == "web_search"]


async def test_web_tools_controls_the_web_search_tool(mock_api, codex_home, tmp_path):
    disabled = await codex_agent(mock_api, codex_home, tmp_path, web_tools=False).run("hi")
    assert disabled.ok, disabled.error
    assert _web_search_tools(mock_api) == []

    enabled = await codex_agent(mock_api, codex_home, tmp_path, web_tools=True).run("hi")
    assert enabled.ok, enabled.error
    assert [t["external_web_access"] for t in _web_search_tools(mock_api)] == [True]


async def test_sandbox_keeps_workspace_write_config(mock_api, codex_home, tmp_path):
    agent = codex_agent(
        mock_api,
        codex_home,
        tmp_path,
        "sandbox_workspace_write.network_access=true",
        provider_options={"sandbox": "workspace-write"},
    )

    result = await agent.run("hi")

    assert result.ok, result.error
    context = json.dumps(mock_api.posts()[0]["body"]["input"])
    assert "Network access is enabled" in context


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
    search = {
        "type": "tool_search_call",
        "id": "ts_1",
        "call_id": "ts_1",
        "execution": "client",
        "status": "completed",
        "arguments": {"query": "spawn agent"},
    }
    mock_api.plan = [
        {"items": [search], "call": {"name": "echo", "namespace": "mcp__echo"}},
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
    followup = _strings(mock_api.posts()[-1]["body"]["input"])
    assert any(f"fox: {{\n{UNICODE_TEXT}\n}}" in text for text in followup)


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
    mock_api.plan = [
        {"call": {"name": "token", "namespace": "mcp__agent_sdk_wrapper_tools"}},
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
    namespace = "mcp__agent_sdk_wrapper_tools"
    mock_api.plan = [
        {
            "calls": [
                {"name": "search", "namespace": namespace, "args": {"query": "q", "lang": "en"}},
                {"name": "shout", "namespace": namespace, "args": {"text": "hi"}},
                {"name": "shout", "namespace": namespace, "args": {"text": ""}},
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
        "call_1_0": (False, "q [('lang', 'en')]"),
        "call_1_1": (False, "HI"),
        "call_1_2": (True, "Error: nothing to shout"),
    }


async def test_session_reports_the_model_and_mcp_startup_failures(
    mock_api, codex_home, tmp_path
):
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
    failed = {"failed": {"code": "server_error", "message": "boom"}}
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
        mock_api.plan = plan
        mock_api.requests.clear()
        usages.append((await agent.run("hi")).usage)

    assert usages == [
        TokenUsage(input_tokens=100, output_tokens=10, total_tokens=110, requests=1),
        TokenUsage(input_tokens=300, output_tokens=30, total_tokens=330, requests=1),
        None,
    ]


async def test_an_exhausted_context_window_reports_no_usage(mock_api, codex_home, tmp_path):
    mock_api.plan = [
        {"failed": {"code": "context_length_exceeded", "message": "Input exceeds the window."}}
    ]

    result = await codex_agent(mock_api, codex_home, tmp_path).run("hi")

    assert result.error_type == "context_window_exceeded"
    assert result.usage is None


async def test_reasoning_tokens_without_a_reasoning_item_yield_empty_thinking(
    mock_api, codex_home, tmp_path
):
    mock_api.plan = [{"text": "answer", "reasoning": 7}]

    result = await codex_agent(mock_api, codex_home, tmp_path).run("hi")

    assert result.ok, result.error
    events = [e.event for e in result.events if e.event.type in ("thinking", "usage")]
    assert [(e.type, getattr(e, "text", None)) for e in events] == [
        ("thinking", ""),
        ("usage", None),
    ]
    assert events[1].usage.reasoning_output_tokens == 7


async def test_rejected_api_key_is_one_authentication_error(mock_api, codex_home, tmp_path):
    mock_api.plan = [
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
    mock_api.plan = [{"status": 400, "text": "Offline gateway probe ✓"}]

    result = await codex_agent(mock_api, codex_home, tmp_path).run("hi")

    errors = [e.event for e in result.events if e.event.type == "error"]
    assert [e.message for e in errors] == ["Offline gateway probe ✓"]


async def test_signal_killed_app_server_raises_process_terminated(
    mock_api, codex_home, tmp_path
):
    from openai_codex import AsyncCodex, CodexConfig

    from agent_sdk_wrapper import ProcessTerminatedError, RunRequest
    from agent_sdk_wrapper.providers.openai_provider import OpenAIProvider

    mock_api.plan = [{"hang": 30}]
    config = codex_config(mock_api, codex_home, 'cli_auth_credentials_store="ephemeral"')
    req = RunRequest(provider="openai", prompt="hi", model=MODEL, cwd=tmp_path)

    async with AsyncCodex(config=CodexConfig(**config)) as codex:
        await codex.login_api_key("sk-mock-key")
        pid = codex._client._sync._proc.pid

        async def kill_once_requested() -> None:
            while not mock_api.posts():
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
    mock_api.plan = [{"text": "hello"}]

    result = await login_agent(mock_api, codex_home, tmp_path, "require").run("hi")

    assert result.ok, result.error
    assert [r["authorization"] for r in mock_api.posts()] == [f"Bearer {access}"]


async def test_cli_login_require_rejects_a_stored_api_key_before_any_request(
    mock_api, codex_home, tmp_path
):
    (codex_home / "auth.json").write_text(
        json.dumps({"auth_mode": "apikey", "OPENAI_API_KEY": "sk-stored"}), encoding="utf-8"
    )

    result = await login_agent(mock_api, codex_home, tmp_path, "require").run("hi")

    errors = [env.event for env in result.events if env.event.type == "error"]
    assert [e.error_type for e in errors] == ["authentication_failed"]
    assert mock_api.posts() == []


async def test_cli_login_deny_refuses_a_chatgpt_logged_in_client_before_any_request(
    mock_api, codex_home, tmp_path
):
    from openai_codex import AsyncCodex, CodexConfig

    from agent_sdk_wrapper import RunRequest
    from agent_sdk_wrapper.providers.openai_provider import OpenAIProvider

    seed_chatgpt_login(codex_home)
    req = RunRequest(provider="openai", prompt="hi", model=MODEL, cwd=tmp_path)

    async with AsyncCodex(config=CodexConfig(**codex_config(mock_api, codex_home))) as codex:
        events = [event async for event in OpenAIProvider(codex=codex).stream(req)]

    assert [(e.type, getattr(e, "error_type", None)) for e in events] == [
        ("error", "authentication_failed")
    ]
    assert mock_api.posts() == []


async def test_cli_login_deny_accepts_a_per_thread_provider_without_openai_auth(
    mock_api, codex_home, tmp_path
):
    config = codex_config(mock_api, codex_home)
    config["config_overrides"] = (
        *(
            item
            for item in config["config_overrides"]
            if not item.startswith(("model_provider=", "model_providers.mock.requires_openai_auth"))
        ),
        "model_providers.mock.requires_openai_auth=false",
    )
    mock_api.plan = [{"text": "hello"}]
    agent = Agent(
        provider="codex",
        model=MODEL,
        cwd=tmp_path,
        timeout=60,
        provider_options={"config": config, "model_provider": "mock"},
    )

    result = await agent.run("hi")

    assert result.ok, result.error
    assert len(mock_api.posts()) == 1

