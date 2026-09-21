"""Run a conformance case (docs/fixtures/CONFORMANCE.md) through the Python Agent."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import copy
import json
import os
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from agent_sdk_wrapper import (
    Agent,
    ConfigError,
    McpStdioServer,
    RunFailedError,
    RunResult,
    SubagentDef,
)

from .mocks import MockApi, MockClaude, MockCodex

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "docs" / "schemas").is_dir())
SPEC = json.loads((ROOT / "docs/fixtures/conformance-v1.json").read_text(encoding="utf-8"))
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
LIVE_MODELS = {
    "anthropic": ("AGENT_SDK_WRAPPER_ANTHROPIC_MODEL", "claude-haiku-4-5"),
    "codex": ("AGENT_SDK_WRAPPER_OPENAI_MODEL", "gpt-5.6-luna"),
}
LIVE_KEYS = {"anthropic": "ANTHROPIC_API_KEY", "codex": "OPENAI_API_KEY"}
# Live runs keep their artifacts for inspection unless a case writes its own.
LIVE_ARTIFACTS = Path(
    os.environ.get("AGENT_SDK_WRAPPER_TEST_ARTIFACTS_DIR")
    or Path(__file__).resolve().parents[2]
    / "results"
    / "integration-runs"
    / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
).resolve()
MOCK_KEY = {"anthropic": "sk-ant-mock", "codex": "sk-mock-key"}
DEAD_PROXY = "http://127.0.0.1:9"
# A run that outlives this is a hung test, not a case result.
RUN_LIMIT_S = 300


def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def shout(text: str) -> str:
    """Upper-case text."""
    if not text:
        raise ValueError("nothing to shout")
    return text.upper()


class Weather(BaseModel):
    city: str
    temp_c: int


TOOLS = {fn.__name__: fn for fn in (add, shout)}
SCHEMAS = {"Weather": Weather}


def python_view(case: dict[str, Any]) -> dict[str, Any] | str:
    """The case with its Python override applied, or the reason Python skips it."""

    override = case.get("languages", {}).get("python")
    if isinstance(override, str):
        return override
    return {**case, **(override or {})}


def live_view(case: dict[str, Any]) -> dict[str, Any]:
    """The case as its live section describes it."""

    live = case["live"]
    live_runs = live.get("runs", [])
    runs = [
        {**run, **(live_runs[n] if n < len(live_runs) else {})}
        for n, run in enumerate(case.get("runs", []))
    ]
    variable, default = LIVE_MODELS[case["provider"]]
    model = os.environ.get(variable) or live.get("options", {}).get("model", default)
    return {
        **case,
        "prompt": live.get("prompt", case["prompt"]),
        "options": {**case.get("options", {}), **live.get("options", {}), "model": model},
        "expect": live.get("expect", {}),
        "runs": runs,
    }


@dataclass
class Scratch:
    """A case's own directories; ``cwd`` is relative to ``root``."""

    root: Path
    cwd: str = "work"

    def __post_init__(self) -> None:
        for name in ("home", "codex_home", "claude_config", "anthropic_config", self.cwd):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def dir(self, name: str) -> Path:
        return self.root / (self.cwd if name == "cwd" else name)


def isolate(monkeypatch: pytest.MonkeyPatch, scratch: Scratch, *, live: bool) -> None:
    """Keep the host's credentials, config and network out of the runtimes."""

    keep = {name: os.environ[name] for name in LIVE_KEYS.values() if live and name in os.environ}
    for name in list(os.environ):
        if name.startswith(("ANTHROPIC_", "OPENAI_", "CODEX_", "CLAUDE_CODE_")) or name in (
            "CLAUDECODE",
            "MODEL",
        ):
            monkeypatch.delenv(name)
    env = {
        **keep,
        "HOME": str(scratch.dir("home")),
        "CODEX_HOME": str(scratch.dir("codex_home")),
        "CLAUDE_CONFIG_DIR": str(scratch.dir("claude_config")),
        "ANTHROPIC_CONFIG_DIR": str(scratch.dir("anthropic_config")),
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    }
    if not live:
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            env[name] = env[name.lower()] = DEAD_PROXY
        env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost"
    for name, value in env.items():
        monkeypatch.setenv(name, value)


def seed_chatgpt_login(home: Path) -> str:
    """Write a stored ChatGPT login Codex accepts offline; return its access token.

    Codex needs plan claims in the ID token, and a fresh last_refresh with a
    far-future exp avoids a token refresh.
    """

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
    auth = {
        "OPENAI_API_KEY": None,
        "tokens": {
            "id_token": jwt(claims),
            "access_token": access,
            "refresh_token": "chatgpt-refresh",
            "account_id": "acct_1",
        },
        "last_refresh": datetime.now(UTC).isoformat(),
    }
    (home / "auth.json").write_text(json.dumps(auth), encoding="utf-8")
    return access


def prepare(setup: dict[str, Any], scratch: Scratch) -> None:
    for name, content in setup.get("files", {}).items():
        root, _, rel = name.partition("/")
        path = scratch.dir(root) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    login = setup.get("codex_login")
    if login == "chatgpt":
        seed_chatgpt_login(scratch.dir("codex_home"))
    elif login == "api_key":
        auth = {"auth_mode": "apikey", "OPENAI_API_KEY": "sk-stored"}
        (scratch.dir("codex_home") / "auth.json").write_text(json.dumps(auth), encoding="utf-8")


def codex_wiring(api: MockApi, *, select: bool) -> tuple[str, ...]:
    """Define the mock as model provider ``mock``, with runtime retries off."""

    overrides = (
        'model_providers.mock.name="mock"',
        f'model_providers.mock.base_url="{api.base_url}"',
        'model_providers.mock.wire_api="responses"',
        "model_providers.mock.requires_openai_auth=true",
        "model_providers.mock.request_max_retries=0",
        "model_providers.mock.stream_max_retries=0",
        "model_providers.mock.supports_websockets=false",
    )
    return ('model_provider="mock"', *overrides) if select else overrides


@dataclass
class Context:
    provider: str
    scratch: Scratch
    api: MockApi | None
    setup: dict[str, Any]
    session_id: str | None = None
    events: list[Any] = field(default_factory=list)
    provider_events: list[Any] = field(default_factory=list)
    clients: contextlib.AsyncExitStack = field(default_factory=contextlib.AsyncExitStack)

    def substitute(self, value: Any) -> Any:
        if isinstance(value, str):
            return value.replace("{session_id}", self.session_id or "")
        if isinstance(value, list):
            return [self.substitute(item) for item in value]
        if isinstance(value, dict):
            return {key: self.substitute(item) for key, item in value.items()}
        return value

    def path(self, value: str) -> Path:
        return self.scratch.root / value

    async def native(self, options: dict[str, Any]) -> dict[str, Any]:
        """Map JSON option values to the Python values they name."""

        out: dict[str, Any] = {}
        for key, value in self.substitute(options).items():
            if key == "tools":
                value = [TOOLS[name] for name in value]
            elif key == "output_schema":
                value = SCHEMAS[value]
            elif key == "mcp_servers":
                value = [mcp_server(spec) for spec in value]
            elif key == "subagents":
                value = {name: SubagentDef(**spec) for name, spec in value.items()}
            elif key in ("cwd", "trace_file", "artifacts_dir"):
                value = self.path(value)
            elif key == "on_event" and value is True:
                value = self.events.append
            elif key == "on_provider_event" and value is True:
                value = self.provider_events.append
            elif key == "provider_options":
                value = await self.provider_options(value)
            out[key] = value
        return out

    async def provider_options(self, options: dict[str, Any]) -> dict[str, Any]:
        options = dict(options)
        if self.provider != "codex":
            return options
        config = copy.deepcopy(options.pop("config", {}))
        if self.api is not None and self.setup.get("codex_provider") != "builtin":
            wiring = codex_wiring(self.api, select="model_provider" not in options)
            config["config_overrides"] = (*wiring, *config.get("config_overrides", ()))
        if options.get("codex") == "client":
            from openai_codex import AsyncCodex, CodexConfig

            options["codex"] = await self.clients.enter_async_context(
                AsyncCodex(config=CodexConfig(**config))
            )
        elif config:
            options["config"] = config
        return options


def mcp_server(spec: dict[str, Any]) -> McpStdioServer:
    spec = dict(spec)
    if spec.pop("fixture", None) == "simple_mcp_server":
        spec.update(command=sys.executable, args=[str(FIXTURES / "simple_mcp_server.py")])
    return McpStdioServer(**spec)


@dataclass
class Outcome:
    result: RunResult | None = None
    config_error: ConfigError | None = None
    raised: Exception | None = None


async def run_case(
    case: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, live: bool
) -> None:
    provider = case["provider"]
    scratch = Scratch(tmp_path, case.get("options", {}).get("cwd", "work"))
    isolate(monkeypatch, scratch, live=live)
    setup = case.get("setup", {})
    prepare(setup, scratch)
    async with _mock(provider, case.get("mock", [{"text": "ok"}]), monkeypatch, live) as api:
        ctx = Context(provider, scratch, api, setup)
        async with ctx.clients:
            await _runs(case, ctx, live)


@contextlib.asynccontextmanager
async def _mock(
    provider: str, steps: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch, live: bool
) -> AsyncIterator[MockApi | None]:
    if live:
        yield None
        return
    api = (MockClaude if provider == "anthropic" else MockCodex)(steps).start()
    if provider == "anthropic":
        monkeypatch.setenv("ANTHROPIC_BASE_URL", api.base_url)
        monkeypatch.setenv("CLAUDE_CODE_MAX_RETRIES", "0")
    monkeypatch.setenv(
        "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY", MOCK_KEY[provider]
    )
    try:
        yield api
    finally:
        api.stop()


async def _runs(case: dict[str, Any], ctx: Context, live: bool) -> None:
    base = {"provider": ctx.provider, "cwd": "work", "provider_options": {}}
    base = _merged(base, case.get("options", {}))
    agent: Agent | None = None
    runs = [
        {
            "prompt": case["prompt"],
            "options": case.get("run_options", {}),
            "expect": case["expect"],
        },
        *case.get("runs", []),
    ]
    for n, run in enumerate(runs):
        new_agent = n == 0 or run.get("agent") == "new"
        agent_options, run_options = base, run.get("options", {})
        if new_agent and n > 0:
            agent_options, run_options = _merged(base, run_options), {}
        first_request = len(ctx.api.requests) if ctx.api else 0
        ctx.events.clear()
        outcome = Outcome()
        try:
            if new_agent:
                agent = Agent(**await ctx.native(agent_options))
            assert agent is not None
            overrides = await ctx.native(run_options)
            if live and "artifacts_dir" not in agent_options | run_options:
                overrides["artifacts_dir"] = LIVE_ARTIFACTS / case["id"] / f"run-{n}"
            outcome.result = await asyncio.wait_for(
                agent.run(run["prompt"], **overrides), RUN_LIMIT_S
            )
        except ConfigError as exc:
            outcome.config_error = exc
        except RunFailedError as exc:
            outcome.raised = exc
            outcome.result = exc.result
        requests = ctx.api.requests[first_request:] if ctx.api else []
        check(run["expect"], outcome, requests, ctx, options=agent_options | run_options, live=live)
        if outcome.result is not None:
            ctx.session_id = outcome.result.session_id


def _merged(base: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
    merged = {**base, **options}
    if "provider_options" in base and "provider_options" in options:
        merged["provider_options"] = {**base["provider_options"], **options["provider_options"]}
    return merged


def check(
    expect: dict[str, Any],
    outcome: Outcome,
    requests: list[dict[str, Any]],
    ctx: Context,
    *,
    options: dict[str, Any],
    live: bool,
) -> None:
    """Assert a run's expectations; ``options`` are the run's JSON options."""

    if expect.get("config_error"):
        assert outcome.config_error is not None, "expected ConfigError"
        assert requests == []
        return
    if outcome.config_error is not None:
        raise outcome.config_error
    raised = outcome.raised is not None
    assert raised == (expect.get("raises") == "RunFailedError"), outcome.raised
    assert outcome.result is not None
    _check_result(expect, outcome.result, ctx)
    _check_outputs(expect, outcome.result, ctx, options)
    if "setup_error" in expect:
        assert requests == []
    if not live:
        wanted = expect.get("requests", {})
        if "count" in wanted:
            assert len(requests) == wanted["count"], [r["body"] for r in requests]
        for match in wanted.get("match", []):
            _check_match(match, requests)


def _check_result(expect: dict[str, Any], result: RunResult, ctx: Context) -> None:
    events = [envelope.event for envelope in result.events]
    types = [event.type for event in events]
    detail = f"{result.status.value} {result.error_type}: {result.error}; events {types}"
    if "setup_error" in expect:
        assert (result.status.value, result.error_type) == ("failure", expect["setup_error"]), (
            detail
        )
    for name in ("status", "error_type", "final_text"):
        if name in expect:
            actual = getattr(result, name)
            assert getattr(actual, "value", actual) == expect[name], detail
    if "final_text_contains" in expect:
        assert expect["final_text_contains"] in result.final_text, result.final_text
    matcher = expect.get("events", {})
    assert set(matcher.get("includes", [])) <= set(types), detail
    assert not set(matcher.get("excludes", [])) & set(types), detail
    if "count" in matcher:
        assert len(types) == matcher["count"], types
    calls = [call.name for call in result.tool_calls()]
    wanted_calls = expect.get("tool_calls", {})
    if isinstance(wanted_calls, list):
        assert calls == wanted_calls, detail
    else:
        assert set(wanted_calls.get("includes", [])) <= set(calls), detail
        assert not set(wanted_calls.get("excludes", [])) & set(calls), detail
    if "tool_results" in expect:
        results = [event for event in events if event.type == "tool_result"]
        assert len(results) == len(expect["tool_results"]), results
        for actual, wanted in zip(results, expect["tool_results"], strict=True):
            assert wanted["contains"] in (actual.output or ""), actual
            assert actual.is_error == wanted.get("is_error", False), actual
    if "structured_output" in expect:
        value = result.structured_output
        value = value.model_dump() if isinstance(value, BaseModel) else value
        assert value == expect["structured_output"], detail
    if "same_session" in expect:
        assert (result.session_id == ctx.session_id) == expect["same_session"], result.session_id
    for name in expect.get("raw", []):
        raw = [event.raw for event in events if event.type == name]
        assert raw and all(item is not None for item in raw), (name, raw)


def _check_outputs(
    expect: dict[str, Any], result: RunResult, ctx: Context, options: dict[str, Any]
) -> None:
    """The callbacks, trace and artifacts a run wrote."""

    kept = [(envelope.sequence, envelope.event.type) for envelope in result.events]
    if expect.get("on_event"):
        _check_envelopes([(e.sequence, e.event.type) for e in ctx.events], kept)
    if expect.get("trace_file"):
        path = options.get("trace_file") or Path(options["artifacts_dir"]) / "trace.jsonl"
        lines = ctx.path(str(path)).read_text(encoding="utf-8").splitlines()
        traced = [json.loads(line) for line in lines]
        _check_envelopes([(t["sequence"], t["event"]["type"]) for t in traced], kept)
    if expect.get("on_provider_event"):
        assert ctx.provider_events, "no native events"
    for name in expect.get("artifacts", []):
        assert (ctx.path(options["artifacts_dir"]) / name).is_file(), name


def _check_envelopes(seen: list[tuple[int, str]], result: list[tuple[int, str]]) -> None:
    assert [seq for seq, _ in seen] == list(range(len(seen))), seen
    assert seen[0][1] == "run_started" and seen[-1][1] == "run_finished", seen
    if result:
        assert seen == result


_MISSING = object()


def _resolve(value: Any, path: str) -> Any:
    for part in path.split(".") if path else []:
        if isinstance(value, list):
            try:
                value = value[int(part)]
            except (ValueError, IndexError):
                return _MISSING
        elif isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return _MISSING
    return value


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _matches(match: dict[str, Any], request: dict[str, Any]) -> bool:
    if "header" in match:
        value = request["headers"].get(match["header"], _MISSING)
    else:
        value = _resolve(request["body"], match["path"])
    if match.get("absent"):
        return value is _MISSING
    if "excludes" in match:
        return value is _MISSING or match["excludes"] not in _text(value)
    if value is _MISSING:
        return False
    if "equals" in match:
        return value == match["equals"]
    return match["contains"] in _text(value)


def _check_match(match: dict[str, Any], requests: list[dict[str, Any]]) -> None:
    if "request" in match:
        index = match["request"]
        assert -len(requests) <= index < len(requests), (match, len(requests))
        assert _matches(match, requests[index]), (match, requests[index])
        return
    if match.get("absent") or "excludes" in match:
        assert all(_matches(match, request) for request in requests), match
    else:
        assert any(_matches(match, request) for request in requests), (match, requests)
