"""OpenAI Codex provider event mapping."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, TypedDict

import pytest
from pydantic import BaseModel, Field

from agent_sdk_wrapper import (
    AgentSdkWrapperError,
    ConfigError,
    Error,
    McpHttpServer,
    McpStdioServer,
    RunRequest,
    StructuredOutput,
    SubagentDef,
    Text,
    Thinking,
    TokenUsage,
    ToolCall,
    ToolResult,
    Usage,
    WarningEvent,
)
from agent_sdk_wrapper.providers.openai_provider import (
    OpenAIProvider,
    _codex_config,
    _codex_env,
    _codex_output_schema,
    _runtime_config,
    _stream_turn,
    _validate_supported,
    _write_sdk_debug_log,
)
from agent_sdk_wrapper.tools import TOOL_NAME_ATTR


class Answer(BaseModel):
    ok: bool


def sample_importable_tool(value: int) -> int:
    """Return the value unchanged."""

    return value


class FakeCollabAgentState:
    def model_dump(self, *, mode: str = "python", by_alias: bool = False):  # noqa: ARG002
        return {"status": "completed"}


class FakeTurn:
    def __init__(self, events):
        self._events = events

    async def stream(self):
        for event in self._events:
            yield event


def turn_completed(status: str = "completed", error: Any = None) -> SimpleNamespace:
    return SimpleNamespace(
        method="turn/completed",
        payload=SimpleNamespace(turn=SimpleNamespace(status=status, error=error)),
    )


def notification(method: str, payload: dict[str, Any]) -> Any:
    """Build a real SDK notification from its wire payload."""

    from openai_codex.generated.notification_registry import NOTIFICATION_MODELS
    from openai_codex.models import Notification

    return Notification(method=method, payload=NOTIFICATION_MODELS[method].model_validate(payload))


def failed_turn(error: dict[str, Any]) -> Any:
    return notification(
        "turn/completed",
        {
            "threadId": "t",
            "turn": {"id": "u", "items": [], "status": "failed", "error": error},
        },
    )


def test_codex_options_default_to_auto_reasoning_summary():
    req = RunRequest(provider="openai", prompt="ignored", effort="high")

    _, turn_options = OpenAIProvider()._build_options(req, None, None)

    assert turn_options["effort"] == "high"
    assert turn_options["summary"] == "auto"


def test_codex_options_pass_native_effort_through():
    req = RunRequest(provider="openai", prompt="ignored")

    _, turn_options = OpenAIProvider(effort="max")._build_options(req, None, None)

    assert turn_options["effort"] == "max"


def test_codex_efforts_match_the_sdk_enum():
    from openai_codex.generated.v2_all import ReasoningEffort

    from agent_sdk_wrapper.request import _OPENAI_EFFORTS

    assert _OPENAI_EFFORTS == {member.value for member in ReasoningEffort}


def test_codex_api_key_requires_a_wrapper_launched_runtime(monkeypatch):
    req = RunRequest(provider="openai", prompt="ignored")
    custom_launch = {"launch_args_override": ("codex", "app-server")}

    with pytest.raises(ConfigError, match="api_key"):
        OpenAIProvider(api_key="sk-test", codex=object()).validate_request(req)
    with pytest.raises(ConfigError, match="api_key"):
        OpenAIProvider(api_key="sk-test", config=custom_launch).validate_request(req)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    plain = RunRequest(provider="openai", prompt="x")
    assert OpenAIProvider(codex=object())._login_api_key(plain) is None
    assert OpenAIProvider(config=custom_launch)._login_api_key(plain) is None
    assert OpenAIProvider()._login_api_key(plain) == "sk-env"


def test_codex_sandbox_is_a_thread_mode_not_a_turn_policy():
    from openai_codex import Sandbox

    req = RunRequest(provider="openai", prompt="ignored")

    thread_options, turn_options = OpenAIProvider()._build_options(
        req, None, Sandbox.workspace_write
    )

    assert thread_options["sandbox"] is Sandbox.workspace_write
    assert "sandbox" not in turn_options


def test_codex_options_allow_summary_constructor_override():
    req = RunRequest(provider="openai", prompt="ignored")

    _, turn_options = OpenAIProvider(summary="none")._build_options(req, None, None)

    assert turn_options["summary"] == "none"


def test_codex_options_allow_turn_summary_override_and_disable():
    detailed = RunRequest(
        provider="openai",
        prompt="ignored",
        extra_options={"turn_options": {"summary": "detailed"}},
    )
    disabled = RunRequest(
        provider="openai",
        prompt="ignored",
        extra_options={"turn_options": {"summary": None}},
    )

    _, detailed_options = OpenAIProvider()._build_options(detailed, None, None)
    _, disabled_options = OpenAIProvider()._build_options(disabled, None, None)

    assert detailed_options["summary"] == "detailed"
    assert "summary" not in disabled_options


@pytest.mark.asyncio
async def test_codex_stream_maps_text_usage_and_structured_output():
    req = RunRequest(
        provider="openai",
        prompt="ignored",
        output_schema=Answer,
    )
    events = [
        SimpleNamespace(
            method="item/agentMessage/delta",
            payload=SimpleNamespace(delta='{"ok":'),
        ),
        SimpleNamespace(
            method="item/agentMessage/delta",
            payload=SimpleNamespace(delta="true}"),
        ),
        SimpleNamespace(
            method="item/completed",
            payload=SimpleNamespace(
                item=SimpleNamespace(
                    root=SimpleNamespace(type="agentMessage", text='{"ok":true}')
                )
            ),
        ),
        SimpleNamespace(
            method="thread/tokenUsage/updated",
            payload=SimpleNamespace(
                token_usage={
                    "total": {
                        "inputTokens": 10,
                        "outputTokens": 3,
                        "totalTokens": 13,
                        "cachedInputTokens": 2,
                        "reasoningOutputTokens": 4,
                    }
                }
            ),
        ),
        SimpleNamespace(
            method="turn/completed",
            payload=SimpleNamespace(turn=SimpleNamespace(status="completed")),
        ),
    ]

    out = [event async for event in _stream_turn(FakeTurn(events), req)]

    assert [type(event) for event in out] == [
        Text,
        Usage,
        StructuredOutput,
    ]
    assert [event.text for event in out if isinstance(event, Text)] == [
        '{"ok":true}'
    ]
    usage = next(event for event in out if isinstance(event, Usage))
    assert usage.usage == TokenUsage(
        input_tokens=10,
        output_tokens=3,
        total_tokens=13,
        cache_read_tokens=2,
        reasoning_output_tokens=4,
        requests=1,
    )
    structured = next(event for event in out if isinstance(event, StructuredOutput))
    assert structured.value == Answer(ok=True)


@pytest.mark.asyncio
async def test_codex_stream_writes_provider_events_sidecar(tmp_path):
    provider_events = []
    req = RunRequest(
        provider="openai",
        prompt="ignored",
        artifacts_dir=tmp_path,
        on_provider_event=provider_events.append,
    )
    events = [
        SimpleNamespace(
            method="item/agentMessage/delta",
            payload=SimpleNamespace(item_id="msg-1", delta="hello"),
        ),
        SimpleNamespace(
            method="item/completed",
            payload=SimpleNamespace(
                item=SimpleNamespace(
                    root=SimpleNamespace(type="agentMessage", id="msg-1", text="hello")
                )
            ),
        ),
    ]

    out = [event async for event in _stream_turn(FakeTurn([*events, turn_completed()]), req)]

    assert out == [Text(text="hello")]
    path = tmp_path / "provider-events.jsonl"
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [line["sequence"] for line in lines] == [0, 1, 2]
    assert [line["provider"] for line in lines] == ["openai", "openai", "openai"]
    assert lines[0]["class"] == "types.SimpleNamespace"
    assert lines[0]["message"]["method"] == "item/agentMessage/delta"
    assert lines[0]["message"]["payload"]["delta"] == "hello"
    assert [event.to_dict() for event in provider_events] == lines
    assert not (tmp_path / "sdk").exists()


@pytest.mark.asyncio
async def test_codex_stream_buffers_text_deltas_until_completed_message():
    req = RunRequest(provider="openai", prompt="ignored")
    events = [
        SimpleNamespace(
            method="item/agentMessage/delta",
            payload=SimpleNamespace(item_id="msg-1", delta="partial"),
        ),
        SimpleNamespace(
            method="item/completed",
            payload=SimpleNamespace(
                item=SimpleNamespace(
                    root=SimpleNamespace(type="agentMessage", id="msg-1", text="complete")
                )
            ),
        ),
    ]

    out = [event async for event in _stream_turn(FakeTurn([*events, turn_completed()]), req)]

    assert out == [Text(text="complete")]


@pytest.mark.asyncio
async def test_codex_stream_emits_buffered_text_if_completion_has_no_text():
    req = RunRequest(provider="openai", prompt="ignored")
    events = [
        SimpleNamespace(
            method="item/agentMessage/delta",
            payload=SimpleNamespace(item_id="msg-1", delta="hello "),
        ),
        SimpleNamespace(
            method="item/agentMessage/delta",
            payload=SimpleNamespace(item_id="msg-1", delta="world"),
        ),
        SimpleNamespace(
            method="item/completed",
            payload=SimpleNamespace(
                item=SimpleNamespace(
                    root=SimpleNamespace(type="agentMessage", id="msg-1", text="")
                )
            ),
        ),
    ]

    out = [event async for event in _stream_turn(FakeTurn([*events, turn_completed()]), req)]

    assert out == [Text(text="hello world")]


@pytest.mark.asyncio
async def test_codex_stream_drains_uncompleted_text_on_turn_completed():
    req = RunRequest(provider="openai", prompt="ignored")
    events = [
        SimpleNamespace(
            method="item/agentMessage/delta",
            payload=SimpleNamespace(item_id="msg-1", delta="orphaned"),
        ),
        SimpleNamespace(
            method="turn/completed",
            payload=SimpleNamespace(turn=SimpleNamespace(status="completed")),
        ),
    ]

    out = [event async for event in _stream_turn(FakeTurn(events), req)]

    assert out == [Text(text="orphaned")]


@pytest.mark.asyncio
async def test_codex_structured_output_validation_failure_is_an_error():
    req = RunRequest(
        provider="openai",
        prompt="ignored",
        output_schema=Answer,
    )
    events = [
        SimpleNamespace(
            method="item/completed",
            payload=SimpleNamespace(
                item=SimpleNamespace(root=SimpleNamespace(type="agentMessage", text="{}"))
            ),
        ),
        turn_completed(),
    ]

    out = [event async for event in _stream_turn(FakeTurn(events), req)]

    assert isinstance(out[-1], Error)
    assert out[-1].error_type == "structured_output_failed"
    assert "structured output did not match" in out[-1].message


@pytest.mark.asyncio
async def test_codex_stream_without_turn_completed_is_a_protocol_error():
    req = RunRequest(provider="openai", prompt="ignored")
    events = [
        SimpleNamespace(
            method="item/completed",
            payload=SimpleNamespace(
                item=SimpleNamespace(root=SimpleNamespace(type="agentMessage", text="partial"))
            ),
        ),
    ]

    out = [event async for event in _stream_turn(FakeTurn(events), req)]

    assert [type(event) for event in out] == [Text, Error]
    assert out[-1].error_type == "provider_protocol_error"


@pytest.mark.asyncio
async def test_codex_stream_maps_command_tool_result():
    req = RunRequest(provider="openai", prompt="ignored")
    events = [
        SimpleNamespace(
            method="item/completed",
            payload=SimpleNamespace(
                item=SimpleNamespace(
                    root=SimpleNamespace(
                        type="commandExecution",
                        id="cmd-1",
                        command="pytest",
                        status="completed",
                        aggregated_output="passed",
                    )
                )
            ),
        )
    ]

    out = [event async for event in _stream_turn(FakeTurn([*events, turn_completed()]), req)]

    assert out == [
        ToolCall(id="cmd-1", name="command", input={"command": "pytest"}),
        ToolResult(id="cmd-1", name="command", output="passed"),
    ]


@pytest.mark.asyncio
async def test_codex_stream_maps_reasoning_deltas_and_completed_items():
    req = RunRequest(provider="openai", prompt="ignored")
    events = [
        SimpleNamespace(
            method="item/reasoning/textDelta",
            payload=SimpleNamespace(delta="thinking"),
        ),
        SimpleNamespace(
            method="item/reasoning/summaryTextDelta",
            payload=SimpleNamespace(delta="summary delta"),
        ),
        SimpleNamespace(
            method="item/completed",
            payload=SimpleNamespace(
                item=SimpleNamespace(
                    root=SimpleNamespace(
                        type="reasoning",
                        summary=["summary"],
                        content=["detail"],
                    )
                )
            ),
        ),
        SimpleNamespace(
            method="item/completed",
            payload=SimpleNamespace(
                item=SimpleNamespace(root=SimpleNamespace(type="plan", text="plan"))
            ),
        ),
    ]

    out = [event async for event in _stream_turn(FakeTurn([*events, turn_completed()]), req)]

    assert out == [
        Thinking(text="summary\ndetail"),
        Thinking(text="plan"),
    ]


@pytest.mark.asyncio
async def test_codex_stream_buffers_reasoning_deltas_until_completed_item():
    req = RunRequest(provider="openai", prompt="ignored")
    events = [
        SimpleNamespace(
            method="item/reasoning/summaryTextDelta",
            payload=SimpleNamespace(item_id="reason-1", delta="partial"),
        ),
        SimpleNamespace(
            method="item/completed",
            payload=SimpleNamespace(
                item=SimpleNamespace(
                    root=SimpleNamespace(
                        type="reasoning",
                        id="reason-1",
                        summary=["complete"],
                        content=[],
                    )
                )
            ),
        ),
    ]

    out = [event async for event in _stream_turn(FakeTurn([*events, turn_completed()]), req)]

    assert out == [Thinking(text="complete")]


@pytest.mark.asyncio
async def test_codex_stream_drains_uncompleted_reasoning_on_turn_completed():
    req = RunRequest(provider="openai", prompt="ignored")
    events = [
        SimpleNamespace(
            method="item/reasoning/summaryTextDelta",
            payload=SimpleNamespace(item_id="reason-1", delta="orphaned"),
        ),
        SimpleNamespace(
            method="turn/completed",
            payload=SimpleNamespace(turn=SimpleNamespace(status="completed")),
        ),
    ]

    out = [event async for event in _stream_turn(FakeTurn(events), req)]

    assert out == [Thinking(text="orphaned")]


@pytest.mark.asyncio
async def test_codex_stream_maps_more_tool_like_items():
    req = RunRequest(provider="openai", prompt="ignored")
    events = [
        SimpleNamespace(
            method="item/completed",
            payload=SimpleNamespace(
                item=SimpleNamespace(
                    root=SimpleNamespace(
                        type="fileChange",
                        id="patch-1",
                        status="completed",
                        changes=[{"path": "a.py", "diff": "@@"}],
                    )
                )
            ),
        ),
        SimpleNamespace(
            method="item/completed",
            payload=SimpleNamespace(
                item=SimpleNamespace(
                    root=SimpleNamespace(
                        type="webSearch",
                        id="web-1",
                        query="codex sdk",
                        action={"type": "search", "query": "codex sdk"},
                    )
                )
            ),
        ),
        SimpleNamespace(
            method="item/completed",
            payload=SimpleNamespace(
                item=SimpleNamespace(
                    root=SimpleNamespace(
                        type="collabAgentToolCall",
                        id="agent-1",
                        tool="spawnAgent",
                        status="completed",
                        prompt="review",
                        model="gpt-5",
                        receiver_thread_ids=["thr-2"],
                        agents_states={"thr-2": FakeCollabAgentState()},
                    )
                )
            ),
        ),
    ]

    out = [event async for event in _stream_turn(FakeTurn([*events, turn_completed()]), req)]

    assert [event.name for event in out if isinstance(event, ToolCall)] == [
        "file_change",
        "web_search",
        "agent.spawnAgent",
    ]
    assert [event.is_error for event in out if isinstance(event, ToolResult)] == [
        False,
        False,
        False,
    ]
    agent_result = [event for event in out if isinstance(event, ToolResult)][-1]
    assert '"status":"completed"' in (agent_result.output or "")


def test_codex_rejects_positional_only_tool_parameters():
    def scale(value: int, /) -> int:
        return value * 2

    req = RunRequest(provider="openai", prompt="ignored", tools=[scale])

    with pytest.raises(ConfigError, match="positional-only"):
        OpenAIProvider().validate_request(req)


MODULE_OFFSET = 3


def _register(fn):
    return fn


class _Tools:
    def method(self, value: int) -> int:
        return value


def _source_fallback_candidates():
    offset = 1

    def uses_global(value: int) -> int:
        return value + MODULE_OFFSET

    def uses_closure(value: int) -> int:
        return value + offset

    @_register
    def decorated(value: int) -> int:
        return value

    def annotated(value: Literal["a", "b"]) -> str:
        return value

    identity = lambda value: value  # noqa: E731
    setattr(identity, TOOL_NAME_ATTR, "identity")

    return {
        "module-level names MODULE_OFFSET": uses_global,
        "closes over offset": uses_closure,
        "decorated": decorated,
        "module-level names Literal": annotated,
        "bound method": _Tools().method,
        "not a plain function definition": identity,
    }


@pytest.mark.parametrize("problem", list(_source_fallback_candidates()))
def test_codex_rejects_tools_the_server_cannot_rebuild_from_source(problem):
    tool = _source_fallback_candidates()[problem]
    req = RunRequest(provider="openai", prompt="ignored", tools=[tool])

    with pytest.raises(ConfigError, match=problem):
        OpenAIProvider().validate_request(req)


def test_codex_accepts_self_contained_local_tools():
    def scale(value: float, factor: int = 2, label: str | None = None) -> dict[str, float]:
        import math

        return {label or "value": math.fsum([value] * factor)}

    OpenAIProvider().validate_request(RunRequest(provider="openai", prompt="x", tools=[scale]))


def test_codex_tool_server_gets_env_names_and_a_long_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("PARENT_ONLY_TOKEN", "parent-secret")
    req = RunRequest(
        provider="openai",
        prompt="ignored",
        tools=[sample_importable_tool],
        env={"REQUEST_TOKEN": "request-secret"},
        cwd=tmp_path,
    )

    with _runtime_config(req) as runtime:
        overrides = runtime.config_overrides
        [env_vars] = [v for v in overrides if v.startswith(f"{WRAPPER_SERVER}.env_vars=")]
        assert '"PARENT_ONLY_TOKEN"' in env_vars
        assert '"REQUEST_TOKEN"' in env_vars
        assert "secret" not in " ".join(overrides)
        assert f"{WRAPPER_SERVER}.tool_timeout_sec=600" in overrides


WRAPPER_SERVER = "mcp_servers.agent_sdk_wrapper_tools"


@pytest.mark.asyncio
async def test_codex_interrupt_the_wrapper_did_not_request_is_cancelled():
    req = RunRequest(provider="openai", prompt="ignored")

    out = [event async for event in _stream_turn(FakeTurn([turn_completed("interrupted")]), req)]

    assert [(type(event), event.error_type) for event in out] == [(Error, "cancelled")]


@pytest.mark.asyncio
async def test_codex_stream_maps_failed_tool_like_items():
    req = RunRequest(provider="openai", prompt="ignored")
    events = [
        SimpleNamespace(
            method="item/completed",
            payload=SimpleNamespace(
                item=SimpleNamespace(
                    root=SimpleNamespace(
                        type="commandExecution",
                        command="rm -rf /tmp/nope",
                        status="declined",
                    )
                )
            ),
        ),
        SimpleNamespace(
            method="item/completed",
            payload=SimpleNamespace(
                item=SimpleNamespace(
                    root=SimpleNamespace(
                        type="mcpToolCall",
                        server="repo",
                        tool="read_file",
                        arguments={"path": "missing.py"},
                        status="failed",
                        error=SimpleNamespace(message="File not found"),
                    )
                )
            ),
        ),
        SimpleNamespace(
            method="item/completed",
            payload=SimpleNamespace(
                item=SimpleNamespace(
                    root=SimpleNamespace(
                        type="dynamicToolCall",
                        namespace="dynamic",
                        tool="lookup",
                        arguments='{"q":"agent-sdk-wrapper"}',
                        status="failed",
                        success=False,
                        content_items=[{"type": "text", "text": "lookup failed"}],
                    )
                )
            ),
        ),
    ]

    out = [event async for event in _stream_turn(FakeTurn([*events, turn_completed()]), req)]

    assert [event.name for event in out if isinstance(event, ToolCall)] == [
        "command",
        "repo.read_file",
        "dynamic.lookup",
    ]
    results = [event for event in out if isinstance(event, ToolResult)]
    assert [event.is_error for event in results] == [True, True, True]
    assert [event.id for event in results] == [None, None, None]
    assert results[0].output == "declined"
    assert results[1].output == "File not found"


@pytest.mark.asyncio
async def test_codex_stream_maps_image_items():
    req = RunRequest(provider="openai", prompt="ignored")
    events = [
        SimpleNamespace(
            method="item/completed",
            payload=SimpleNamespace(
                item=SimpleNamespace(
                    root=SimpleNamespace(
                        type="imageView",
                        id="image-1",
                        path="/tmp/screenshot.png",
                    )
                )
            ),
        ),
        SimpleNamespace(
            method="item/completed",
            payload=SimpleNamespace(
                item=SimpleNamespace(
                    root=SimpleNamespace(
                        type="imageGeneration",
                        id="image-2",
                        status="failed",
                        result=None,
                        saved_path=None,
                        revised_prompt="draw a test fixture",
                    )
                )
            ),
        ),
    ]

    out = [event async for event in _stream_turn(FakeTurn([*events, turn_completed()]), req)]

    assert [event.name for event in out if isinstance(event, ToolCall)] == [
        "view_image",
        "image_generation",
    ]
    results = [event for event in out if isinstance(event, ToolResult)]
    assert results[0].id == "image-1"
    assert results[0].output == "/tmp/screenshot.png"
    assert results[0].is_error is False
    assert results[1].id == "image-2"
    assert results[1].is_error is True
    assert '"status":"failed"' in (results[1].output or "")


def _options_request(**kwargs: Any) -> RunRequest:
    return RunRequest(provider="openai", prompt="ignored", **kwargs)


@pytest.mark.parametrize(
    ("provider_options", "request_options", "match"),
    [
        ({"thread_options": {"thread_source": "user"}}, {"session_id": "t"}, "thread_resume"),
        ({"thread_options": {"include_turns": True}}, {}, "thread_start options: include_turns"),
        ({"turn_options": {"output_format": "json"}}, {}, "turn options: output_format"),
        ({}, {"extra_options": {"thread": {}}}, "extra_options keys: thread"),
        ({"ephemeral": True}, {"session_id": "t"}, "ephemeral"),
        ({"ephemeral": True}, {"continue_session": True}, "ephemeral"),
        (
            {},
            {"continue_session": True, "extra_options": {"thread_options": {"ephemeral": True}}},
            "ephemeral",
        ),
        ({"ephemeral": True, "thread_id": "t"}, {}, "ephemeral"),
        ({"codex": object()}, {"web_tools": False}, "launch Codex"),
        ({"sandbox": "workspace"}, {}, "invalid Sandbox value"),
        ({"approval_mode": "sometimes"}, {}, "invalid ApprovalMode value"),
        (
            {"config": {"launch_args_override": ("codex",)}},
            {"tools": [sample_importable_tool]},
            "launch Codex",
        ),
    ],
)
def test_codex_rejects_options_the_sdk_cannot_take(provider_options, request_options, match):
    with pytest.raises(ConfigError, match=match):
        OpenAIProvider(**provider_options).validate_request(_options_request(**request_options))


def test_codex_accepts_sdk_native_options():
    provider = OpenAIProvider(
        ephemeral=False,
        thread_options={"base_instructions": "Be brief.", "service_tier": "flex"},
        turn_options={"turn_service_tier": "flex", "summary": "concise"},
    )

    provider.validate_request(_options_request())
    provider.validate_request(_options_request(session_id="t", continue_session=True))
    thread_options, _ = provider._build_options(_options_request(session_id="t"), None, None)
    assert "ephemeral" not in thread_options


def test_codex_filters_require_wrapper_managed_tools():
    req = RunRequest(provider="openai", prompt="ignored", allowed_tools=["Read"])

    with pytest.raises(ConfigError, match="without callable tools or MCP servers"):
        _validate_supported(req)


def test_codex_rejects_max_turns():
    req = RunRequest(provider="openai", prompt="ignored", max_turns=1)

    with pytest.raises(ConfigError, match="max_turns. Codex has no turn limit"):
        _validate_supported(req)


def test_codex_filters_reject_native_builtins_even_with_mcp_server():
    req = RunRequest(
        provider="openai",
        prompt="ignored",
        allowed_tools=["command"],
        mcp_servers=[McpStdioServer(name="repo", command="repo-mcp")],
    )

    with pytest.raises(ConfigError, match="Codex native tool filters: command"):
        _validate_supported(req)


def test_codex_filters_allow_qualified_tool_named_like_native_builtin():
    def command(value: str) -> str:
        return value

    req = RunRequest(
        provider="openai",
        prompt="ignored",
        tools=[command],
        allowed_tools=["agent_sdk_wrapper_tools.command", "repo.command"],
        mcp_servers=[
            McpStdioServer(
                name="repo",
                command="repo-mcp",
                enabled_tools=["command"],
            )
        ],
    )

    _validate_supported(req)


def test_codex_filters_reject_unknown_wrapper_server():
    req = RunRequest(
        provider="openai",
        prompt="ignored",
        allowed_tools=["missing.search"],
        mcp_servers=[McpStdioServer(name="repo", command="repo-mcp")],
    )

    with pytest.raises(ConfigError, match="non-wrapper tools: missing.search"):
        _validate_supported(req)


def test_codex_filters_reject_unknown_callable_tool_when_mcp_also_present():
    def add(a: int, b: int) -> int:
        return a + b

    req = RunRequest(
        provider="openai",
        prompt="ignored",
        tools=[add],
        allowed_tools=["agent_sdk_wrapper_tools.subtract"],
        mcp_servers=[McpStdioServer(name="repo", command="repo-mcp")],
    )

    with pytest.raises(ConfigError, match="non-wrapper tools: agent_sdk_wrapper_tools.subtract"):
        _validate_supported(req)


def test_codex_filters_reject_unknown_external_mcp_tool_when_known():
    req = RunRequest(
        provider="openai",
        prompt="ignored",
        allowed_tools=["repo.write_file"],
        mcp_servers=[
            McpStdioServer(
                name="repo",
                command="repo-mcp",
                enabled_tools=["read_file", "grep_files"],
            )
        ],
    )

    with pytest.raises(ConfigError, match="non-wrapper tools: repo.write_file"):
        _validate_supported(req)


def test_codex_filters_reject_unknown_unqualified_tool_when_all_tools_known():
    req = RunRequest(
        provider="openai",
        prompt="ignored",
        allowed_tools=["write_file"],
        mcp_servers=[
            McpStdioServer(
                name="repo",
                command="repo-mcp",
                enabled_tools=["read_file", "grep_files"],
            )
        ],
    )

    with pytest.raises(ConfigError, match="non-wrapper tools: write_file"):
        _validate_supported(req)


def test_write_sdk_debug_log(tmp_path):
    class SyncClient:
        def _stderr_tail(self, *, limit: int = 400) -> str:
            return f"tail:{limit}"

    codex = SimpleNamespace(_client=SimpleNamespace(_sync=SyncClient()))

    path = _write_sdk_debug_log(codex, tmp_path, debug=True)

    assert path == tmp_path / "sdk" / "openai-codex.debug.log"
    assert "tail:400" in path.read_text()


def test_write_sdk_debug_log_skips_without_debug(tmp_path):
    codex = SimpleNamespace()

    path = _write_sdk_debug_log(codex, tmp_path)

    assert path is None
    assert not (tmp_path / "sdk").exists()


def test_codex_env_sets_debug_only_when_enabled():
    assert _codex_env({}) == {}

    env = _codex_env({"RUST_LOG": "info"}, debug=True)

    assert env["RUST_LOG"] == "info"
    assert env["RUST_BACKTRACE"] == "1"


def test_codex_config_uses_path_codex_when_sdk_bin_missing(monkeypatch):
    from agent_sdk_wrapper.providers import openai_provider as op_mod

    monkeypatch.setattr(op_mod, "_path_codex_bin_when_sdk_bin_missing", lambda: "/usr/bin/codex")

    config = _codex_config(None, {}, None)

    assert config.codex_bin == "/usr/bin/codex"


def test_codex_config_lets_caller_overrides_win(monkeypatch):
    from agent_sdk_wrapper.providers import openai_provider as op_mod

    monkeypatch.setattr(op_mod, "_path_codex_bin_when_sdk_bin_missing", lambda: None)

    config = _codex_config(
        {"config_overrides": ("mcp_servers.agent_sdk_wrapper_tools.tool_timeout_sec=5",)},
        {},
        None,
        config_overrides=("mcp_servers.agent_sdk_wrapper_tools.tool_timeout_sec=600",),
    )

    # Codex applies overrides in order, so the caller's value is the effective one.
    assert config.config_overrides == (
        "mcp_servers.agent_sdk_wrapper_tools.tool_timeout_sec=600",
        "mcp_servers.agent_sdk_wrapper_tools.tool_timeout_sec=5",
    )


def test_codex_config_rejects_launch_args_with_generated_overrides(monkeypatch):
    from agent_sdk_wrapper.providers import openai_provider as op_mod

    monkeypatch.setattr(op_mod, "_path_codex_bin_when_sdk_bin_missing", lambda: None)

    with pytest.raises(ConfigError, match="launch_args_override"):
        _codex_config(
            {"launch_args_override": ("codex", "app-server", "--listen", "stdio://")},
            {},
            None,
            config_overrides=("features.multi_agent=true",),
        )


def test_runtime_config_builds_codex_tool_and_subagent_overrides(tmp_path):
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    req = RunRequest(
        provider="openai",
        prompt="ignored",
        tools=[add],
        subagents={
            "reviewer": SubagentDef(
                description="Reviews code.",
                prompt="Review tersely.",
                model="gpt-5",
            )
        },
        cwd=tmp_path,
    )

    with _runtime_config(req) as runtime:
        overrides = set(runtime.config_overrides)
        assert any(
            value.startswith("mcp_servers.agent_sdk_wrapper_tools.command=")
            for value in overrides
        )
        assert any(
            value.startswith("mcp_servers.agent_sdk_wrapper_tools.args=")
            for value in overrides
        )
        assert any(
            value.startswith("mcp_servers.agent_sdk_wrapper_tools.cwd=") and str(tmp_path) in value
            for value in overrides
        )
        assert "features.multi_agent=true" in overrides
        assert 'agents.reviewer.description="Reviews code."' in overrides
        assert any(value.startswith("agents.reviewer.config_file=") for value in overrides)
        assert runtime.warnings == ()


TRICKY_TEXT = 'fox \U0001f98a "quoted" \\ tab\t line\nDEL\x7f bell\x07 café'


def test_codex_config_values_are_valid_toml():
    import tomllib

    from agent_sdk_wrapper.providers.openai_provider import _toml_literal

    value = {"text": TRICKY_TEXT, "list": [TRICKY_TEXT, 1, 2.5, True], TRICKY_TEXT: "key"}

    assert tomllib.loads(f"x = {_toml_literal(value)}")["x"] == value
    with pytest.raises(ConfigError, match="surrogate"):
        _toml_literal("\ud83d")


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"subagents": {"two words": SubagentDef(description="d", prompt="p")}}, "key part"),
        ({"subagents": {"fox": SubagentDef(description="\ud83d", prompt="p")}}, "surrogate"),
        ({"mcp_servers": [McpStdioServer(name="repo", command="\ud83d")]}, "surrogate"),
    ],
)
def test_codex_rejects_unencodable_config_before_running(options, match):
    req = RunRequest(provider="openai", prompt="ignored", **options)

    with pytest.raises(ConfigError, match=match):
        OpenAIProvider().validate_request(req)


def test_codex_subagent_config_file_is_valid_toml(tmp_path):
    import tomllib

    req = RunRequest(
        provider="openai",
        prompt="ignored",
        subagents={"fox": SubagentDef(description=TRICKY_TEXT, prompt=TRICKY_TEXT)},
    )

    with _runtime_config(req) as runtime:
        [config_file] = [
            v.split("=", 1)[1] for v in runtime.config_overrides if ".config_file=" in v
        ]
        path = tomllib.loads(f"x = {config_file}")["x"]
        with open(path, "rb") as handle:
            assert tomllib.load(handle) == {"developer_instructions": TRICKY_TEXT}
        [description] = [
            v.split("=", 1)[1] for v in runtime.config_overrides if ".description=" in v
        ]
        assert tomllib.loads(f"x = {description}")["x"] == TRICKY_TEXT


def test_codex_rejects_builtin_tools():
    req = RunRequest(provider="openai", prompt="ignored", builtin_tools="none")

    with pytest.raises(ConfigError, match="builtin_tools"):
        _validate_supported(req)


def test_codex_web_tools_coexists_with_tools(tmp_path):
    req = RunRequest(
        provider="openai",
        prompt="ignored",
        tools=[sample_importable_tool],
        cwd=tmp_path,
        web_tools=False,
    )

    with _runtime_config(req) as runtime:
        assert 'web_search="disabled"' in runtime.config_overrides
        assert any(
            value.startswith("mcp_servers.agent_sdk_wrapper_tools.command=")
            for value in runtime.config_overrides
        )


def test_codex_rejects_subagent_tool_controls():
    req = RunRequest(
        provider="openai",
        prompt="ignored",
        subagents={
            "reviewer": SubagentDef(
                description="Reviews code.",
                prompt="Review tersely.",
                tools=["Read"],
                max_turns=2,
            )
        },
    )

    with pytest.raises(ConfigError, match="SubagentDef.tools"):
        _validate_supported(req)

    with pytest.raises(ConfigError, match="SubagentDef.tools"):
        with _runtime_config(req):
            pass


def test_runtime_config_builds_external_mcp_server_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("INHERITED_MODE", "parent")
    monkeypatch.setenv("OVERRIDE_MODE", "parent")

    req = RunRequest(
        provider="openai",
        prompt="ignored",
        mcp_servers=[
            McpStdioServer(
                name="auditor",
                command="uv",
                args=["run", "auditor-mcp"],
                cwd=tmp_path,
                env={"AUDITOR_MODE": "test", "OVERRIDE_MODE": "explicit"},
                env_passthrough=["INHERITED_MODE", "OVERRIDE_MODE", "MISSING_MODE"],
                enabled_tools=["review", "search"],
                disabled_tools=["delete"],
                default_tools_approval_mode="approve",
                tool_approval_modes={"review": "prompt"},
                required=True,
                startup_timeout_sec=5,
                tool_timeout_sec=30,
            ),
            McpHttpServer(
                name="remote",
                url="https://example.test/mcp",
                headers={"X-Test": "1"},
                env_http_headers={"Authorization": "REMOTE_TOKEN"},
                bearer_token_env_var="REMOTE_BEARER",
                disabled_tools=["expensive"],
            ),
        ],
    )

    with _runtime_config(req) as runtime:
        overrides = set(runtime.config_overrides)
        assert 'mcp_servers.auditor.command="uv"' in overrides
        assert 'mcp_servers.auditor.args=["run", "auditor-mcp"]' in overrides
        assert f'mcp_servers.auditor.cwd="{tmp_path}"' in overrides
        env_override = next(
            value for value in overrides if value.startswith("mcp_servers.auditor.env=")
        )
        assert '"INHERITED_MODE" = "parent"' in env_override
        assert '"OVERRIDE_MODE" = "explicit"' in env_override
        assert '"AUDITOR_MODE" = "test"' in env_override
        assert "MISSING_MODE" not in env_override
        assert 'mcp_servers.auditor.enabled_tools=["review", "search"]' in overrides
        assert 'mcp_servers.auditor.disabled_tools=["delete"]' in overrides
        assert 'mcp_servers.auditor.default_tools_approval_mode="approve"' in overrides
        assert 'mcp_servers.auditor.tools.review.approval_mode="prompt"' in overrides
        assert "mcp_servers.auditor.required=true" in overrides
        assert "mcp_servers.auditor.startup_timeout_sec=5" in overrides
        assert "mcp_servers.auditor.tool_timeout_sec=30" in overrides
        assert 'mcp_servers.remote.url="https://example.test/mcp"' in overrides
        assert 'mcp_servers.remote.http_headers={ "X-Test" = "1" }' in overrides
        assert (
            'mcp_servers.remote.env_http_headers={ "Authorization" = "REMOTE_TOKEN" }'
            in overrides
        )
        assert 'mcp_servers.remote.bearer_token_env_var="REMOTE_BEARER"' in overrides
        assert 'mcp_servers.remote.disabled_tools=["expensive"]' in overrides


def test_runtime_config_applies_codex_tool_filters(tmp_path):
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    def multiply(a: int, b: int) -> int:
        """Multiply two integers."""
        return a * b

    req = RunRequest(
        provider="openai",
        prompt="ignored",
        tools=[add, multiply],
        allowed_tools=["agent_sdk_wrapper_tools.add"],
        disallowed_tools=["mcp__agent_sdk_wrapper_tools__multiply"],
        cwd=tmp_path,
    )

    with _runtime_config(req) as runtime:
        overrides = set(runtime.config_overrides)
        assert 'mcp_servers.agent_sdk_wrapper_tools.enabled_tools=["add"]' in overrides
        assert 'mcp_servers.agent_sdk_wrapper_tools.disabled_tools=["multiply"]' in overrides


def test_runtime_config_targets_one_external_mcp_server_among_many():
    req = RunRequest(
        provider="openai",
        prompt="ignored",
        allowed_tools=["repo.read_file"],
        disallowed_tools=["bugs.search_bugs"],
        mcp_servers=[
            McpStdioServer(
                name="repo",
                command="repo-mcp",
                enabled_tools=["read_file", "grep_files"],
            ),
            McpStdioServer(
                name="bugs",
                command="bugs-mcp",
                enabled_tools=["search_bugs"],
            ),
        ],
    )

    with _runtime_config(req) as runtime:
        overrides = set(runtime.config_overrides)
        assert 'mcp_servers.repo.enabled_tools=["read_file"]' in overrides
        assert 'mcp_servers.bugs.enabled_tools=[]' in overrides
        assert 'mcp_servers.bugs.disabled_tools=["search_bugs"]' in overrides


class OptionalAnswer(BaseModel):
    a: int
    b: str | None = None


class Inner(BaseModel):
    x: int
    label: str = "none"


class Outer(BaseModel):
    inner: Inner = Field(description="The inner part.")
    items: list[Inner]
    count: int = 3


def test_codex_output_schema_is_strict():
    assert _codex_output_schema(OptionalAnswer) == {
        "properties": {
            "a": {"title": "A", "type": "integer"},
            "b": {"anyOf": [{"type": "string"}, {"type": "null"}], "title": "B"},
        },
        "required": ["a", "b"],
        "title": "OptionalAnswer",
        "type": "object",
        "additionalProperties": False,
    }


def test_codex_output_schema_inlines_described_refs_and_nulls_defaults():
    schema = _codex_output_schema(Outer)

    inner = schema["properties"]["inner"]
    assert "$ref" not in inner
    assert inner["description"] == "The inner part."
    assert inner["required"] == ["x", "label"]
    assert inner["additionalProperties"] is False
    assert inner["properties"]["label"] == {
        "anyOf": [{"title": "Label", "type": "string"}, {"type": "null"}]
    }
    assert schema["properties"]["items"]["items"] == {"$ref": "#/$defs/Inner"}
    assert schema["$defs"]["Inner"]["required"] == ["x", "label"]
    assert schema["required"] == ["inner", "items", "count"]
    assert "default" not in json.dumps(schema)
    assert "x-agent-sdk-wrapper" not in json.dumps(schema)


@pytest.mark.asyncio
async def test_codex_structured_output_restores_defaults_for_forced_nulls():
    req = RunRequest(provider="openai", prompt="ignored", output_schema=Outer)
    text = json.dumps(
        {
            "inner": {"x": 1, "label": None},
            "items": [{"x": 2, "label": "two"}, {"x": 3, "label": None}],
            "count": None,
        }
    )
    events = [
        SimpleNamespace(
            method="item/completed",
            payload=SimpleNamespace(
                item=SimpleNamespace(root=SimpleNamespace(type="agentMessage", text=text))
            ),
        ),
        SimpleNamespace(
            method="turn/completed",
            payload=SimpleNamespace(turn=SimpleNamespace(status="completed")),
        ),
    ]

    out = [event async for event in _stream_turn(FakeTurn(events), req)]

    structured = next(event for event in out if isinstance(event, StructuredOutput))
    assert structured.value == Outer(
        inner=Inner(x=1), items=[Inner(x=2, label="two"), Inner(x=3)], count=3
    )


@pytest.mark.parametrize(
    ("output_schema", "match"),
    [
        (list[int], "object schema"),
        (dict[str, int], "free-form"),
        (TypedDict("Loose", {"meta": dict[str, int]}), "free-form"),
        (TypedDict("Anything", {"value": Any}), "any value"),
    ],
)
def test_codex_rejects_output_schemas_strict_mode_cannot_express(output_schema, match):
    req = RunRequest(provider="openai", prompt="ignored", output_schema=output_schema)

    with pytest.raises(ConfigError, match=match):
        OpenAIProvider().validate_request(req)


CODEX_FIXTURES = Path(__file__).parent / "fixtures" / "codex"


def codex_frames(name: str) -> list[Any]:
    """Load redacted real provider events as SDK notification objects."""

    from openai_codex.generated.notification_registry import NOTIFICATION_MODELS
    from openai_codex.models import Notification

    frames = []
    for line in (CODEX_FIXTURES / name).read_text(encoding="utf-8").splitlines():
        message = json.loads(line)["message"]
        payload = NOTIFICATION_MODELS[message["method"]].model_validate(message["payload"])
        frames.append(Notification(method=message["method"], payload=payload))
    return frames


@pytest.mark.asyncio
async def test_codex_model_reroute_updates_the_session_model():
    from agent_sdk_wrapper import SessionInfo

    req = RunRequest(provider="openai", prompt="ignored")
    reroute = notification(
        "model/rerouted",
        {
            "fromModel": "gpt-5.4",
            "toModel": "gpt-5.4-safe",
            "reason": "highRiskCyberActivity",
            "threadId": "thread-1",
            "turnId": "turn-1",
        },
    )

    out = [event async for event in _stream_turn(FakeTurn([reroute, turn_completed()]), req)]

    assert [type(event) for event in out] == [WarningEvent, SessionInfo]
    assert "gpt-5.4-safe" in out[0].message
    assert out[1] == SessionInfo(id="thread-1", model="gpt-5.4-safe")


def test_codex_runtime_warnings_report_this_threads_mcp_failures():
    import queue

    from agent_sdk_wrapper.providers.openai_provider import _RuntimeWarnings

    def mcp_status(thread_id: str, status: str, error: str | None = None):
        payload = {"name": "broken", "status": status, "threadId": thread_id, "error": error}
        return notification("mcpServer/startupStatus/updated", payload)

    notifications = queue.Queue()
    for item in (
        mcp_status("thread-1", "starting"),
        mcp_status("thread-2", "failed", "other thread"),
        mcp_status("thread-1", "failed", "MCP client for `broken` failed to start"),
        notification("configWarning", {"summary": "Unknown key", "details": "x.y"}),
    ):
        notifications.put(item)
    router = SimpleNamespace(_global_notifications=notifications)
    codex = SimpleNamespace(_client=SimpleNamespace(_sync=SimpleNamespace(_router=router)))

    warnings = _RuntimeWarnings(codex, "thread-1", include_raw=False).drain()

    assert [w.message for w in warnings] == [
        "MCP client for `broken` failed to start",
        "Unknown key: x.y",
    ]
    assert notifications.empty()


@pytest.mark.asyncio
async def test_codex_tool_call_is_emitted_when_the_item_starts():
    req = RunRequest(provider="openai", prompt="ignored")
    frames = codex_frames("tool-turn.provider-events.jsonl")
    started, completed = frames[:2]
    commentary = SimpleNamespace(
        method="item/completed",
        payload=SimpleNamespace(
            item=SimpleNamespace(root=SimpleNamespace(type="agentMessage", text="working"))
        ),
    )

    out = [
        event
        async for event in _stream_turn(
            FakeTurn([started, commentary, completed, turn_completed()]), req
        )
    ]

    command = "/bin/zsh -lc \"python3 -c 'print((3 + 4) ** 2)'\""
    assert out == [
        ToolCall(id="exec-1", name="command", input={"command": command}),
        Text(text="working"),
        ToolResult(id="exec-1", name="command", output="49\n"),
    ]


@pytest.mark.asyncio
async def test_codex_usage_replays_a_multi_request_turn():
    req = RunRequest(provider="openai", prompt="ignored")
    turn = FakeTurn(codex_frames("tool-turn.provider-events.jsonl"))

    out = [event async for event in _stream_turn(turn, req)]

    [usage] = [event.usage for event in out if isinstance(event, Usage)]
    assert usage == TokenUsage(
        input_tokens=27707,
        output_tokens=109,
        total_tokens=27816,
        cache_read_tokens=13786,
        cache_write_tokens=13915,
        reasoning_output_tokens=22,
        requests=2,
    )


@pytest.mark.asyncio
async def test_codex_usage_of_a_resumed_turn_excludes_thread_history():
    req = RunRequest(provider="openai", prompt="ignored", session_id="thread-fixture")

    out = [
        event
        async for event in _stream_turn(
            FakeTurn(codex_frames("resumed-turn.provider-events.jsonl")), req
        )
    ]

    [usage] = [event.usage for event in out if isinstance(event, Usage)]
    assert usage == TokenUsage(
        input_tokens=12730,
        output_tokens=7,
        total_tokens=12737,
        cache_read_tokens=12672,
        requests=1,
    )


@pytest.mark.asyncio
async def test_codex_failed_turn_yields_one_classified_error():
    req = RunRequest(provider="openai", prompt="ignored")
    error = {"codexErrorInfo": "serverOverloaded", "message": "Selected model is at capacity."}
    events = [
        notification(
            "error",
            {
                "error": {"message": "Reconnecting... 1/5", "codexErrorInfo": "other"},
                "threadId": "t",
                "turnId": "u",
                "willRetry": True,
            },
        ),
        notification(
            "error", {"error": error, "threadId": "t", "turnId": "u", "willRetry": False}
        ),
        failed_turn(error),
    ]

    out = [event async for event in _stream_turn(FakeTurn(events), req)]

    assert [type(event) for event in out] == [WarningEvent, Error]
    assert out[0].message == "Reconnecting... 1/5"
    assert out[1].message == "Selected model is at capacity."
    assert out[1].error_type == "transient_api_error"


@pytest.mark.asyncio
async def test_codex_error_text_survives_empty_messages_and_unparsed_payloads():
    from openai_codex.models import Notification, UnknownNotification

    req = RunRequest(provider="openai", prompt="ignored")
    probe = "Offline gateway probe ✓"
    unparsed = Notification(
        method="error",
        payload=UnknownNotification(
            params={"error": {"message": probe, "codexErrorInfo": "newKind"}, "willRetry": False}
        ),
    )
    only_details = failed_turn(
        {"codexErrorInfo": "other", "message": "", "additionalDetails": probe}
    )

    first = [e async for e in _stream_turn(FakeTurn([unparsed, turn_completed("failed")]), req)]
    second = [e async for e in _stream_turn(FakeTurn([only_details]), req)]

    assert [(type(e), e.message) for e in first] == [(Error, probe)]
    assert [(type(e), e.message) for e in second] == [(Error, probe)]


@pytest.mark.parametrize(
    ("message", "transient"),
    [
        ("Connection reset by peer", True),
        ("request timed out", True),
        ("unexpected status 502 Bad Gateway", True),
        ("wrote 1500 tokens", False),
        ("tool_timeout_sec must be positive", False),
        ("unexpected status 400 Bad Request", False),
    ],
)
def test_codex_transient_exception_patterns_are_word_bounded(message, transient):
    from agent_sdk_wrapper.providers.openai_provider import _looks_transient

    assert _looks_transient(RuntimeError(message)) is transient


# Payloads captured from Codex 0.154 against a mock Responses API.
@pytest.mark.parametrize(
    ("error", "error_type"),
    [
        (
            {
                "codexErrorInfo": "other",
                "message": "unexpected status 401 Unauthorized: Incorrect API key provided: "
                "sk-x., url: http://127.0.0.1:9/v1/responses",
            },
            "authentication_failed",
        ),
        (
            {
                "codexErrorInfo": "other",
                "message": "unexpected status 403 Forbidden: You are not allowed to sample "
                "from this model, url: http://127.0.0.1:9/v1/responses",
            },
            "permission_denied",
        ),
        (
            {
                "codexErrorInfo": "other",
                "message": "unexpected status 404 Not Found: The model `gpt-5.9` does not "
                "exist or you do not have access to it., url: http://127.0.0.1:9/v1/responses",
            },
            "model_not_found",
        ),
        (
            {
                "codexErrorInfo": "other",
                "message": '{"error": {"message": "Invalid value for \'input\'.", '
                '"type": "invalid_request_error", "code": null}}',
            },
            "invalid_request",
        ),
        (
            {
                "codexErrorInfo": "other",
                "message": '{"error": {"message": "Your input exceeds the context window of '
                'this model.", "type": "invalid_request_error", '
                '"code": "context_length_exceeded"}}',
            },
            "context_window_exceeded",
        ),
        (
            {
                "codexErrorInfo": "other",
                "message": "unexpected status 503 Service Unavailable: The engine is currently "
                "overloaded, please try again later., url: http://127.0.0.1:9/v1/responses",
            },
            "transient_api_error",
        ),
        (
            {
                "codexErrorInfo": "other",
                "message": "stream disconnected before completion: stream closed before "
                "response.completed",
            },
            "transient_api_error",
        ),
        (
            {
                "codexErrorInfo": {"responseTooManyFailedAttempts": {"httpStatusCode": 429}},
                "message": "exceeded retry limit, last status: 429 Too Many Requests",
            },
            "transient_api_error",
        ),
        (
            {"codexErrorInfo": "internalServerError", "message": "We're currently experiencing"},
            "transient_api_error",
        ),
        (
            {"codexErrorInfo": "contextWindowExceeded", "message": "Codex ran out of room"},
            "context_window_exceeded",
        ),
        (
            {"codexErrorInfo": "usageLimitExceeded", "message": "Quota exceeded."},
            "usage_limit_exceeded",
        ),
        ({"codexErrorInfo": "sessionBudgetExceeded", "message": "Budget spent."}, "max_budget"),
        ({"codexErrorInfo": "unauthorized", "message": "Log in again."}, "authentication_failed"),
        ({"codexErrorInfo": "badRequest", "message": "Bad input."}, "invalid_request"),
        (
            {"codexErrorInfo": "other", "message": "Wrote 1500 tokens before timeout_sec."},
            "provider_exception",
        ),
    ],
)
@pytest.mark.asyncio
async def test_codex_failed_turns_are_classified(error, error_type):
    req = RunRequest(provider="openai", prompt="ignored")

    out = [event async for event in _stream_turn(FakeTurn([failed_turn(error)]), req)]

    assert [type(event) for event in out] == [Error]
    assert out[0].message == error["message"]
    assert out[0].error_type == error_type


@pytest.mark.asyncio
async def test_codex_context_compaction_is_surfaced():
    from agent_sdk_wrapper.events import ContextCompacted

    req = RunRequest(provider="openai", prompt="ignored")
    events = [
        SimpleNamespace(
            method="item/completed",
            payload=SimpleNamespace(
                item=SimpleNamespace(root=SimpleNamespace(type="contextCompaction"))
            ),
        ),
    ]

    out = [event async for event in _stream_turn(FakeTurn([*events, turn_completed()]), req)]

    assert [type(event) for event in out] == [ContextCompacted]
    assert out[0].trigger == "codex"


@pytest.mark.asyncio
async def test_codex_reasoning_item_without_a_summary_still_emits_thinking():
    """Preserve reasoning events with empty summaries."""
    req = RunRequest(provider="openai", prompt="ignored")
    events = [
        SimpleNamespace(
            method="item/completed",
            payload=SimpleNamespace(
                item=SimpleNamespace(
                    root=SimpleNamespace(
                        type="reasoning", id="rs_1", summary=[], content=[]
                    )
                )
            ),
        ),
    ]

    out = [event async for event in _stream_turn(FakeTurn([*events, turn_completed()]), req)]

    assert [type(event) for event in out] == [Thinking]
    assert out[0].text == ""


@pytest.mark.parametrize(
    ("account", "requires_auth", "cli_login", "allowed"),
    [
        ({"type": "apiKey"}, True, "deny", True),
        ({"type": "amazonBedrock"}, True, "deny", True),
        (None, False, "deny", True),
        ({"type": "chatgpt", "email": None, "planType": "pro"}, True, "deny", False),
        (None, True, "deny", False),
        ({"type": "chatgpt", "email": None, "planType": "pro"}, True, "require", True),
        ({"type": "apiKey"}, True, "require", False),
        (None, True, "require", False),
    ],
)
def test_codex_account_check_enforces_cli_login(account, requires_auth, cli_login, allowed):
    from openai_codex.generated.v2_all import GetAccountResponse

    from agent_sdk_wrapper.providers.openai_provider import _account_problem

    response = GetAccountResponse.model_validate(
        {"account": account, "requiresOpenaiAuth": requires_auth}
    )
    assert (_account_problem(response, cli_login) is None) is allowed


def test_codex_cli_login_deny_needs_an_api_key_and_require_takes_none(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    req = RunRequest(provider="openai", prompt="x")

    assert "OPENAI_API_KEY" in (OpenAIProvider().check_credentials(req) or "")
    assert OpenAIProvider(api_key="sk-test").check_credentials(req) is None
    assert OpenAIProvider(model_provider="local").check_credentials(req) is None

    required = RunRequest(provider="openai", prompt="x", cli_login="require")
    assert OpenAIProvider().check_credentials(required) is None
    with pytest.raises(ConfigError, match="cli_login='require'"):
        OpenAIProvider(api_key="sk-test").validate_request(required)


def test_codex_api_key_comes_from_the_run_env_first(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-host")
    provider = OpenAIProvider()

    run_key = RunRequest(provider="openai", prompt="x", env={"OPENAI_API_KEY": "sk-run"})
    assert provider._login_api_key(run_key) == "sk-run"
    no_key = RunRequest(provider="openai", prompt="x", env={"OPENAI_API_KEY": ""})
    assert provider._login_api_key(no_key) is None
    assert provider.check_credentials(no_key) is not None


@pytest.mark.parametrize(
    ("override", "request_kwargs"),
    [
        ('cli_auth_credentials_store="file"', {}),
        ('forced_login_method="api"', {}),
        ('web_search="live"', {"web_tools": False}),
    ],
)
def test_codex_caller_overrides_cannot_undo_wrapper_controls(override, request_kwargs):
    provider = OpenAIProvider(config={"config_overrides": (override,)})
    with pytest.raises(ConfigError, match="conflict"):
        provider.validate_request(RunRequest(provider="openai", prompt="x", **request_kwargs))


def test_codex_custom_model_providers_skip_the_openai_key_gate(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    req = RunRequest(provider="openai", prompt="x")

    by_thread = OpenAIProvider(thread_options={"model_provider": "local"})
    by_override = OpenAIProvider(config={"config_overrides": ('model_provider="local"',)})
    assert by_thread.check_credentials(req) is None
    assert by_override.check_credentials(req) is None


def test_codex_web_search_call_waits_for_its_query():
    started = SimpleNamespace(
        method="item/started",
        payload=SimpleNamespace(
            item=SimpleNamespace(root=SimpleNamespace(type="webSearch", id="w1", query=""))
        ),
    )
    completed = SimpleNamespace(
        method="item/completed",
        payload=SimpleNamespace(
            item=SimpleNamespace(
                root=SimpleNamespace(type="webSearch", id="w1", query="codex sdk", action=None)
            )
        ),
    )
    req = RunRequest(provider="openai", prompt="ignored")

    async def collect():
        return [event async for event in _stream_turn(FakeTurn([started, completed]), req)]

    events = asyncio.run(collect())
    calls = [event for event in events if isinstance(event, ToolCall)]
    assert [call.input["query"] for call in calls] == ["codex sdk"]


def test_codex_optional_nulls_follow_the_matching_union_member():
    from agent_sdk_wrapper.providers.openai_provider import _structured_value

    class Cat(BaseModel):
        kind: Literal["cat"]
        lives: int = 9

    class Dog(BaseModel):
        kind: Literal["dog"]
        breed: str = "mutt"
        lives: int | None

    class Owner(BaseModel):
        pet: Cat | Dog

    value = {"pet": {"kind": "dog", "breed": None, "lives": None}}
    cleaned = _structured_value(Owner, value)
    assert Owner.model_validate(cleaned).pet == Dog(kind="dog", lives=None)


@pytest.mark.parametrize(
    ("cli_login", "blanked"),
    [
        ("deny", ("CODEX_ACCESS_TOKEN", "OPENAI_API_KEY", "CODEX_API_KEY")),
        ("require", ("OPENAI_API_KEY", "CODEX_API_KEY")),
    ],
)
def test_codex_login_policy_blanks_credential_env(monkeypatch, cli_login, blanked):
    import openai_codex

    seen = {}

    class CapturingCodex:
        def __init__(self, config=None):
            seen["env"] = dict(config.env or {})

        async def __aenter__(self):
            raise RuntimeError("stop after capturing config")

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(openai_codex, "AsyncCodex", CapturingCodex)
    req = RunRequest(provider="openai", prompt="x", cli_login=cli_login)

    async def collect():
        provider = OpenAIProvider(api_key="sk-test" if cli_login == "deny" else None)
        return [event async for event in provider.stream(req)]

    with pytest.raises(AgentSdkWrapperError):
        asyncio.run(collect())
    assert {name: seen["env"].get(name) for name in blanked} == dict.fromkeys(blanked, "")


def test_codex_recursive_output_schemas_terminate():
    from agent_sdk_wrapper.providers.openai_provider import _codex_output_schema

    class Node(BaseModel):
        name: str
        parent: Node = Field(default=None, description="parent")

    Node.model_rebuild()
    schema = _codex_output_schema(Node)
    assert schema["required"] == ["name", "parent"]


@pytest.mark.parametrize(
    ("thread_config", "request_kwargs"),
    [
        ({"web_search": "live"}, {"web_tools": False}),
        ({"tools": {"web_search": True}}, {"web_tools": False}),
        ({"cli_auth_credentials_store": "file"}, {}),
    ],
)
def test_codex_thread_config_cannot_undo_wrapper_controls(thread_config, request_kwargs):
    req = RunRequest(
        provider="openai",
        prompt="x",
        extra_options={"thread_options": {"config": thread_config}},
        **request_kwargs,
    )
    with pytest.raises(ConfigError, match="config"):
        OpenAIProvider().validate_request(req)


def test_codex_tool_manifest_skips_undecodable_sys_path_entries(monkeypatch):
    import sys

    from agent_sdk_wrapper.providers.openai_provider import _tool_manifest

    monkeypatch.setattr(sys, "path", ["/ok", "/bad\udcff"])
    assert json.dumps(_tool_manifest([])["sys_path"]) == '["/ok"]'



def test_codex_plan_updates_become_one_thinking_checklist():
    def step(text, status):
        return SimpleNamespace(step=text, status=status)

    def plan(*steps):
        payload = SimpleNamespace(plan=list(steps))
        return SimpleNamespace(method="turn/plan/updated", payload=payload)
    req = RunRequest(provider="openai", prompt="ignored")
    events = [
        plan(step("read", "inProgress")),
        plan(step("read", "completed"), step("fix", "pending")),
        turn_completed(),
    ]

    async def collect():
        return [event async for event in _stream_turn(FakeTurn(events), req)]

    thinking = [event.text for event in asyncio.run(collect()) if isinstance(event, Thinking)]
    assert thinking == ["- [x] read\n- [ ] fix"]
