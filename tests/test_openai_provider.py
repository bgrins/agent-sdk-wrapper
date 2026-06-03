"""OpenAI Codex provider event mapping."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

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
)
from agent_sdk_wrapper.providers.openai_provider import (
    OpenAIProvider,
    _codex_config,
    _codex_env,
    _codex_output_schema,
    _runtime_config,
    _stream_turn,
    _tool_entry,
    _validate_supported,
    _validate_thread_resume_options,
    _write_sdk_debug_log,
)


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
        self.interrupt_count = 0

    async def stream(self):
        for event in self._events:
            yield event

    async def interrupt(self):
        self.interrupt_count += 1


def test_codex_options_default_to_auto_reasoning_summary():
    req = RunRequest(provider="openai", prompt="ignored", effort="high")

    _, turn_options = OpenAIProvider()._build_options(req, None, None)

    assert turn_options["effort"] == "high"
    assert turn_options["summary"] == "auto"


def test_codex_options_normalize_provider_effort_alias():
    req = RunRequest(provider="openai", prompt="ignored")

    _, turn_options = OpenAIProvider(effort="max")._build_options(req, None, None)

    assert turn_options["effort"] == "xhigh"


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

    out = [event async for event in _stream_turn(FakeTurn(events), req)]

    assert out == [Text(text="hello")]
    path = tmp_path / "provider-events.jsonl"
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [line["sequence"] for line in lines] == [0, 1]
    assert [line["provider"] for line in lines] == ["openai", "openai"]
    assert lines[0]["class"] == "types.SimpleNamespace"
    assert lines[0]["message"]["method"] == "item/agentMessage/delta"
    assert lines[0]["message"]["payload"]["delta"] == "hello"
    assert len(provider_events) == 2
    assert provider_events[0].to_dict() == lines[0]
    assert provider_events[1].to_dict() == lines[1]
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

    out = [event async for event in _stream_turn(FakeTurn(events), req)]

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

    out = [event async for event in _stream_turn(FakeTurn(events), req)]

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
async def test_codex_structured_output_validation_failure_raises():
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
        SimpleNamespace(
            method="turn/completed",
            payload=SimpleNamespace(turn=SimpleNamespace(status="completed")),
        ),
    ]

    with pytest.raises(AgentSdkWrapperError, match="structured output did not match"):
        [event async for event in _stream_turn(FakeTurn(events), req)]


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

    out = [event async for event in _stream_turn(FakeTurn(events), req)]

    assert out == [
        ToolCall(id="cmd-1", name="command", input={"command": "pytest"}),
        ToolResult(id="cmd-1", output="passed"),
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

    out = [event async for event in _stream_turn(FakeTurn(events), req)]

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

    out = [event async for event in _stream_turn(FakeTurn(events), req)]

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

    out = [event async for event in _stream_turn(FakeTurn(events), req)]

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
    assert '"status": "completed"' in (agent_result.output or "")


def test_codex_tool_entry_keeps_source_fallback_for_importable_tool():
    entry = _tool_entry(sample_importable_tool)

    assert entry["module"] == __name__
    assert "def sample_importable_tool" in entry["source"]


@pytest.mark.asyncio
async def test_codex_max_turns_interrupts_after_action_item():
    req = RunRequest(provider="openai", prompt="ignored", max_turns=1)
    turn = FakeTurn(
        [
            SimpleNamespace(
                method="item/completed",
                payload=SimpleNamespace(
                    item=SimpleNamespace(
                        root=SimpleNamespace(
                            type="commandExecution",
                            id="cmd-1",
                            command="python -m pytest",
                            status="completed",
                            aggregated_output="passed",
                        )
                    )
                ),
            ),
            SimpleNamespace(
                method="item/completed",
                payload=SimpleNamespace(
                    item=SimpleNamespace(
                        root=SimpleNamespace(type="agentMessage", text="done")
                    )
                ),
            ),
            SimpleNamespace(
                method="turn/completed",
                payload=SimpleNamespace(turn=SimpleNamespace(status="completed")),
            ),
        ]
    )

    out = [event async for event in _stream_turn(turn, req)]

    assert turn.interrupt_count == 1
    assert [type(event) for event in out] == [ToolCall, ToolResult, Error]
    error = next(event for event in out if isinstance(event, Error))
    assert error.error_type == "max_turns"
    assert "max_turns=1" in error.message


@pytest.mark.asyncio
async def test_codex_max_turns_does_not_interrupt_simple_message():
    req = RunRequest(provider="openai", prompt="ignored", max_turns=1)
    turn = FakeTurn(
        [
            SimpleNamespace(
                method="item/completed",
                payload=SimpleNamespace(
                    item=SimpleNamespace(
                        root=SimpleNamespace(type="agentMessage", text="done")
                    )
                ),
            ),
            SimpleNamespace(
                method="turn/completed",
                payload=SimpleNamespace(turn=SimpleNamespace(status="completed")),
            ),
        ]
    )

    out = [event async for event in _stream_turn(turn, req)]

    assert turn.interrupt_count == 0
    assert out == [Text(text="done")]


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

    out = [event async for event in _stream_turn(FakeTurn(events), req)]

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

    out = [event async for event in _stream_turn(FakeTurn(events), req)]

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
    assert '"status": "failed"' in (results[1].output or "")


def test_thread_resume_rejects_start_only_options():
    with pytest.raises(ConfigError, match="ephemeral"):
        _validate_thread_resume_options({"model": "gpt-5.4", "ephemeral": True})


def test_thread_resume_filter_allows_resume_safe_options():
    _validate_thread_resume_options(
        {
            "approval_mode": "never",
            "model": "gpt-5.4",
            "sandbox": "workspace-write",
            "service_tier": "priority",
        }
    )


def test_thread_resume_filter_rejects_turn_and_start_only_options():
    with pytest.raises(ConfigError, match="ephemeral, turn_options"):
        _validate_thread_resume_options({"ephemeral": True, "turn_options": {"effort": "high"}})


def test_codex_filters_require_wrapper_managed_tools():
    req = RunRequest(provider="openai", prompt="ignored", allowed_tools=["Read"])

    with pytest.raises(ConfigError, match="without callable tools or MCP servers"):
        _validate_supported(req)


def test_codex_accepts_positive_max_turns():
    req = RunRequest(provider="openai", prompt="ignored", max_turns=1)

    _validate_supported(req)


def test_codex_rejects_non_positive_max_turns():
    req = RunRequest(provider="openai", prompt="ignored", max_turns=0)

    with pytest.raises(ConfigError, match="max_turns < 1"):
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


def test_codex_config_merges_config_overrides(monkeypatch):
    from agent_sdk_wrapper.providers import openai_provider as op_mod

    monkeypatch.setattr(op_mod, "_path_codex_bin_when_sdk_bin_missing", lambda: None)

    config = _codex_config(
        {"config_overrides": ("model=\"gpt-5\"",)},
        {},
        None,
        config_overrides=("features.multi_agent=true",),
    )

    assert config.config_overrides == ("model=\"gpt-5\"", "features.multi_agent=true")


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


def test_codex_rejects_builtin_tools():
    req = RunRequest(provider="openai", prompt="ignored", builtin_tools="none")

    with pytest.raises(ConfigError, match="builtin_tools"):
        _validate_supported(req)


def test_codex_web_tools_emits_config_override():
    with _runtime_config(
        RunRequest(provider="openai", prompt="ignored", web_tools=False)
    ) as runtime:
        assert "tools.web_search=false" in runtime.config_overrides
    with _runtime_config(
        RunRequest(provider="openai", prompt="ignored", web_tools=True)
    ) as runtime:
        assert "tools.web_search=true" in runtime.config_overrides


def test_codex_web_tools_coexists_with_tools(tmp_path):
    req = RunRequest(
        provider="openai",
        prompt="ignored",
        tools=[sample_importable_tool],
        cwd=tmp_path,
        web_tools=False,
    )

    with _runtime_config(req) as runtime:
        assert "tools.web_search=false" in runtime.config_overrides
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


def test_codex_output_schema_disallows_additional_properties():
    schema = _codex_output_schema(Answer)

    assert schema["additionalProperties"] is False
