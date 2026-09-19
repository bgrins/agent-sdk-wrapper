"""Anthropic provider option mapping."""

from __future__ import annotations

import asyncio
import json

import pytest

from agent_sdk_wrapper import (
    AgentSdkWrapperError,
    ConfigError,
    McpHttpServer,
    McpStdioServer,
    RunRequest,
)
from agent_sdk_wrapper.events import WarningEvent
from agent_sdk_wrapper.providers.anthropic_provider import AnthropicProvider


def test_anthropic_options_map_mcp_filters_session_and_env_passthrough(monkeypatch):
    monkeypatch.setenv("INHERITED_MODE", "parent")
    monkeypatch.setenv("OVERRIDE_MODE", "parent")

    req = RunRequest(
        provider="anthropic",
        prompt="ignored",
        allowed_tools=["Read"],
        disallowed_tools=["Bash"],
        session_id="sess-123",
        mcp_servers=[
            McpStdioServer(
                name="local",
                command="uv",
                args=["run", "server"],
                env={"MODE": "test", "OVERRIDE_MODE": "explicit"},
                env_passthrough=["INHERITED_MODE", "OVERRIDE_MODE", "MISSING_MODE"],
                enabled_tools=["review"],
                disabled_tools=["delete"],
            ),
            McpHttpServer(
                name="remote",
                url="https://example.test/mcp",
                headers={"X-Test": "1"},
                enabled_tools=["search"],
            ),
        ],
    )

    options = AnthropicProvider()._build_options(req)

    assert options.resume == "sess-123"
    assert options.allowed_tools == [
        "Read",
        "mcp__local__review",
        "mcp__remote__search",
    ]
    assert options.disallowed_tools == ["Bash", "mcp__local__delete"]
    assert options.mcp_servers["local"] == {
        "type": "stdio",
        "command": "uv",
        "args": ["run", "server"],
        "env": {
            "INHERITED_MODE": "parent",
            "OVERRIDE_MODE": "explicit",
            "MODE": "test",
        },
    }
    assert options.mcp_servers["remote"] == {
        "type": "http",
        "url": "https://example.test/mcp",
        "headers": {"X-Test": "1"},
    }


def test_anthropic_options_skip_disabled_mcp_servers():
    req = RunRequest(
        provider="anthropic",
        prompt="ignored",
        mcp_servers=[
            McpStdioServer(
                name="disabled",
                command="uv",
                enabled=False,
                enabled_tools=["review"],
            )
        ],
    )

    options = AnthropicProvider()._build_options(req)

    assert "disabled" not in options.mcp_servers
    assert options.allowed_tools == []


def test_anthropic_options_map_builtin_tools():
    none_options = AnthropicProvider()._build_options(
        RunRequest(provider="anthropic", prompt="ignored", builtin_tools="none")
    )
    assert none_options.tools == []

    allowlist_options = AnthropicProvider()._build_options(
        RunRequest(provider="anthropic", prompt="ignored", builtin_tools=["Read", "Grep"])
    )
    assert allowlist_options.tools == ["Read", "Grep"]


def test_anthropic_web_tools_false_adds_web_tool_denylist():
    options = AnthropicProvider()._build_options(
        RunRequest(
            provider="anthropic",
            prompt="ignored",
            web_tools=False,
            disallowed_tools=["WebSearch"],
        )
    )

    assert options.disallowed_tools.count("WebSearch") == 1
    assert "WebFetch" in options.disallowed_tools


def test_anthropic_options_default_to_adaptive_summarized_thinking():
    options = AnthropicProvider()._build_options(
        RunRequest(provider="anthropic", prompt="ignored", effort="high")
    )

    assert options.effort == "high"
    assert options.thinking == {"type": "adaptive", "display": "summarized"}


def test_anthropic_options_allow_thinking_override():
    options = AnthropicProvider()._build_options(
        RunRequest(
            provider="anthropic",
            prompt="ignored",
            extra_options={"thinking": None},
        )
    )

    assert options.thinking is None


def test_anthropic_options_reject_partial_messages():
    req = RunRequest(
        provider="anthropic",
        prompt="ignored",
        extra_options={"include_partial_messages": True},
    )

    with pytest.raises(ConfigError, match="include_partial_messages"):
        AnthropicProvider()._build_options(req)


def test_anthropic_options_reject_builtin_tools_extra_option_conflict():
    req = RunRequest(
        provider="anthropic",
        prompt="ignored",
        builtin_tools="none",
        extra_options={"tools": []},
    )

    with pytest.raises(ConfigError, match="builtin_tools"):
        AnthropicProvider()._build_options(req)


def test_anthropic_options_reject_unsupported_mcp_fields(tmp_path):
    req = RunRequest(
        provider="anthropic",
        prompt="ignored",
        mcp_servers=[
            McpStdioServer(
                name="local",
                command="uv",
                cwd=tmp_path,
                required=True,
                default_tools_approval_mode="approve",
            )
        ],
    )

    with pytest.raises(
        ConfigError,
        match="default_tools_approval_mode, required, cwd",
    ):
        AnthropicProvider()._build_options(req)


def test_anthropic_stream_maps_rate_limit_events(monkeypatch, tmp_path):
    import claude_agent_sdk
    from claude_agent_sdk import RateLimitEvent, RateLimitInfo

    async def fake_query(**kwargs):
        yield RateLimitEvent(
            rate_limit_info=RateLimitInfo(
                status="allowed_warning",
                rate_limit_type="five_hour",
                utilization=0.8,
                resets_at=123,
            ),
            uuid="rate-limit-uuid",
            session_id="sess-rate-limit",
        )

    monkeypatch.setattr(claude_agent_sdk, "query", fake_query)

    provider_events = []

    async def collect():
        return [
            event
            async for event in AnthropicProvider().stream(
                RunRequest(
                    provider="anthropic",
                    prompt="ignored",
                    artifacts_dir=tmp_path,
                    on_provider_event=provider_events.append,
                )
            )
        ]

    events = asyncio.run(collect())

    assert len(events) == 1
    assert isinstance(events[0], WarningEvent)
    assert "allowed_warning" in events[0].message
    assert "five_hour" in events[0].message
    path = tmp_path / "provider-events.jsonl"
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 1
    assert lines[0]["sequence"] == 0
    assert lines[0]["provider"] == "anthropic"
    assert lines[0]["class"].endswith(".RateLimitEvent")
    assert lines[0]["message"]["rate_limit_info"]["status"] == "allowed_warning"
    assert len(provider_events) == 1
    assert provider_events[0].to_dict() == lines[0]


def test_anthropic_stream_rejects_unexpected_stream_events(monkeypatch):
    import claude_agent_sdk
    from claude_agent_sdk import StreamEvent

    async def fake_query(**kwargs):
        yield StreamEvent(
            uuid="partial-uuid",
            session_id="sess-partial",
            event={
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "partial"},
            },
        )

    monkeypatch.setattr(claude_agent_sdk, "query", fake_query)

    async def collect():
        return [
            event
            async for event in AnthropicProvider().stream(
                RunRequest(provider="anthropic", prompt="ignored")
            )
        ]

    with pytest.raises(AgentSdkWrapperError, match="partial StreamEvent"):
        asyncio.run(collect())


def _result(**kwargs):
    from claude_agent_sdk import ResultMessage

    defaults = {
        "subtype": "success",
        "duration_ms": 10,
        "duration_api_ms": 5,
        "is_error": False,
        "num_turns": 3,
        "session_id": "sess-1",
    }
    return ResultMessage(**{**defaults, **kwargs})


def test_anthropic_usage_folds_cache_into_input_and_counts_requests():
    from agent_sdk_wrapper.providers.anthropic_provider import _usage_event

    event = _usage_event(
        {
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_read_input_tokens": 700,
            "cache_creation_input_tokens": 200,
        },
        0.25,
        requests=4,
    )

    # Normalized input includes cache.
    assert event.usage.input_tokens == 1000
    assert event.usage.cache_read_tokens == 700
    assert event.usage.cache_write_tokens == 200
    assert event.usage.total_tokens == 1020
    assert event.usage.requests == 4
    assert event.cost_usd == 0.25


@pytest.mark.parametrize("legacy_usage", [None, {"input_tokens": 100, "output_tokens": 50}])
async def test_anthropic_run_counts_subagent_models_without_adding_main_loop_twice(
    monkeypatch, legacy_usage
):
    import claude_agent_sdk

    from agent_sdk_wrapper import Agent, TokenUsage

    models = {
        "claude-main": {
            "inputTokens": 100,
            "outputTokens": 50,
            "cacheReadInputTokens": 1000,
            "cacheCreationInputTokens": 200,
            "costUSD": 1.5,
        },
        "claude-subagent": {
            "inputTokens": 10,
            "outputTokens": 5,
            "cacheReadInputTokens": 100,
            "cacheCreationInputTokens": 0,
            "costUSD": 0.01,
        },
    }

    async def fake_query(**kwargs):
        yield _result(usage=legacy_usage, model_usage=models, total_cost_usd=1.51)

    monkeypatch.setattr(claude_agent_sdk, "query", fake_query)
    result = await Agent(provider="anthropic", include_raw=True).run(
        "Count main and subagent usage"
    )
    assert result.status == "success"
    assert result.usage == TokenUsage(
        input_tokens=1410,
        output_tokens=55,
        total_tokens=1465,
        cache_read_tokens=1100,
        cache_write_tokens=200,
        requests=3,
    )
    assert result.cost_usd == 1.51
    usage_events = [env.event for env in result.events if env.event.type == "usage"]
    assert len(usage_events) == 1
    assert usage_events[0].raw["model_usage"] == models


def test_anthropic_sparse_model_usage_is_authoritative_even_with_zero_counts():
    from agent_sdk_wrapper.providers.anthropic_provider import _usage_event

    event = _usage_event(
        {"input_tokens": 999},
        0.0,
        model_usage={"claude-main": {"outputTokens": None}},
    )
    assert event.usage.total_tokens == 0
    assert event.cost_usd == 0.0


def test_anthropic_max_turns_result_reports_max_turns_not_a_generic_error():
    from agent_sdk_wrapper.providers.anthropic_provider import _result_error

    error = _result_error(_result(subtype="error_max_turns", is_error=True))

    assert error is not None
    assert error.error_type == "max_turns"
    assert error.retryable is False


def test_anthropic_refusal_reports_refused():
    from agent_sdk_wrapper.providers.anthropic_provider import _result_error

    error = _result_error(_result(stop_reason="refusal", result="I can't help with that"))

    assert error is not None
    assert error.error_type == "refused"


@pytest.mark.parametrize("status", [429, 503, 529])
def test_anthropic_retryable_api_status_is_marked_retryable(status):
    from agent_sdk_wrapper.providers.anthropic_provider import _result_error

    error = _result_error(_result(is_error=True, api_error_status=status))

    assert error is not None
    assert error.error_type == "transient_api_error"
    assert error.retryable is True


def test_anthropic_client_error_is_not_retryable():
    from agent_sdk_wrapper.providers.anthropic_provider import _result_error

    error = _result_error(
        _result(is_error=True, api_error_status=400, errors=["bad request"])
    )

    assert error is not None
    assert error.retryable is False
    assert error.message == "bad request"


def test_anthropic_successful_result_reports_no_error():
    from agent_sdk_wrapper.providers.anthropic_provider import _result_error

    assert _result_error(_result(stop_reason="end_turn")) is None


def test_anthropic_compact_boundary_maps_to_context_compacted():
    from claude_agent_sdk import SystemMessage

    from agent_sdk_wrapper.providers.anthropic_provider import _compaction_event

    event = _compaction_event(
        SystemMessage(
            subtype="compact_boundary",
            data={"compact_metadata": {"trigger": "auto", "pre_tokens": 150_000}},
        )
    )

    assert event is not None
    assert event.trigger == "auto"
    assert event.pre_tokens == 150_000
    assert _compaction_event(SystemMessage(subtype="init", data={})) is None


def test_anthropic_thinking_reports_redacted_size_when_text_is_hidden():
    from claude_agent_sdk import ThinkingBlock

    from agent_sdk_wrapper.providers.anthropic_provider import _thinking_event

    visible = _thinking_event(ThinkingBlock(thinking="a plan", signature="sig"))
    hidden = _thinking_event(ThinkingBlock(thinking="", signature="x" * 42))

    assert visible.text == "a plan"
    assert visible.redacted_bytes is None
    assert hidden.redacted_bytes == 42


def test_anthropic_signal_killed_runtime_raises_process_terminated(monkeypatch):
    import claude_agent_sdk
    from claude_agent_sdk import ProcessError

    from agent_sdk_wrapper import ProcessTerminatedError

    async def fake_query(**kwargs):
        raise ProcessError("Command failed", exit_code=143)
        yield  # pragma: no cover - generator marker

    monkeypatch.setattr(claude_agent_sdk, "query", fake_query)

    async def collect():
        return [
            event
            async for event in AnthropicProvider().stream(
                RunRequest(provider="anthropic", prompt="ignored")
            )
        ]

    with pytest.raises(ProcessTerminatedError) as excinfo:
        asyncio.run(collect())
    assert excinfo.value.signal == 15


def test_anthropic_stream_maps_subagent_lifecycle_and_names_tool_results(monkeypatch):
    import claude_agent_sdk
    from claude_agent_sdk import (
        AssistantMessage,
        TaskNotificationMessage,
        TaskStartedMessage,
        ToolResultBlock,
        ToolUseBlock,
        UserMessage,
    )

    from agent_sdk_wrapper.events import SubagentEnded, SubagentStarted, ToolResult

    async def fake_query(**kwargs):
        yield TaskStartedMessage(
            subtype="task_started",
            task_id="task-1",
            description="review the diff",
            uuid="u1",
            session_id="sess-1",
            task_type="local_agent",
            data={"subagent_type": "reviewer"},
        )
        yield AssistantMessage(
            content=[ToolUseBlock(id="tool-1", name="Read", input={"path": "a.py"})],
            model="claude-opus-5",
        )
        yield UserMessage(
            content=[ToolResultBlock(tool_use_id="tool-1", content="file body")]
        )
        yield TaskNotificationMessage(
            subtype="task_notification",
            data={},
            task_id="task-1",
            status="completed",
            output_file="out.md",
            summary="looks fine",
            uuid="u2",
            session_id="sess-1",
        )

    monkeypatch.setattr(claude_agent_sdk, "query", fake_query)

    async def collect():
        return [
            event
            async for event in AnthropicProvider().stream(
                RunRequest(provider="anthropic", prompt="ignored")
            )
        ]

    events = asyncio.run(collect())

    started = next(e for e in events if isinstance(e, SubagentStarted))
    ended = next(e for e in events if isinstance(e, SubagentEnded))
    result = next(e for e in events if isinstance(e, ToolResult))
    assert (started.task_id, started.name, started.description) == (
        "task-1",
        "reviewer",
        "review the diff",
    )
    assert (ended.task_id, ended.status, ended.summary) == ("task-1", "completed", "looks fine")
    assert result.name == "Read"


def test_anthropic_error_type_ignores_a_normal_stop_reason():
    """Classify HTTP 400 as a failure even when ``stop_reason`` is ``stop_sequence``."""
    from agent_sdk_wrapper.providers.anthropic_provider import _result_error

    error = _result_error(
        _result(is_error=True, api_error_status=400, stop_reason="stop_sequence")
    )

    assert error is not None
    assert error.error_type == "invalid_request"
    assert error.retryable is False


def test_anthropic_error_type_falls_back_to_an_error_subtype():
    from agent_sdk_wrapper.providers.anthropic_provider import _result_error

    error = _result_error(
        _result(subtype="error_during_execution", is_error=True, stop_reason="end_turn")
    )

    assert error is not None
    assert error.error_type == "execution_error"


def _stream(monkeypatch, messages, **request):
    """Run the adapter over scripted SDK messages; return (events, options)."""
    import claude_agent_sdk

    seen = {}

    async def fake_query(*, prompt, options):
        seen["options"] = options
        for message in messages:
            if isinstance(message, BaseException):
                raise message
            yield message

    monkeypatch.setattr(claude_agent_sdk, "query", fake_query)

    async def collect():
        req = RunRequest(provider="anthropic", prompt="ignored", **request)
        return [event async for event in AnthropicProvider().stream(req)]

    events = asyncio.run(collect())
    return events, seen["options"]


def _assistant(*blocks, **kwargs):
    from claude_agent_sdk import AssistantMessage

    model = kwargs.pop("model", "claude-test")
    return AssistantMessage(content=list(blocks), model=model, **kwargs)


def test_anthropic_options_isolate_settings_by_default():
    options = AnthropicProvider()._build_options(RunRequest(provider="anthropic", prompt="x"))
    assert options.setting_sources == []

    explicit = AnthropicProvider()._build_options(
        RunRequest(provider="anthropic", prompt="x", setting_sources=["project"])
    )
    assert explicit.setting_sources == ["project"]

    with pytest.raises(ConfigError, match="setting_sources"):
        AnthropicProvider()._build_options(
            RunRequest(provider="anthropic", prompt="x", setting_sources=["global"])
        )


def test_anthropic_env_pins_effort_and_disables_background_tasks():
    options = AnthropicProvider()._build_options(
        RunRequest(provider="anthropic", prompt="x", effort="high", env={"KEEP": "1"})
    )
    assert options.env == {
        "KEEP": "1",
        "CLAUDE_CODE_EFFORT_LEVEL": "high",
        "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
    }

    caller = AnthropicProvider()._build_options(
        RunRequest(
            provider="anthropic",
            prompt="x",
            env={"CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "0"},
        )
    )
    assert caller.env == {"CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "0"}

    with pytest.raises(ConfigError, match="CLAUDE_CODE_EFFORT_LEVEL"):
        AnthropicProvider()._build_options(
            RunRequest(
                provider="anthropic",
                prompt="x",
                effort="high",
                env={"CLAUDE_CODE_EFFORT_LEVEL": "low"},
            )
        )


def test_anthropic_options_leave_buffer_headroom_unless_overridden():
    default = AnthropicProvider()._build_options(RunRequest(provider="anthropic", prompt="x"))
    assert default.max_buffer_size == 16 * 1024 * 1024

    override = AnthropicProvider()._build_options(
        RunRequest(provider="anthropic", prompt="x", extra_options={"max_buffer_size": 1})
    )
    assert override.max_buffer_size == 1


@pytest.mark.parametrize(
    ("request_kwargs", "match"),
    [
        ({"max_turns": 0}, "max_turns"),
        ({"extra_options": {"allowed_tools": ["Bash"]}}, "allowed_tools"),
        ({"extra_options": {"mcp_servers": {}}}, "mcp_servers"),
        ({"web_tools": True, "builtin_tools": "none"}, "web_tools"),
        ({"web_tools": True, "disallowed_tools": ["WebFetch"]}, "WebFetch"),
    ],
)
def test_anthropic_rejects_options_the_sdk_would_ignore_or_override(request_kwargs, match):
    with pytest.raises(ConfigError, match=match):
        AnthropicProvider()._build_options(
            RunRequest(provider="anthropic", prompt="x", **request_kwargs)
        )


def test_anthropic_web_tools_true_and_subagents_extend_a_builtin_allowlist():
    from agent_sdk_wrapper import SubagentDef

    options = AnthropicProvider()._build_options(
        RunRequest(
            provider="anthropic",
            prompt="x",
            builtin_tools=["Read"],
            web_tools=True,
            subagents={"helper": SubagentDef(description="d", prompt="p")},
        )
    )
    assert options.tools == ["Read", "WebSearch", "WebFetch", "Agent"]
    assert "Agent" in options.allowed_tools


def test_anthropic_session_info_reports_the_resolved_model(monkeypatch):
    from claude_agent_sdk import SystemMessage

    from agent_sdk_wrapper.events import SessionInfo

    events, _ = _stream(
        monkeypatch,
        [
            SystemMessage(
                subtype="init", data={"session_id": "sess-1", "model": "claude-opus-5"}
            )
        ],
    )
    assert events == [SessionInfo(id="sess-1", model="claude-opus-5")]


def test_anthropic_usage_counts_thinking_tokens_from_either_usage_shape():
    from agent_sdk_wrapper.providers.anthropic_provider import _usage_event

    by_model = _usage_event(
        {"input_tokens": 1, "output_tokens": 600},
        0.01,
        model_usage={
            "claude-main": {"inputTokens": 40, "outputTokens": 620, "thinkingTokens": 360},
            "claude-helper": {"inputTokens": 900, "outputTokens": 17, "thinkingTokens": 0},
        },
    )
    assert by_model.usage.output_tokens == 637
    assert by_model.usage.reasoning_output_tokens == 360
    assert by_model.raw is None

    legacy = _usage_event(
        {
            "input_tokens": 40,
            "output_tokens": 620,
            "output_tokens_details": {"thinking_tokens": 360},
        },
        0.01,
        include_raw=True,
    )
    assert legacy.usage.reasoning_output_tokens == 360
    assert legacy.raw["output_tokens"] == 620


def test_anthropic_reports_reasoning_even_without_a_thinking_block(monkeypatch):
    from agent_sdk_wrapper.events import SessionInfo, Thinking, Usage

    events, _ = _stream(
        monkeypatch,
        [_result(model_usage={"m": {"outputTokens": 10, "thinkingTokens": 4}})],
    )
    assert [type(event) for event in events] == [SessionInfo, Thinking, Usage]


def test_anthropic_synthetic_error_message_is_a_classified_failure_not_text(monkeypatch):
    from claude_agent_sdk import TextBlock

    from agent_sdk_wrapper.events import Error, Text

    events, _ = _stream(
        monkeypatch,
        [
            _assistant(
                TextBlock(text="Not logged in · Please run /login"),
                model="<synthetic>",
                error="authentication_failed",
            ),
            _result(is_error=True, result="Not logged in · Please run /login"),
        ],
    )
    assert not any(isinstance(event, Text) for event in events)
    error = next(event for event in events if isinstance(event, Error))
    assert (error.error_type, error.retryable) == ("authentication_failed", False)
    assert error.message == "Not logged in · Please run /login"


@pytest.mark.parametrize(
    ("result", "error_type", "retryable"),
    [
        ({"is_error": True, "result": "API Error: Connection error."}, "transient_api_error", True),
        ({"is_error": True, "result": "Connection refused"}, "transient_api_error", True),
        ({"is_error": True, "result": "something odd"}, "execution_error", False),
        (
            {"is_error": True, "terminal_reason": "prompt_too_long"},
            "context_window_exceeded",
            False,
        ),
        ({"subtype": "error_max_budget_usd", "is_error": True}, "max_budget", False),
        (
            {"subtype": "error_max_structured_output_retries", "is_error": True},
            "structured_output_failed",
            False,
        ),
        ({"is_error": True, "terminal_reason": "aborted_tools"}, "cancelled", False),
        ({"is_error": True, "api_error_status": 401}, "authentication_failed", False),
    ],
)
def test_anthropic_result_errors_use_structured_signals_and_keep_evidence(
    result, error_type, retryable
):
    from agent_sdk_wrapper.providers.anthropic_provider import _result_error

    error = _result_error(_result(**result))

    assert error is not None
    assert (error.error_type, error.retryable) == (error_type, retryable)


def test_anthropic_omits_subagent_scoped_messages(monkeypatch):
    from claude_agent_sdk import TextBlock, ToolResultBlock, ToolUseBlock, UserMessage

    from agent_sdk_wrapper.events import Text, ToolCall, ToolResult, WarningEvent

    events, _ = _stream(
        monkeypatch,
        [
            _assistant(
                ToolUseBlock(id="sub-tool", name="Bash", input={}),
                parent_tool_use_id="agent-1",
            ),
            UserMessage(
                content=[ToolResultBlock(tool_use_id="sub-tool", content="x")],
                parent_tool_use_id="agent-1",
            ),
            _assistant(TextBlock(text="done")),
        ],
    )
    assert not any(isinstance(event, (ToolCall, ToolResult)) for event in events)
    assert [type(event) for event in events] == [WarningEvent, Text]


def test_anthropic_subagents_pair_start_with_either_terminal_message(monkeypatch):
    from claude_agent_sdk import TaskStartedMessage, TaskUpdatedMessage

    from agent_sdk_wrapper.events import SubagentEnded, SubagentStarted

    def started(task_id, task_type):
        return TaskStartedMessage(
            subtype="task_started",
            data={},
            task_id=task_id,
            description="work",
            uuid="u",
            session_id="s",
            task_type=task_type,
        )

    events, _ = _stream(
        monkeypatch,
        [
            started("shell", "local_bash"),
            started("agent", "local_agent"),
            TaskUpdatedMessage(
                subtype="task_updated", data={}, task_id="shell", patch={"status": "killed"}
            ),
            TaskUpdatedMessage(
                subtype="task_updated", data={}, task_id="agent", patch={"status": "killed"}
            ),
        ],
    )
    assert events == [
        SubagentStarted(task_id="agent", name="local_agent", description="work"),
        SubagentEnded(task_id="agent", status="killed"),
    ]


def test_anthropic_refusal_retraction_is_a_protocol_error(monkeypatch):
    from claude_agent_sdk import SystemMessage, TextBlock

    from agent_sdk_wrapper.events import Error

    events, _ = _stream(
        monkeypatch,
        [
            _assistant(TextBlock(text="partial")),
            SystemMessage(
                subtype="model_refusal_fallback", data={"retracted_message_uuids": ["m1"]}
            ),
            _assistant(TextBlock(text="fallback answer")),
        ],
    )
    assert isinstance(events[-1], Error)
    assert events[-1].error_type == "provider_protocol_error"


def test_anthropic_process_errors_carry_the_stderr_tail(monkeypatch):
    import claude_agent_sdk
    from claude_agent_sdk import ProcessError

    from agent_sdk_wrapper import TransientError

    async def fake_query(*, prompt, options):
        options.stderr("API Error: 529 overloaded_error")
        raise ProcessError("Command failed with exit code 1", exit_code=1)
        yield  # pragma: no cover - generator marker

    monkeypatch.setattr(claude_agent_sdk, "query", fake_query)

    async def collect():
        return [
            event
            async for event in AnthropicProvider().stream(
                RunRequest(provider="anthropic", prompt="ignored")
            )
        ]

    with pytest.raises(TransientError, match="529 overloaded_error"):
        asyncio.run(collect())


def test_anthropic_chains_a_user_stderr_callback():
    lines = []
    options = AnthropicProvider()._build_options(
        RunRequest(provider="anthropic", prompt="x", extra_options={"stderr": lines.append}),
        stderr_tail=None,
    )
    options.stderr("hello")
    assert lines == ["hello"]


def test_anthropic_joins_text_blocks_and_maps_server_tool_results(monkeypatch):
    from claude_agent_sdk import ServerToolResultBlock, ServerToolUseBlock, TextBlock

    from agent_sdk_wrapper.events import Text, ToolCall, ToolResult

    events, _ = _stream(
        monkeypatch,
        [
            _assistant(
                TextBlock(text="Searching."),
                ServerToolUseBlock(id="srv-1", name="web_search", input={"query": "q"}),
                ServerToolResultBlock(tool_use_id="srv-1", content={"type": "web_search_result"}),
                TextBlock(text="The answer "),
                TextBlock(text="is 42."),
            )
        ],
    )
    assert [type(event) for event in events] == [Text, ToolCall, ToolResult, Text]
    assert events[-1].text == "The answer is 42."
    assert events[2].name == "web_search"
    assert events[2].output == '{"type": "web_search_result"}'


def test_anthropic_uses_result_text_when_no_text_block_arrived(monkeypatch):
    from agent_sdk_wrapper.events import Text

    events, _ = _stream(monkeypatch, [_result(result="final answer")])
    assert [event.text for event in events if isinstance(event, Text)] == ["final answer"]


def test_anthropic_structured_output_mismatch_is_a_typed_failure(monkeypatch):
    from pydantic import BaseModel

    from agent_sdk_wrapper.events import Error

    class Answer(BaseModel):
        value: int

    events, _ = _stream(
        monkeypatch,
        [_result(structured_output={"value": "not an int"})],
        output_schema=Answer,
    )
    assert events[-1] == Error(message=events[-1].message, error_type="structured_output_failed")
