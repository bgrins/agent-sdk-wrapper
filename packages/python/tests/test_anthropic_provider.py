"""Anthropic provider option mapping."""

from __future__ import annotations

import asyncio
import dataclasses
import json

import pytest

from agent_sdk_wrapper import (
    AgentSdkWrapperError,
    ConfigError,
    McpHttpServer,
    McpStdioServer,
    RunRequest,
    SubagentDef,
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


def test_every_native_option_a_first_class_field_sets_is_owned(tmp_path):
    from claude_agent_sdk import ClaudeAgentOptions
    from pydantic import BaseModel

    from agent_sdk_wrapper.providers.anthropic_provider import (
        _WRAPPER_OWNED_OPTIONS,
        _native_option_names,
    )

    class Answer(BaseModel):
        text: str

    def add(a: int, b: int) -> int:
        return a + b

    first_class = {
        "model": "claude-haiku-4-5",
        "system_prompt": "be brief",
        "tools": [add],
        "subagents": {"reviewer": SubagentDef(description="d", prompt="p")},
        "mcp_servers": [McpStdioServer(name="local", command="uv")],
        "output_schema": Answer,
        "max_turns": 2,
        "effort": "low",
        "cwd": tmp_path,
        "builtin_tools": ["Read"],
        "web_tools": True,
        "allowed_tools": ["Read"],
        "disallowed_tools": ["Bash"],
        "session_id": "sess-1",
        "permission_mode": "default",
        "setting_sources": ["project"],
    }
    # Fields that never reach ClaudeAgentOptions, or reach it only through env.
    not_native = {
        "provider", "prompt", "env", "timeout", "max_retries", "include_raw",
        "include_events_in_result", "artifacts_dir", "on_provider_event",
        "continue_session", "extra_options", "cli_login", "run_id", "attempt",
    }
    new_fields = {f.name for f in dataclasses.fields(RunRequest)} - not_native
    assert new_fields == set(first_class), "classify new RunRequest fields here"

    options = AnthropicProvider()._build_options(
        RunRequest(provider="anthropic", prompt="x", **first_class)
    )
    defaults = ClaudeAgentOptions()
    changed = {
        f.name
        for f in dataclasses.fields(options)
        if getattr(options, f.name) != getattr(defaults, f.name)
    }
    # Wrapper defaults extra_options may replace; builtin_tools guards tools separately.
    replaceable = {"max_buffer_size", "stderr", "thinking", "tools"}
    assert changed - replaceable <= set(_WRAPPER_OWNED_OPTIONS)
    assert set(_WRAPPER_OWNED_OPTIONS) <= _native_option_names()


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
        yield _result(session_id="")

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
    assert len(lines) == 2
    assert lines[0]["sequence"] == 0
    assert lines[0]["provider"] == "anthropic"
    assert lines[0]["class"].endswith(".RateLimitEvent")
    assert lines[0]["message"]["rate_limit_info"]["status"] == "allowed_warning"
    assert len(provider_events) == 2
    assert provider_events[0].to_dict() == lines[0]


def test_anthropic_runtime_retries_are_warnings(monkeypatch):
    import claude_agent_sdk
    from claude_agent_sdk import SystemMessage

    retry = {
        "type": "system",
        "subtype": "api_retry",
        "attempt": 1,
        "max_retries": 10,
        "retry_delay_ms": 600,
        "error_status": 529,
        "error": "overloaded",
        "session_id": "sess-retry",
    }

    async def fake_query(**kwargs):
        yield SystemMessage(subtype="api_retry", data=retry)
        yield _result(session_id="sess-retry")

    monkeypatch.setattr(claude_agent_sdk, "query", fake_query)

    async def collect():
        req = RunRequest(provider="anthropic", prompt="ignored")
        return [event async for event in AnthropicProvider().stream(req)]

    warnings = [e for e in asyncio.run(collect()) if isinstance(e, WarningEvent)]

    assert [w.message for w in warnings] == [
        "Claude API error 529 (overloaded); runtime retry 1/10 in 0.6s"
    ]


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


@pytest.mark.parametrize("status", [409, 429, 501, 503, 529])
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


def test_anthropic_spend_limit_400_is_a_usage_limit():
    from agent_sdk_wrapper.providers.anthropic_provider import _result_error

    text = "API Error: 400 You have reached your specified API usage limits."
    error = _result_error(
        _result(is_error=True, api_error_status=400, terminal_reason="api_error", result=text),
        ("unknown", text),
    )

    assert error is not None
    assert error.error_type == "usage_limit_exceeded"


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

    from claude_agent_sdk import ResultMessage

    if not any(isinstance(m, (ResultMessage, BaseException)) for m in messages):
        # The CLI always ends a run with a result.
        messages = [*messages, _result(session_id="")]

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
    blanked_logins = {
        "CLAUDE_CODE_OAUTH_TOKEN": "",
        "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR": "",
        "CLAUDE_CODE_OAUTH_REFRESH_TOKEN": "",
        "CLAUDE_CODE_SESSION_ACCESS_TOKEN": "",
    }
    # The SDK layers options.env over os.environ, so these values beat inherited ones.
    assert options.env == {
        "KEEP": "1",
        "CLAUDE_CODE_EFFORT_LEVEL": "high",
        "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
        **blanked_logins,
    }

    caller = AnthropicProvider()._build_options(
        RunRequest(
            provider="anthropic",
            prompt="x",
            env={"CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "0"},
        )
    )
    assert caller.env == {
        "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "0",
        "CLAUDE_CODE_EFFORT_LEVEL": "",
        **blanked_logins,
    }

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
        (
            {"allowed_tools": ["Read"], "extra_options": {"allowed_tools": ["Bash"]}},
            "allowed_tools",
        ),
        (
            {
                "mcp_servers": [McpHttpServer(name="docs", url="https://example.test")],
                "extra_options": {"mcp_servers": {}},
            },
            "mcp_servers",
        ),
        ({"extra_options": {"env": {}}}, "env"),
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
        ({"is_error": True, "result": "something odd"}, "transient_api_error", True),
        (
            {"is_error": True, "terminal_reason": "malformed_tool_use_exhausted"},
            "execution_error",
            False,
        ),
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
    assert events[2].output == '{"type":"web_search_result"}'


def test_anthropic_uses_result_text_when_no_text_block_arrived(monkeypatch):
    from agent_sdk_wrapper.events import Text

    events, _ = _stream(monkeypatch, [_result(result="final answer")])
    assert [event.text for event in events if isinstance(event, Text)] == ["final answer"]


def test_anthropic_structured_output_mismatch_is_a_typed_failure(monkeypatch):
    from pydantic import BaseModel


    class Answer(BaseModel):
        value: int

    events, _ = _stream(
        monkeypatch,
        [_result(structured_output={"value": "not an int"})],
        output_schema=Answer,
    )
    assert events[-1].error_type == "structured_output_failed"
    assert "value" in events[-1].message


def test_anthropic_joins_text_frames_of_one_message(monkeypatch):
    from claude_agent_sdk import TextBlock, ThinkingBlock

    from agent_sdk_wrapper.events import Text

    events, _ = _stream(
        monkeypatch,
        [
            _assistant(ThinkingBlock(thinking="plan", signature="s"), message_id="m1"),
            _assistant(TextBlock(text="The answer "), message_id="m1"),
            _assistant(TextBlock(text="is 42."), message_id="m1"),
            _assistant(TextBlock(text="Separate message."), message_id="m2"),
            _result(result="The answer is 42."),
        ],
    )
    assert [event.text for event in events if isinstance(event, Text)] == [
        "The answer is 42.",
        "Separate message.",
    ]


def test_anthropic_auth_failure_with_403_is_permission_denied():
    from agent_sdk_wrapper.providers.anthropic_provider import _result_error

    error = _result_error(
        _result(is_error=True, api_error_status=403), ("authentication_failed", "denied")
    )
    assert error is not None
    assert error.error_type == "permission_denied"


def test_anthropic_cli_login_require_and_login_tokens_are_rejected():
    with pytest.raises(ConfigError, match="cli_login='require'"):
        AnthropicProvider().validate_request(
            RunRequest(provider="anthropic", prompt="x", cli_login="require")
        )
    with pytest.raises(ConfigError, match="CLAUDE_CODE_OAUTH_TOKEN"):
        AnthropicProvider().validate_request(
            RunRequest(provider="anthropic", prompt="x", env={"CLAUDE_CODE_OAUTH_TOKEN": "t"})
        )


@pytest.mark.parametrize(
    "env",
    [
        {"ANTHROPIC_API_KEY": "k"},
        {"ANTHROPIC_AUTH_TOKEN": "t"},
        {"CLAUDE_CODE_USE_BEDROCK": "1"},
    ],
)
def test_anthropic_accepts_api_and_cloud_credentials(monkeypatch, env):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    req = RunRequest(provider="anthropic", prompt="x", env=env)
    assert AnthropicProvider().check_credentials(req) is None


def test_anthropic_without_credentials_fails_before_launching(monkeypatch):
    import claude_agent_sdk

    from agent_sdk_wrapper import Agent, ProviderNotAvailableError
    from agent_sdk_wrapper.events import Error
    from agent_sdk_wrapper.providers.anthropic_provider import _API_KEY_ENV, _PROVIDER_FLAG_ENV

    for name in (*_API_KEY_ENV, *_PROVIDER_FLAG_ENV):
        monkeypatch.delenv(name, raising=False)

    def fail_query(**kwargs):
        raise AssertionError("the runtime must not start")

    monkeypatch.setattr(claude_agent_sdk, "query", fail_query)
    events, _ = [], None

    async def collect():
        req = RunRequest(provider="anthropic", prompt="x", env={"ANTHROPIC_API_KEY": ""})
        return [event async for event in AnthropicProvider().stream(req)]

    events = asyncio.run(collect())
    assert len(events) == 1 and isinstance(events[0], Error)
    assert events[0].error_type == "authentication_failed"

    with pytest.raises(ProviderNotAvailableError, match="stored claude.ai login"):
        Agent(provider="anthropic", env={"ANTHROPIC_API_KEY": ""}).check_runtime()


def test_anthropic_keeps_a_finished_answer_when_the_runtime_then_fails(monkeypatch):
    import claude_agent_sdk
    from claude_agent_sdk import ProcessError, TextBlock

    from agent_sdk_wrapper import Agent

    calls = []

    async def fake_query(*, prompt, options):
        calls.append(prompt)
        yield _assistant(TextBlock(text="final answer"), message_id="m1")
        options.stderr("API Error: 529 overloaded_error")
        raise ProcessError("Command failed with exit code 1", exit_code=1)

    monkeypatch.setattr(claude_agent_sdk, "query", fake_query)
    result = asyncio.run(Agent(provider="anthropic", max_retries=2).run("x"))

    assert result.final_text == "final answer"
    assert result.status == "failure"
    assert len(calls) == 1


def test_anthropic_hook_stops_are_successful_runs():
    from agent_sdk_wrapper.providers.anthropic_provider import _result_error

    for reason in ("hook_stopped", "stop_hook_prevented", "tool_deferred"):
        message = _result(terminal_reason=reason, result="I updated the billing page.")
        assert _result_error(message) is None, reason


def test_anthropic_a_recovered_api_error_does_not_classify_a_later_failure(monkeypatch):
    from claude_agent_sdk import TextBlock

    from agent_sdk_wrapper.events import Error

    events, _ = _stream(
        monkeypatch,
        [
            _assistant(
                TextBlock(text="API Error: Rate limit reached"),
                model="<synthetic>",
                error="rate_limit",
            ),
            _assistant(TextBlock(text="working"), message_id="m2"),
            _result(is_error=True, api_error_status=400, result="Credit balance is too low"),
        ],
    )
    error = next(event for event in events if isinstance(event, Error))
    assert (error.error_type, error.retryable) == ("billing_error", False)


@pytest.mark.parametrize(
    "flag",
    [
        "CLAUDE_CODE_USE_ANTHROPIC_AWS",
        "CLAUDE_CODE_USE_ANTHROPIC_GOOGLE_CLOUD",
        "CLAUDE_CODE_USE_MANTLE",
        "CLAUDE_CODE_USE_GATEWAY",
    ],
)
def test_anthropic_accepts_every_cloud_provider_flag(monkeypatch, flag):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    req = RunRequest(provider="anthropic", prompt="x", env={flag: "1"})
    assert AnthropicProvider().check_credentials(req) is None


def test_anthropic_stops_at_the_first_result(monkeypatch):
    from agent_sdk_wrapper.events import Usage

    usage = {"input_tokens": 100, "output_tokens": 1}
    events, _ = _stream(
        monkeypatch,
        [_result(usage=usage, session_id=""), _result(usage=usage, session_id="")],
    )
    assert len([event for event in events if isinstance(event, Usage)]) == 1


def test_anthropic_stream_without_a_result_is_a_protocol_error(monkeypatch):
    import claude_agent_sdk
    from claude_agent_sdk import TextBlock

    from agent_sdk_wrapper.events import Error, Text

    async def fake_query(*, prompt, options):
        yield _assistant(TextBlock(text="partial"), message_id="m1")

    monkeypatch.setattr(claude_agent_sdk, "query", fake_query)

    async def collect():
        req = RunRequest(provider="anthropic", prompt="x")
        return [event async for event in AnthropicProvider().stream(req)]

    events = asyncio.run(collect())
    assert [type(event) for event in events] == [Text, Error]
    assert events[-1].error_type == "provider_protocol_error"


def test_anthropic_extra_options_may_set_keys_whose_options_are_unused():
    options = AnthropicProvider()._build_options(
        RunRequest(
            provider="anthropic",
            prompt="x",
            extra_options={"setting_sources": ["project"], "output_format": {"type": "text"}},
        )
    )
    assert options.setting_sources == ["project"]
    assert options.output_format == {"type": "text"}


@pytest.mark.parametrize(("value", "accepted"), [("1", True), ("on", True), ("no", False)])
def test_anthropic_provider_flags_count_only_when_the_cli_enables_them(
    monkeypatch, value, accepted
):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    req = RunRequest(provider="anthropic", prompt="x", env={"CLAUDE_CODE_USE_VERTEX": value})
    assert (AnthropicProvider().check_credentials(req) is None) is accepted


def test_anthropic_rejects_unknown_extra_options_before_running():
    with pytest.raises(ConfigError, match="not Claude Agent SDK options"):
        AnthropicProvider().validate_request(
            RunRequest(provider="anthropic", prompt="x", extra_options={"max_budget": 1})
        )


def test_anthropic_stderr_reaches_the_terminal_without_a_user_callback(capsys):
    import collections

    tail = collections.deque()
    options = AnthropicProvider()._build_options(
        RunRequest(provider="anthropic", prompt="x"), tail
    )
    options.stderr("MCP server failed to start")
    assert list(tail) == ["MCP server failed to start"]
    assert "MCP server failed to start" in capsys.readouterr().err


def test_anthropic_retraction_still_reports_usage(monkeypatch):
    from claude_agent_sdk import SystemMessage, TextBlock

    from agent_sdk_wrapper.events import Error, Usage

    events, _ = _stream(
        monkeypatch,
        [
            _assistant(TextBlock(text="partial"), message_id="m1"),
            SystemMessage(
                subtype="model_refusal_fallback", data={"retracted_message_uuids": ["m1"]}
            ),
            _assistant(TextBlock(text="fallback"), message_id="m2"),
            _result(
                usage={"input_tokens": 5, "output_tokens": 1}, total_cost_usd=0.5, session_id=""
            ),
        ],
    )
    assert [type(event) for event in events][-2:] == [Error, Usage]
    assert events[-1].cost_usd == 0.5


def test_anthropic_status_frames_do_not_split_a_message(monkeypatch):
    from claude_agent_sdk import RateLimitEvent, RateLimitInfo, TextBlock

    from agent_sdk_wrapper.events import Text

    events, _ = _stream(
        monkeypatch,
        [
            _assistant(TextBlock(text="The answer "), message_id="m1"),
            RateLimitEvent(
                rate_limit_info=RateLimitInfo(status="allowed_warning"),
                uuid="r",
                session_id="s",
            ),
            _assistant(TextBlock(text="is 42."), message_id="m1"),
        ],
    )
    assert [event.text for event in events if isinstance(event, Text)] == ["The answer is 42."]


def test_anthropic_cancellation_outranks_a_refusal_and_duplicates_are_dropped(monkeypatch):
    from claude_agent_sdk import SystemMessage, TextBlock

    from agent_sdk_wrapper.events import SessionInfo, Text
    from agent_sdk_wrapper.providers.anthropic_provider import _result_error

    error = _result_error(_result(stop_reason="refusal", terminal_reason="aborted_streaming"))
    assert error is not None and error.error_type == "cancelled"

    frame = _assistant(TextBlock(text="once"), message_id="m1", uuid="u1")
    events, _ = _stream(
        monkeypatch,
        [
            SystemMessage(subtype="status", data={"session_id": "s1"}),
            SystemMessage(subtype="init", data={"session_id": "s1", "model": "claude-x"}),
            frame,
            frame,
        ],
    )
    assert [e.text for e in events if isinstance(e, Text)] == ["once"]
    assert [e.model for e in events if isinstance(e, SessionInfo)] == [None, "claude-x"]
