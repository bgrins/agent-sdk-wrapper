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

    # Anthropic reports input net of cache; the wrapper publishes the full
    # prompt count so both providers mean the same thing by input_tokens.
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
    result = await Agent(provider="anthropic").run("Count main and subagent usage")
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


@pytest.mark.parametrize("status", [429, 503, 529, None])
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
            data={},
            task_id="task-1",
            description="review the diff",
            uuid="u1",
            session_id="sess-1",
            task_type="reviewer",
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
    """A failing status must not be labelled with how generation happened to end.

    A real 400 arrives with ``stop_reason='stop_sequence'``; reporting that as
    the error type hides the actual failure behind an unrelated, healthy-looking
    label.
    """
    from agent_sdk_wrapper.providers.anthropic_provider import _result_error

    error = _result_error(
        _result(is_error=True, api_error_status=400, stop_reason="stop_sequence")
    )

    assert error is not None
    assert error.error_type == "api_error_400"
    assert error.retryable is False


def test_anthropic_error_type_falls_back_to_an_error_subtype():
    from agent_sdk_wrapper.providers.anthropic_provider import _result_error

    error = _result_error(
        _result(subtype="error_during_execution", is_error=True, stop_reason="end_turn")
    )

    assert error is not None
    assert error.error_type == "error_during_execution"
