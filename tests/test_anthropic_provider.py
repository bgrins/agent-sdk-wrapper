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
