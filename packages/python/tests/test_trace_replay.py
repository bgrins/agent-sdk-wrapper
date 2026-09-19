"""Replay committed trace fixtures through fake providers."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import get_args

import pytest

from agent_sdk_wrapper import (
    Agent,
    AgentEvent,
    AgentUpdated,
    ContextCompacted,
    Error,
    RunEndedReason,
    RunFinished,
    RunStarted,
    RunStatus,
    SessionInfo,
    StructuredOutput,
    SubagentEnded,
    SubagentStarted,
    Text,
    Thinking,
    TokenUsage,
    ToolCall,
    ToolResult,
    Usage,
    WarningEvent,
    event_from_dict,
    install_fake_providers,
    load_trace_replay,
    run_result_summary,
)

ROOT = Path(__file__).resolve().parents[1]
TRACE_FIXTURES = ROOT / "tests" / "fixtures" / "traces"
SAMPLE_EVENTS = [
    RunStarted(provider="openai", model="gpt-5", cwd="/w", prompt="p", system_prompt="s"),
    Text(text="hi", raw={"k": 1}),
    Thinking(text="", redacted_bytes=12),
    ToolCall(id="t1", name="echo", input={"x": 1}),
    ToolResult(id="t1", name="echo", output="out", is_error=True),
    AgentUpdated(name="reviewer"),
    SubagentStarted(task_id="a1", name="reviewer", description="Review"),
    SubagentEnded(task_id="a1", status="completed", summary="done"),
    Usage(usage=TokenUsage(input_tokens=3, output_tokens=2, total_tokens=5), cost_usd=0.5),
    SessionInfo(id="s1", model="claude-haiku-4-5-20251001"),
    StructuredOutput(value={"ok": True}),
    ContextCompacted(trigger="auto", pre_tokens=100),
    WarningEvent(message="slow"),
    Error(message="bad", error_type="max_turns", retryable=True),
    RunFinished(status=RunStatus.FAILURE, duration_ms=5, ended_reason=RunEndedReason.MAX_TURNS),
]


def test_event_from_dict_round_trips_every_event_type() -> None:
    assert {type(event) for event in SAMPLE_EVENTS} == set(get_args(AgentEvent))
    for event in SAMPLE_EVENTS:
        assert event_from_dict(event.to_dict()) == event


def test_replay_summary_uses_last_text_and_reported_model(monkeypatch, tmp_path) -> None:
    install_fake_providers(
        monkeypatch,
        events=[
            SessionInfo(id="s1", model="gpt-5-2026"),
            Text(text="Checking."),
            Text(text="Done."),
        ],
    )
    trace = tmp_path / "trace.jsonl"
    Agent(provider="openai", model="gpt-5", trace_file=trace).run_sync("hi")
    replay = load_trace_replay(trace)
    install_fake_providers(monkeypatch, events=replay.events)

    result = Agent(provider="openai", model="gpt-5").run_sync(replay.prompt)

    assert replay.expected["final_text"] == "Done."
    assert replay.expected["model"] == "gpt-5-2026"
    assert run_result_summary(result) == replay.expected


@pytest.mark.parametrize(
    "trace_path",
    sorted(TRACE_FIXTURES.glob("*.trace.jsonl")),
    ids=lambda path: path.name,
)
def test_trace_fixture_replays_to_expected_result(monkeypatch, trace_path: Path) -> None:
    replay = load_trace_replay(trace_path)
    install_fake_providers(
        monkeypatch,
        events=replay.events,
        providers=[replay.provider],
    )

    result = asyncio.run(
        Agent(
            provider=replay.provider,
            model=replay.model,
            cwd=replay.cwd,
            system_prompt=replay.system_prompt,
            max_retries=0,
        ).run(replay.prompt)
    )

    assert run_result_summary(result) == replay.expected
