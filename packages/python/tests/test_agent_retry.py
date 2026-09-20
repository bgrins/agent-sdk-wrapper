"""Retry/backoff behavior for the unified Agent."""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from agent_sdk_wrapper import (
    Agent,
    AgentUpdated,
    ContextCompacted,
    Error,
    EventEnvelope,
    ProcessTerminatedError,
    RunFinished,
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
    TransientError,
    Usage,
    WarningEvent,
    install_fake_providers,
)
from agent_sdk_wrapper import agent as agent_mod
from agent_sdk_wrapper.providers import base
from agent_sdk_wrapper.providers import openai_provider as op_mod


@pytest.fixture
def no_backoff(monkeypatch):
    monkeypatch.setattr(agent_mod, "_backoff", lambda attempt: 0)


def attempts(*scripts):
    """Return an event factory that plays one script per attempt.

    A script item that is an exception is raised; anything else is yielded.
    """

    seen = []

    async def play(req):
        seen.append((req.attempt, req.session_id))
        for item in scripts[len(seen) - 1]:
            if isinstance(item, BaseException):
                raise item
            yield item

    return play, seen


async def collect(stream) -> list[EventEnvelope]:
    return [env async for env in stream]


def event_types(envelopes: list[EventEnvelope]) -> list[str]:
    return [env.event.type for env in envelopes]


class FlakyProvider(base.ProviderAdapter):
    """Fails the first N attempts with TransientError, then succeeds."""

    name = "openai"
    fail_n = 1
    calls = 0

    async def stream(self, req):  # type: ignore[override]
        FlakyProvider.calls += 1
        if FlakyProvider.calls <= FlakyProvider.fail_n:
            raise TransientError("rate limit (synthetic)")
        yield Text(text="ok")


def test_run_retries_transient_then_succeeds(monkeypatch, no_backoff):
    monkeypatch.setattr(op_mod, "OpenAIProvider", FlakyProvider)
    FlakyProvider.calls = 0
    FlakyProvider.fail_n = 2

    agent = Agent(provider="openai", max_retries=3)
    result = asyncio.run(agent.run("hi"))

    assert FlakyProvider.calls == 3
    assert result.status == RunStatus.SUCCESS
    assert result.final_text == "ok"


def test_run_gives_up_after_max_retries(monkeypatch, no_backoff):
    monkeypatch.setattr(op_mod, "OpenAIProvider", FlakyProvider)
    FlakyProvider.calls = 0
    FlakyProvider.fail_n = 99

    agent = Agent(provider="openai", max_retries=1)
    result = asyncio.run(agent.run("hi"))

    assert FlakyProvider.calls == 2  # initial + 1 retry
    assert result.status == RunStatus.FAILURE
    assert "rate limit" in (result.error or "").lower()


def test_raise_on_error(monkeypatch):
    monkeypatch.setattr(op_mod, "OpenAIProvider", FlakyProvider)
    FlakyProvider.calls = 0
    FlakyProvider.fail_n = 99

    agent = Agent(provider="openai", max_retries=0, raise_on_error=True)
    from agent_sdk_wrapper import RunFailedError

    with pytest.raises(RunFailedError):
        asyncio.run(agent.run("hi"))


def test_stream_retries_transient_error(monkeypatch, no_backoff):
    play, seen = attempts([TransientError("rate limit")], [Text(text="ok")])
    install_fake_providers(monkeypatch, events=play)

    envelopes = asyncio.run(collect(Agent(provider="openai", max_retries=1).stream("hi")))

    assert len(seen) == 2
    assert event_types(envelopes) == ["run_started", "warning", "text", "run_finished"]
    assert "rate limit" in envelopes[1].event.message
    assert envelopes[-1].event.status == RunStatus.SUCCESS


def test_retryable_error_event_is_replaced_by_warning(monkeypatch, no_backoff):
    overloaded = Error(message="overloaded (529)", error_type="transient_api_error", retryable=True)
    play, seen = attempts(
        [
            SessionInfo(id="failed-thread"),
            Usage(usage=TokenUsage(input_tokens=5, total_tokens=5), cost_usd=0.01),
            overloaded,
        ],
        [SessionInfo(id="good-thread"), Text(text="ok")],
    )
    install_fake_providers(monkeypatch, events=play)
    agent = Agent(provider="openai", continue_session=True, max_retries=2)

    result = asyncio.run(agent.run("hi"))

    assert result.ok
    assert seen == [(0, None), (1, None)]
    assert event_types(result.events) == [
        "run_started",
        "session_info",
        "usage",
        "warning",
        "session_info",
        "text",
        "run_finished",
    ]
    assert "overloaded (529)" in result.events[3].event.message
    assert result.usage is not None and result.usage.input_tokens == 5
    assert result.cost_usd == pytest.approx(0.01)
    assert result.session_id == agent.session_id == "good-thread"


def test_a_resumed_session_is_not_retried_once_it_started(monkeypatch, no_backoff):
    overloaded = Error(message="overloaded", error_type="transient_api_error", retryable=True)
    play, seen = attempts([SessionInfo(id="original"), overloaded], [Text(text="again")])
    install_fake_providers(monkeypatch, events=play)

    result = asyncio.run(Agent(provider="openai", session_id="original", max_retries=2).run("hi"))

    assert len(seen) == 1
    assert result.error == "overloaded"


def test_a_fatal_exception_after_a_held_error_is_recorded_not_retried(monkeypatch, no_backoff):
    from agent_sdk_wrapper import ProviderNotAvailableError

    overloaded = Error(message="overloaded", error_type="transient_api_error", retryable=True)
    play, seen = attempts([overloaded, ProviderNotAvailableError("runtime crashed")], [])
    install_fake_providers(monkeypatch, events=play)

    result = asyncio.run(Agent(provider="openai", max_retries=2).run("hi"))

    assert len(seen) == 1
    errors = [env.event for env in result.events if isinstance(env.event, Error)]
    assert [e.error_type for e in errors] == ["transient_api_error", "runtime_unavailable"]


def test_a_stream_method_that_raises_is_a_failed_run(monkeypatch):
    from agent_sdk_wrapper import FakeProvider

    class Broken(FakeProvider):
        def stream(self, req):
            raise RuntimeError("adapter bug")

    install_fake_providers(monkeypatch, fake=Broken())
    result = asyncio.run(Agent(provider="openai").run("hi"))

    assert result.status == RunStatus.FAILURE
    assert "adapter bug" in (result.error or "")


def test_runs_leave_retries_to_the_runtime_by_default(monkeypatch, no_backoff):
    overloaded = Error(message="overloaded", error_type="transient_api_error", retryable=True)
    play, seen = attempts([overloaded], [Text(text="again")])
    install_fake_providers(monkeypatch, events=play)

    result = asyncio.run(Agent(provider="openai").run("hi"))

    assert len(seen) == 1
    assert result.error == "overloaded"


def test_retryable_error_is_emitted_when_retries_are_exhausted(monkeypatch, no_backoff):
    overloaded = Error(message="overloaded", error_type="transient_api_error", retryable=True)
    play, seen = attempts([overloaded], [overloaded])
    install_fake_providers(monkeypatch, events=play)

    result = asyncio.run(Agent(provider="openai", max_retries=1).run("hi"))

    assert len(seen) == 2
    assert event_types(result.events) == ["run_started", "warning", "error", "run_finished"]
    assert result.status == RunStatus.FAILURE
    assert result.error == "overloaded"


def test_retryable_error_before_progress_keeps_order_and_is_not_retried(monkeypatch):
    overloaded = Error(message="overloaded", error_type="transient_api_error", retryable=True)
    play, seen = attempts([overloaded, Text(text="partial")])
    install_fake_providers(monkeypatch, events=play)

    result = asyncio.run(Agent(provider="openai", max_retries=3).run("hi"))

    assert len(seen) == 1
    assert event_types(result.events) == ["run_started", "error", "text", "run_finished"]
    assert result.status == RunStatus.FAILURE


@pytest.mark.parametrize(
    "event",
    [
        Text(text="partial"),
        Thinking(text="plan"),
        ToolCall(id="t1", name="echo"),
        ToolResult(id="t1", output="out"),
        StructuredOutput(value={"ok": True}),
        SubagentStarted(task_id="a1", name="reviewer"),
        SubagentEnded(task_id="a1", status="completed"),
        ContextCompacted(trigger="auto"),
        AgentUpdated(name="reviewer"),
    ],
    ids=lambda event: event.type,
)
def test_progress_events_prevent_retry(monkeypatch, no_backoff, event):
    play, seen = attempts([event, TransientError("dropped")], [Text(text="retried")])
    install_fake_providers(monkeypatch, events=play)

    result = asyncio.run(Agent(provider="openai", max_retries=1).run("hi"))

    assert len(seen) == 1
    assert result.status == RunStatus.FAILURE
    assert result.error == "dropped"


@pytest.mark.parametrize(
    "event",
    [SessionInfo(id="s1"), WarningEvent(message="slow"), Usage(usage=TokenUsage())],
    ids=lambda event: event.type,
)
def test_non_progress_events_allow_retry(monkeypatch, no_backoff, event):
    play, seen = attempts([event, TransientError("dropped")], [Text(text="retried")])
    install_fake_providers(monkeypatch, events=play)

    result = asyncio.run(Agent(provider="openai", max_retries=1).run("hi"))

    assert len(seen) == 2
    assert result.ok
    assert result.final_text == "retried"


def test_retry_backoff_is_bounded_by_timeout(monkeypatch):
    monkeypatch.setattr(agent_mod, "_backoff", lambda attempt: 3)
    play, _ = attempts([TransientError("rate limit")], [Text(text="too late")])
    install_fake_providers(monkeypatch, events=play)

    started = time.monotonic()
    result = asyncio.run(Agent(provider="openai", max_retries=1, timeout=0.05).run("hi"))

    assert time.monotonic() - started < 2
    assert result.status == RunStatus.TIMEOUT


@pytest.mark.parametrize("mode", ["run", "stream"])
def test_process_terminated_is_recorded_then_raised(monkeypatch, tmp_path, mode):
    play, seen = attempts([ProcessTerminatedError(9)], [Text(text="unreachable")])
    install_fake_providers(monkeypatch, events=play)
    agent = Agent(provider="openai", max_retries=2, artifacts_dir=tmp_path)
    streamed: list[EventEnvelope] = []

    async def consume() -> None:
        if mode == "run":
            await agent.run("hi")
        else:
            async for env in agent.stream("hi"):
                streamed.append(env)

    with pytest.raises(ProcessTerminatedError):
        asyncio.run(consume())

    trace = [
        json.loads(line)["event"]
        for line in (tmp_path / "trace.jsonl").read_text().splitlines()
    ]
    assert len(seen) == 1
    assert [event["type"] for event in trace] == ["run_started", "error", "run_finished"]
    assert trace[1]["error_type"] == "process_terminated"
    assert trace[2]["status"] == "failure"
    assert json.loads((tmp_path / "result.json").read_text())["status"] == "failure"
    if mode == "stream":
        assert [env.event.type for env in streamed] == ["run_started", "error", "run_finished"]
        assert isinstance(streamed[-1].event, RunFinished)


async def test_provider_cleanup_error_at_a_deadline_is_logged_not_escaped(
    monkeypatch, caplog
):
    answered = asyncio.Event()

    async def play(req):
        try:
            yield Text(text="answer")
            answered.set()
            await asyncio.sleep(10)
        finally:
            raise RuntimeError("cleanup broke")

    install_fake_providers(monkeypatch, events=play)
    result = await Agent(provider="openai", timeout=0.5).run("x")

    assert answered.is_set(), "the deadline must fall after the first event"
    assert result.status == RunStatus.TIMEOUT
    assert [env.event.type for env in result.events][-3:] == ["text", "error", "run_finished"]
    assert "cleanup broke" in caplog.text


async def test_closing_a_stream_with_a_held_retryable_error_is_cancelled(
    monkeypatch, tmp_path
):
    gate = asyncio.Event()

    async def play(req):
        yield Error(message="overloaded", error_type="transient_api_error", retryable=True)
        yield Usage(usage=TokenUsage(input_tokens=1, total_tokens=1))
        await gate.wait()

    install_fake_providers(monkeypatch, events=play)
    trace = tmp_path / "trace.jsonl"
    stream = Agent(provider="openai", max_retries=1, trace_file=trace).stream("x")
    async for env in stream:
        if env.event.type == "usage":
            break
    await stream.aclose()

    events = [json.loads(line)["event"] for line in trace.read_text().splitlines()]
    assert [(e["type"], e.get("error_type")) for e in events] == [
        ("run_started", None),
        ("usage", None),
        ("warning", None),
        ("error", "cancelled"),
        ("run_finished", None),
    ]
    assert events[-1]["status"] == "cancelled"
