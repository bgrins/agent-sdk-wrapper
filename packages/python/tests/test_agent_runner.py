"""Run outcomes the unified Agent records: errors, kills and deadlines."""

from __future__ import annotations

import asyncio
import json

import pytest

from agent_sdk_wrapper import (
    Agent,
    Error,
    EventEnvelope,
    ProcessTerminatedError,
    RunFailedError,
    RunFinished,
    RunStatus,
    Text,
    TransientError,
    install_fake_providers,
)


def script(*items):
    """Return an event factory that yields items, raising any that are exceptions."""

    calls = []

    async def play(req):
        calls.append(req)
        for item in items:
            if isinstance(item, BaseException):
                raise item
            yield item

    return play, calls


def test_a_transient_failure_ends_the_run_with_its_type(monkeypatch):
    play, calls = script(TransientError("rate limit"))
    install_fake_providers(monkeypatch, events=play)

    result = asyncio.run(Agent(provider="openai").run("hi"))

    assert len(calls) == 1
    assert result.status == RunStatus.FAILURE
    assert (result.error, result.error_type) == ("rate limit", "transient_api_error")


def test_the_first_provider_error_sets_the_result_error(monkeypatch):
    play, _ = script(
        Text(text="partial"),
        Error(message="overloaded", error_type="transient_api_error"),
        Error(message="later", error_type="execution_error"),
    )
    install_fake_providers(monkeypatch, events=play)

    result = asyncio.run(Agent(provider="openai").run("hi"))

    assert (result.error, result.error_type) == ("overloaded", "transient_api_error")
    assert result.final_text == "partial"
    assert result.to_dict()["error_type"] == "transient_api_error"


def test_raise_on_error(monkeypatch):
    play, _ = script(TransientError("rate limit"))
    install_fake_providers(monkeypatch, events=play)

    with pytest.raises(RunFailedError):
        asyncio.run(Agent(provider="openai", raise_on_error=True).run("hi"))


def test_a_stream_method_that_raises_is_a_failed_run(monkeypatch):
    from agent_sdk_wrapper import FakeProvider

    class Broken(FakeProvider):
        def stream(self, req):
            raise RuntimeError("adapter bug")

    install_fake_providers(monkeypatch, fake=Broken())
    result = asyncio.run(Agent(provider="openai").run("hi"))

    assert result.status == RunStatus.FAILURE
    assert "adapter bug" in (result.error or "")


@pytest.mark.parametrize("mode", ["run", "stream"])
def test_process_terminated_is_recorded_then_raised(monkeypatch, tmp_path, mode):
    play, _ = script(ProcessTerminatedError(9), Text(text="unreachable"))
    install_fake_providers(monkeypatch, events=play)
    agent = Agent(provider="openai", artifacts_dir=tmp_path)
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
    assert [event["type"] for event in trace] == ["run_started", "error", "run_finished"]
    assert trace[1]["error_type"] == "process_terminated"
    assert trace[2]["status"] == "failure"
    saved = json.loads((tmp_path / "result.json").read_text())
    assert (saved["status"], saved["error_type"]) == ("failure", "process_terminated")
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
    assert (result.status, result.error_type) == (RunStatus.TIMEOUT, "timeout")
    assert [env.event.type for env in result.events][-3:] == ["text", "error", "run_finished"]
    assert "cleanup broke" in caplog.text
