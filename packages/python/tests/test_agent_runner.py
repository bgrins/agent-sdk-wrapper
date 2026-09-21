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


def trace_events(path):
    return [json.loads(line)["event"] for line in path.read_text().splitlines()]


async def hangs_after_an_error(req):
    yield Error(message="bad key", error_type="authentication_failed")
    await asyncio.Event().wait()


def test_a_deadline_after_a_provider_error_keeps_that_error(monkeypatch, tmp_path):
    install_fake_providers(monkeypatch, events=hangs_after_an_error)

    result = Agent(provider="openai", timeout=0.2, artifacts_dir=tmp_path).run_sync("hi")

    assert (result.status, result.error, result.error_type) == (
        RunStatus.FAILURE,
        "bad key",
        "authentication_failed",
    )
    saved = json.loads((tmp_path / "result.json").read_text())
    assert (saved["status"], saved["error_type"]) == ("failure", "authentication_failed")
    assert trace_events(tmp_path / "trace.jsonl")[-1]["status"] == "failure"


def test_cancelling_after_a_provider_error_keeps_that_error(monkeypatch, tmp_path):
    install_fake_providers(monkeypatch, events=hangs_after_an_error)
    agent = Agent(provider="openai", artifacts_dir=tmp_path)

    async def cancel_run() -> None:
        task = asyncio.create_task(agent.run("hi"))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_run())

    saved = json.loads((tmp_path / "result.json").read_text())
    assert (saved["status"], saved["error"], saved["error_type"]) == (
        "failure",
        "bad key",
        "authentication_failed",
    )
    assert [event["type"] for event in trace_events(tmp_path / "trace.jsonl")] == [
        "run_started",
        "error",
        "run_finished",
    ]


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


def assert_failed_artifacts(artifacts_dir) -> list[dict]:
    saved = json.loads((artifacts_dir / "result.json").read_text())
    manifest = json.loads((artifacts_dir / "manifest.json").read_text())
    trace = trace_events(artifacts_dir / "trace.jsonl")
    assert (saved["status"], manifest["status"]) == ("failure", "failure")
    assert (trace[-1]["type"], trace[-1]["status"]) == ("run_finished", "failure")
    return trace


def test_a_failed_trace_write_fails_the_run(monkeypatch, tmp_path):
    from agent_sdk_wrapper.logging import TraceWriter

    install_fake_providers(monkeypatch, events=[Text(text="hi")])
    write = TraceWriter.write

    def fail_on_text(self, env):
        if isinstance(env.event, Text):
            raise OSError("disk full")
        write(self, env)

    monkeypatch.setattr(TraceWriter, "write", fail_on_text)

    with pytest.raises(OSError, match="disk full"):
        Agent(provider="openai", artifacts_dir=tmp_path).run_sync("hi")

    assert assert_failed_artifacts(tmp_path)[-2]["message"] == "OSError: disk full"


def test_a_non_event_from_the_provider_fails_the_run(monkeypatch, tmp_path):
    install_fake_providers(monkeypatch, events=[Text(text="a"), "not an event", Text(text="b")])

    with pytest.raises(TypeError, match="not an event"):
        Agent(provider="openai", artifacts_dir=tmp_path).run_sync("hi")

    trace = assert_failed_artifacts(tmp_path)
    assert [event["type"] for event in trace] == ["run_started", "text", "error", "run_finished"]
    lines = (tmp_path / "trace.jsonl").read_text().splitlines()
    assert [json.loads(line)["sequence"] for line in lines] == [0, 1, 2, 3]


def test_system_exit_from_on_event_fails_the_run(monkeypatch, tmp_path):
    install_fake_providers(monkeypatch, events=[Text(text="a")])

    def exit_on_text(env):
        if isinstance(env.event, Text):
            raise SystemExit(3)

    with pytest.raises(SystemExit):
        Agent(provider="openai", artifacts_dir=tmp_path, on_event=exit_on_text).run_sync("hi")

    assert assert_failed_artifacts(tmp_path)[-2]["message"] == "SystemExit: 3"


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
