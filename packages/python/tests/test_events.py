"""Tests for the event model and result aggregation."""

from __future__ import annotations

import json

import pytest

from agent_sdk_wrapper import (
    INHERIT_MODEL,
    Agent,
    ConfigError,
    Error,
    EventEnvelope,
    EventFactory,
    EventSource,
    FakeProvider,
    ProviderEventEnvelope,
    ProviderNotAvailableError,
    RunEndedReason,
    RunFinished,
    RunRequest,
    RunResult,
    RunStatus,
    SessionInfo,
    StructuredOutput,
    SubagentDef,
    Text,
    TokenUsage,
    ToolCall,
    TransientError,
    Usage,
    install_fake_providers,
    normalize_builtin_tools,
    normalize_effort_for_provider,
    parse_model_spec,
)
from agent_sdk_wrapper.events import _jsonable


def test_event_to_dict_drops_none_and_tags_type():
    delta = Text(text="hi")
    assert delta.to_dict() == {"type": "text", "text": "hi"}


def test_envelope_roundtrips_json():
    env = EventEnvelope(
        run_id="abc",
        sequence=0,
        timestamp="2026-05-26T00:00:00+00:00",
        event=Text(text="hi"),
    )
    parsed = json.loads(env.to_json())
    assert parsed["run_id"] == "abc"
    assert parsed["event"] == {"type": "text", "text": "hi"}


def test_token_usage_addition():
    a = TokenUsage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15)
    b = TokenUsage(requests=2, input_tokens=2, output_tokens=3, total_tokens=5)
    s = a + b
    assert (s.requests, s.input_tokens, s.output_tokens, s.total_tokens) == (3, 12, 8, 20)


def test_jsonable_handles_pydantic_dataclass_enum():
    from pydantic import BaseModel

    class M(BaseModel):
        x: int

    assert _jsonable(M(x=3)) == {"x": 3}
    assert _jsonable(TokenUsage(input_tokens=1)) == {
        "requests": 0,
        "input_tokens": 1,
        "output_tokens": 0,
        "total_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_output_tokens": 0,
    }
    assert _jsonable(RunStatus.SUCCESS) == "success"


def test_run_finished_and_result_default_to_unknown_ended_reason():
    assert RunFinished().ended_reason == RunEndedReason.UNKNOWN
    assert RunResult("run", "openai", RunStatus.FAILURE).ended_reason == (
        RunEndedReason.UNKNOWN
    )


def test_fake_provider_accepts_exported_event_factory():
    source: EventSource = [Text(text="factory response")]

    def factory(req: RunRequest) -> EventSource:
        return source

    factory_ref: EventFactory = factory
    provider = FakeProvider(factory_ref)

    import asyncio

    async def collect():
        return [
            event
            async for event in provider.stream(
                RunRequest(provider="openai", prompt="ignored")
            )
        ]

    events = asyncio.run(collect())

    assert events == [Text(text="factory response")]


def test_normalize_builtin_tools():
    assert normalize_builtin_tools(None) is None
    assert normalize_builtin_tools("none") == "none"
    assert normalize_builtin_tools([]) == "none"
    assert normalize_builtin_tools(("Read", "Grep")) == ["Read", "Grep"]

    with pytest.raises(ConfigError, match="must be 'none'"):
        normalize_builtin_tools("Read")


def test_normalize_effort_for_provider():
    assert normalize_effort_for_provider("anthropic", "HIGH") == "high"
    assert normalize_effort_for_provider("anthropic", "max") == "max"
    assert normalize_effort_for_provider("openai", "max") == "xhigh"
    assert normalize_effort_for_provider("openai", "minimal") == "minimal"

    with pytest.raises(ConfigError, match="not supported"):
        normalize_effort_for_provider("anthropic", "minimal")


def test_unknown_provider_raises():
    from agent_sdk_wrapper import ConfigError

    with pytest.raises(ConfigError):
        Agent(provider="bogus")  # type: ignore[arg-type]


def test_unknown_run_override_raises(monkeypatch):
    install_fake_providers(monkeypatch)
    agent = Agent(provider="openai")

    with pytest.raises(ConfigError, match="web_toolz"):
        agent.run_sync("hi", web_toolz=False)


def test_unknown_stream_override_raises(monkeypatch):
    install_fake_providers(monkeypatch)
    agent = Agent(provider="openai")

    import asyncio

    async def consume() -> None:
        async for _ in agent.stream("hi", web_toolz=False):
            pass

    with pytest.raises(ConfigError, match="web_toolz"):
        asyncio.run(consume())


def test_on_event_callback_exception_is_logged_not_swallowed(monkeypatch, caplog):
    install_fake_providers(monkeypatch)

    def broken_callback(env):
        raise RuntimeError("callback boom")

    agent = Agent(provider="openai", on_event=broken_callback)

    import asyncio
    import logging

    with caplog.at_level(logging.ERROR, logger="agent_sdk_wrapper"):
        result = asyncio.run(agent.run("hi"))

    assert result.status == RunStatus.SUCCESS  # run continues despite callback failure
    assert any("on_event callback raised" in record.message for record in caplog.records)


def test_on_provider_event_callback_exception_is_logged_not_swallowed(
    monkeypatch, caplog
):
    from agent_sdk_wrapper.artifacts import ProviderEventLogger
    from agent_sdk_wrapper.providers import base
    from agent_sdk_wrapper.providers import openai_provider as op_mod

    class FakeProvider(base.ProviderAdapter):
        name = "openai"

        async def stream(self, req):  # type: ignore[override]
            ProviderEventLogger(
                "openai", req.artifacts_dir, req.on_provider_event
            ).write({"method": "fake/raw"})
            yield Text(text="ok")

    monkeypatch.setattr(op_mod, "OpenAIProvider", FakeProvider)

    def broken_callback(env):
        raise RuntimeError("callback boom")

    agent = Agent(provider="openai", on_provider_event=broken_callback)

    import asyncio
    import logging

    with caplog.at_level(logging.ERROR, logger="agent_sdk_wrapper"):
        result = asyncio.run(agent.run("hi"))

    assert result.status == RunStatus.SUCCESS
    assert any(
        "on_provider_event callback raised" in record.message
        for record in caplog.records
    )


def test_provider_can_be_inferred_from_model():
    assert Agent(model="claude-haiku-4-5").provider == "anthropic"
    assert Agent(model="gpt-5").provider == "openai"
    assert Agent(model="codex:gpt-5").provider == "openai"
    assert Agent(model="codex:gpt-5").model == "gpt-5"
    assert Agent(provider="codex").provider == "openai"
    assert Agent(provider="anthropic", model="gpt-5").provider == "anthropic"
    assert Agent(model="codex:gpt-5", effort="max").effort == "xhigh"

    with pytest.raises(ConfigError, match="not supported"):
        Agent(model="anthropic:claude-haiku-4-5", effort="none")


def test_parse_model_spec_accepts_provider_prefixes():
    assert parse_model_spec("codex:gpt-5") == ("openai", "gpt-5")
    assert parse_model_spec("anthropic:claude-haiku-4-5") == (
        "anthropic",
        "claude-haiku-4-5",
    )
    assert parse_model_spec("gpt-5") == (None, "gpt-5")


def test_provider_model_spec_conflict_raises():
    with pytest.raises(ConfigError, match="conflicts"):
        Agent(provider="anthropic", model="codex:gpt-5")


def test_subagent_model_specs_are_normalized_and_validated():
    agent = Agent(
        provider="openai",
        model="gpt-5",
        subagents={
            "reviewer": SubagentDef(
                description="Review",
                prompt="Review.",
                model="codex:gpt-5-mini",
            ),
            "inherited": SubagentDef(
                description="Inherit",
                prompt="Inherit.",
                model=INHERIT_MODEL,
            ),
        },
    )

    assert agent.subagents["reviewer"].model == "gpt-5-mini"
    assert agent.subagents["inherited"].model is None

    with pytest.raises(ConfigError, match="conflicts"):
        Agent(
            provider="openai",
            subagents={
                "reviewer": SubagentDef(
                    description="Review",
                    prompt="Review.",
                    model="anthropic:claude-haiku-4-5",
                )
            },
        )


def test_provider_inference_requires_known_model():
    from agent_sdk_wrapper import ConfigError

    with pytest.raises(ConfigError, match="could not infer provider"):
        Agent(model="not-a-known-model-family")

    with pytest.raises(ConfigError, match="provider is required"):
        Agent()


def test_aggregate_via_fake_provider(monkeypatch, tmp_path):
    """Drive Agent.run() against a fake provider that yields a known stream."""

    from agent_sdk_wrapper.providers import base
    from agent_sdk_wrapper.providers import openai_provider as op_mod

    class FakeProvider(base.ProviderAdapter):
        name = "openai"

        async def stream(self, req):  # type: ignore[override]
            yield Text(text="Hello ")
            yield Text(text="world")
            yield ToolCall(id="t1", name="echo", input={"x": 1})
            yield SessionInfo(id="sess-1")
            yield Usage(
                usage=TokenUsage(input_tokens=10, output_tokens=2, total_tokens=12),
                cost_usd=0.01,
            )
            yield Usage(
                usage=TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                cost_usd=0.02,
            )
            yield StructuredOutput(value={"ok": True})

    monkeypatch.setattr(op_mod, "OpenAIProvider", FakeProvider)

    trace_file = tmp_path / "trace.jsonl"
    agent = Agent(provider="openai", trace_file=trace_file)

    import asyncio
    result = asyncio.run(agent.run("ignored"))

    assert result.status == RunStatus.SUCCESS
    assert result.final_text == "Hello world"
    assert result.usage and result.usage.total_tokens == 14
    assert result.cost_usd == pytest.approx(0.03)
    assert result.session_id == "sess-1"
    assert result.structured_output == {"ok": True}
    types = [type(e.event).__name__ for e in result.events]
    assert types[0] == "RunStarted"
    assert types[-1] == "RunFinished"
    assert "ToolCall" in types
    # Trace file got every envelope on its own line.
    lines = trace_file.read_text().splitlines()
    assert len(lines) == len(result.events)
    assert all(json.loads(line) for line in lines)


def test_public_fake_provider_helper(monkeypatch):
    seen_requests = []
    install_fake_providers(
        monkeypatch,
        events=lambda req: [
            SessionInfo(id=req.session_id or "fake-session"),
            Text(text="ok"),
        ],
        seen_requests=seen_requests,
    )

    import asyncio

    result = asyncio.run(Agent(provider="openai", continue_session=True).run("ignored"))

    assert result.final_text == "ok"
    assert result.session_id == "fake-session"
    assert seen_requests and seen_requests[0].prompt == "ignored"


def test_check_runtime_uses_provider_adapter(monkeypatch):
    class UnavailableProvider(FakeProvider):
        def ensure_available(self) -> None:
            raise ProviderNotAvailableError("missing runtime")

    install_fake_providers(monkeypatch, fake=UnavailableProvider())

    with pytest.raises(ProviderNotAvailableError, match="missing runtime"):
        Agent(provider="openai").check_runtime()


def test_check_runtime_validates_codex_request_before_runtime_check():
    agent = Agent(provider="openai", builtin_tools="none")

    with pytest.raises(ConfigError, match="builtin_tools"):
        agent.check_runtime()


def test_check_runtime_validates_anthropic_request_before_runtime_check():
    agent = Agent(
        provider="anthropic",
        builtin_tools="none",
        extra_options={"tools": []},
    )

    with pytest.raises(ConfigError, match="builtin_tools"):
        agent.check_runtime()


def test_agent_can_continue_provider_session(monkeypatch):
    from agent_sdk_wrapper.providers import base
    from agent_sdk_wrapper.providers import openai_provider as op_mod

    seen_session_ids: list[str | None] = []

    class FakeProvider(base.ProviderAdapter):
        name = "openai"

        async def stream(self, req):  # type: ignore[override]
            seen_session_ids.append(req.session_id)
            yield SessionInfo(id=req.session_id or "sess-1")
            yield Text(text="ok")

    monkeypatch.setattr(op_mod, "OpenAIProvider", FakeProvider)

    agent = Agent(provider="openai", continue_session=True)

    import asyncio

    first = asyncio.run(agent.run("one"))
    second = asyncio.run(agent.run("two"))

    assert first.session_id == "sess-1"
    assert second.session_id == "sess-1"
    assert agent.session_id == "sess-1"
    assert seen_session_ids == [None, "sess-1"]


def test_dump_context_writes_summary(monkeypatch, tmp_path):
    from agent_sdk_wrapper.providers import base
    from agent_sdk_wrapper.providers import openai_provider as op_mod

    class FakeProvider(base.ProviderAdapter):
        name = "openai"

        async def stream(self, req):  # type: ignore[override]
            yield SessionInfo(id=req.session_id or "sess-context")
            yield Text(text=f"summary:{req.prompt}")

    monkeypatch.setattr(op_mod, "OpenAIProvider", FakeProvider)

    agent = Agent(provider="openai", continue_session=True)

    import asyncio

    path = tmp_path / "context.md"
    result = asyncio.run(agent.dump_context(path, prompt="summarize"))

    assert result.final_text == "summary:summarize"
    assert path.read_text() == "summary:summarize"
    assert agent.session_id == "sess-context"


def test_artifacts_dir_writes_trace_manifest_and_result(monkeypatch, tmp_path):
    from agent_sdk_wrapper.artifacts import ProviderEventLogger
    from agent_sdk_wrapper.providers import base
    from agent_sdk_wrapper.providers import openai_provider as op_mod

    class FakeProvider(base.ProviderAdapter):
        name = "openai"

        async def stream(self, req):  # type: ignore[override]
            ProviderEventLogger(
                "openai", req.artifacts_dir, req.on_provider_event
            ).write(
                {"method": "fake/raw", "payload": {"text": "artifacted"}}
            )
            yield Text(text="artifacted")
            yield SessionInfo(id="sess-artifacts")

    monkeypatch.setattr(op_mod, "OpenAIProvider", FakeProvider)

    artifacts_dir = tmp_path / "artifacts"
    provider_events: list[ProviderEventEnvelope] = []
    agent = Agent(
        provider="openai",
        artifacts_dir=artifacts_dir,
        on_provider_event=provider_events.append,
    )

    import asyncio
    result = asyncio.run(agent.run("ignored"))

    trace_file = artifacts_dir / "trace.jsonl"
    result_file = artifacts_dir / "result.json"
    manifest_file = artifacts_dir / "manifest.json"
    assert result.artifacts_dir == str(artifacts_dir)
    assert trace_file.exists()
    assert result_file.exists()
    manifest = json.loads(manifest_file.read_text())
    assert manifest["schema_version"] == 1
    assert manifest["trace_format"] == "agent-sdk-wrapper.event-envelope-jsonl.v1"
    assert manifest["run_id"] == result.run_id
    assert manifest["status"] == "success"
    assert not (artifacts_dir / "README.txt").exists()
    assert "viewer_hint" not in manifest["files"]
    assert manifest["files"]["trace"] == "trace.jsonl"
    assert manifest["files"]["result"] == "result.json"
    assert manifest["files"]["provider_events"] == "provider-events.jsonl"
    raw_events = [
        json.loads(line)
        for line in (artifacts_dir / "provider-events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert raw_events[0]["message"] == {
        "method": "fake/raw",
        "payload": {"text": "artifacted"},
    }
    assert len(provider_events) == 1
    assert provider_events[0].to_dict() == raw_events[0]
    assert provider_events[0].raw == {
        "method": "fake/raw",
        "payload": {"text": "artifacted"},
    }
    assert not (artifacts_dir / "sdk").exists()
    saved_result = json.loads(result_file.read_text())
    assert saved_result["final_text"] == "artifacted"
    assert saved_result["ended_reason"] == "success"


def test_on_provider_event_runs_without_artifacts_dir(monkeypatch):
    from types import SimpleNamespace

    from agent_sdk_wrapper.artifacts import ProviderEventLogger
    from agent_sdk_wrapper.providers import base
    from agent_sdk_wrapper.providers import openai_provider as op_mod

    raw_message = SimpleNamespace(method="fake/raw", payload={"text": "live"})

    class FakeProvider(base.ProviderAdapter):
        name = "openai"

        async def stream(self, req):  # type: ignore[override]
            ProviderEventLogger(
                "openai", req.artifacts_dir, req.on_provider_event
            ).write(raw_message)
            yield Text(text="live")

    monkeypatch.setattr(op_mod, "OpenAIProvider", FakeProvider)

    provider_events: list[ProviderEventEnvelope] = []
    agent = Agent(provider="openai", on_provider_event=provider_events.append)

    import asyncio

    result = asyncio.run(agent.run("ignored"))

    assert result.final_text == "live"
    assert len(provider_events) == 1
    assert provider_events[0].provider == "openai"
    assert provider_events[0].class_name == "types.SimpleNamespace"
    assert provider_events[0].message == {
        "method": "fake/raw",
        "payload": {"text": "live"},
    }
    assert provider_events[0].raw is raw_message


def test_run_can_omit_events_from_result_while_writing_trace(monkeypatch, tmp_path):
    from agent_sdk_wrapper.providers import base
    from agent_sdk_wrapper.providers import openai_provider as op_mod

    class FakeProvider(base.ProviderAdapter):
        name = "openai"

        async def stream(self, req):  # type: ignore[override]
            yield Text(text="compact")
            yield SessionInfo(id="sess-compact")
            yield Usage(
                usage=TokenUsage(input_tokens=1, output_tokens=2, total_tokens=3),
                cost_usd=0.01,
            )
            yield StructuredOutput(value={"ok": True})

    monkeypatch.setattr(op_mod, "OpenAIProvider", FakeProvider)

    trace_file = tmp_path / "trace.jsonl"

    import asyncio

    result = asyncio.run(
        Agent(
            provider="openai",
            trace_file=trace_file,
            include_events_in_result=False,
        ).run("ignored")
    )

    assert result.final_text == "compact"
    assert result.session_id == "sess-compact"
    assert result.usage and result.usage.total_tokens == 3
    assert result.cost_usd == pytest.approx(0.01)
    assert result.structured_output == {"ok": True}
    assert result.events == []
    assert len(trace_file.read_text().splitlines()) == 6


def test_stream_artifacts_dir_writes_trace_manifest_and_result(monkeypatch, tmp_path):
    from agent_sdk_wrapper.providers import base
    from agent_sdk_wrapper.providers import openai_provider as op_mod

    class FakeProvider(base.ProviderAdapter):
        name = "openai"

        async def stream(self, req):  # type: ignore[override]
            yield Text(text="streamed")
            yield SessionInfo(id="sess-stream")
            yield Usage(usage=TokenUsage(input_tokens=1, output_tokens=2, total_tokens=3))

    monkeypatch.setattr(op_mod, "OpenAIProvider", FakeProvider)

    artifacts_dir = tmp_path / "stream-artifacts"
    agent = Agent(provider="openai", artifacts_dir=artifacts_dir)

    async def collect() -> list[EventEnvelope]:
        return [env async for env in agent.stream("ignored")]

    import asyncio

    events = asyncio.run(collect())

    trace_file = artifacts_dir / "trace.jsonl"
    result_file = artifacts_dir / "result.json"
    manifest_file = artifacts_dir / "manifest.json"
    assert trace_file.exists()
    assert result_file.exists()
    assert len(trace_file.read_text().splitlines()) == len(events)
    manifest = json.loads(manifest_file.read_text())
    assert manifest["status"] == "success"
    assert "viewer_hint" not in manifest["files"]
    assert not (artifacts_dir / "README.txt").exists()
    assert manifest["files"]["trace"] == "trace.jsonl"
    assert manifest["files"]["result"] == "result.json"
    saved_result = json.loads(result_file.read_text())
    assert saved_result["final_text"] == "streamed"
    assert saved_result["ended_reason"] == "success"
    assert saved_result["session_id"] == "sess-stream"
    assert saved_result["usage"]["total_tokens"] == 3


def test_stream_artifacts_can_omit_events_from_result(monkeypatch, tmp_path):
    from agent_sdk_wrapper.providers import base
    from agent_sdk_wrapper.providers import openai_provider as op_mod

    class FakeProvider(base.ProviderAdapter):
        name = "openai"

        async def stream(self, req):  # type: ignore[override]
            yield Text(text="stream compact")
            yield SessionInfo(id="sess-stream-compact")

    monkeypatch.setattr(op_mod, "OpenAIProvider", FakeProvider)

    artifacts_dir = tmp_path / "stream-compact"
    agent = Agent(
        provider="openai",
        artifacts_dir=artifacts_dir,
        include_events_in_result=False,
    )

    async def collect() -> list[EventEnvelope]:
        return [env async for env in agent.stream("ignored")]

    import asyncio

    events = asyncio.run(collect())

    saved_result = json.loads((artifacts_dir / "result.json").read_text())
    assert len(events) == 4
    assert len((artifacts_dir / "trace.jsonl").read_text().splitlines()) == 4
    assert saved_result["final_text"] == "stream compact"
    assert saved_result["session_id"] == "sess-stream-compact"
    assert saved_result["events"] == []


def test_stream_close_marks_artifacts_cancelled(monkeypatch, tmp_path):
    from agent_sdk_wrapper.providers import base
    from agent_sdk_wrapper.providers import openai_provider as op_mod

    class FakeProvider(base.ProviderAdapter):
        name = "openai"

        async def stream(self, req):  # type: ignore[override]
            yield Text(text="partial")
            yield Text(text="unconsumed")

    monkeypatch.setattr(op_mod, "OpenAIProvider", FakeProvider)

    artifacts_dir = tmp_path / "stream-cancelled"
    agent = Agent(provider="openai", artifacts_dir=artifacts_dir)

    async def consume_partially() -> None:
        stream = agent.stream("ignored")
        await anext(stream)
        delta = await anext(stream)
        assert isinstance(delta.event, Text)
        assert delta.event.text == "partial"
        await stream.aclose()

    import asyncio

    asyncio.run(consume_partially())

    manifest = json.loads((artifacts_dir / "manifest.json").read_text())
    saved_result = json.loads((artifacts_dir / "result.json").read_text())
    trace_events = [
        json.loads(line)["event"]
        for line in (artifacts_dir / "trace.jsonl").read_text().splitlines()
    ]
    assert manifest["status"] == "cancelled"
    assert saved_result["status"] == "cancelled"
    assert saved_result["ended_reason"] == "cancelled"
    assert saved_result["final_text"] == "partial"
    assert saved_result["error"] == "stream closed before completion"
    assert trace_events[-1]["type"] == "run_finished"
    assert trace_events[-1]["status"] == "cancelled"
    assert trace_events[-1]["ended_reason"] == "cancelled"


def test_run_cancellation_writes_cancelled_artifacts(monkeypatch, tmp_path):
    from agent_sdk_wrapper.providers import base
    from agent_sdk_wrapper.providers import openai_provider as op_mod

    class FakeProvider(base.ProviderAdapter):
        name = "openai"

        async def stream(self, req):  # type: ignore[override]
            import asyncio

            yield Text(text="started")
            await asyncio.Event().wait()

    monkeypatch.setattr(op_mod, "OpenAIProvider", FakeProvider)

    artifacts_dir = tmp_path / "run-cancelled"
    agent = Agent(provider="openai", artifacts_dir=artifacts_dir)

    async def cancel_run() -> None:
        import asyncio

        task = asyncio.create_task(agent.run("ignored"))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    import asyncio

    asyncio.run(cancel_run())

    manifest = json.loads((artifacts_dir / "manifest.json").read_text())
    saved_result = json.loads((artifacts_dir / "result.json").read_text())
    trace_events = [
        json.loads(line)["event"]
        for line in (artifacts_dir / "trace.jsonl").read_text().splitlines()
    ]
    assert manifest["status"] == "cancelled"
    assert saved_result["status"] == "cancelled"
    assert saved_result["ended_reason"] == "cancelled"
    assert saved_result["final_text"] == "started"
    assert saved_result["error"] == "cancelled"
    assert trace_events[-1]["type"] == "run_finished"
    assert trace_events[-1]["status"] == "cancelled"
    assert trace_events[-1]["ended_reason"] == "cancelled"


def test_run_keeps_provider_error_when_sdk_raises_afterward(monkeypatch):
    from agent_sdk_wrapper.providers import base
    from agent_sdk_wrapper.providers import openai_provider as op_mod

    class FakeProvider(base.ProviderAdapter):
        name = "openai"

        async def stream(self, req):  # type: ignore[override]
            yield Text(text="provider explanation")
            yield Error(message="actual provider error", error_type="result_error")
            raise Exception("misleading cleanup error")

    monkeypatch.setattr(op_mod, "OpenAIProvider", FakeProvider)

    agent = Agent(provider="openai")

    import asyncio
    result = asyncio.run(agent.run("ignored"))

    assert result.status == RunStatus.FAILURE
    assert result.ended_reason == RunEndedReason.ERROR
    assert result.error == "actual provider error"
    assert [
        event.event.message for event in result.events if isinstance(event.event, Error)
    ] == ["actual provider error"]


def test_run_result_distinguishes_max_turns_from_generic_error(monkeypatch):
    from agent_sdk_wrapper.providers import base
    from agent_sdk_wrapper.providers import openai_provider as op_mod

    class FakeProvider(base.ProviderAdapter):
        name = "openai"

        async def stream(self, req):  # type: ignore[override]
            yield Error(message="hit max turns", error_type="max_turns")

    monkeypatch.setattr(op_mod, "OpenAIProvider", FakeProvider)

    import asyncio

    result = asyncio.run(Agent(provider="openai").run("ignored"))

    assert result.status == RunStatus.FAILURE
    assert result.ended_reason == RunEndedReason.MAX_TURNS
    assert result.to_dict()["ended_reason"] == "max_turns"


@pytest.mark.parametrize(
    "exc",
    [
        ProviderNotAvailableError("missing runtime"),
        ConfigError("bad config"),
        TransientError("rate limit"),
    ],
)
def test_run_records_internal_errors_in_result_and_trace(monkeypatch, tmp_path, exc):
    from agent_sdk_wrapper.providers import base
    from agent_sdk_wrapper.providers import openai_provider as op_mod

    class FakeProvider(base.ProviderAdapter):
        name = "openai"

        async def stream(self, req):  # type: ignore[override]
            if False:
                yield Text(text="unreachable")
            raise exc

    monkeypatch.setattr(op_mod, "OpenAIProvider", FakeProvider)

    trace_file = tmp_path / "trace.jsonl"
    agent = Agent(provider="openai", max_retries=0, trace_file=trace_file)

    import asyncio

    result = asyncio.run(agent.run("ignored"))

    assert result.status == RunStatus.FAILURE
    assert result.ended_reason == RunEndedReason.ERROR
    assert result.error == str(exc)
    assert any(isinstance(event.event, Error) for event in result.events)
    assert len(trace_file.read_text().splitlines()) == len(result.events)


def test_run_records_retry_warning_in_result_and_trace(monkeypatch, tmp_path):
    from agent_sdk_wrapper import agent as agent_mod
    from agent_sdk_wrapper.providers import base
    from agent_sdk_wrapper.providers import openai_provider as op_mod

    class FakeProvider(base.ProviderAdapter):
        name = "openai"
        calls = 0

        async def stream(self, req):  # type: ignore[override]
            FakeProvider.calls += 1
            if FakeProvider.calls == 1:
                raise TransientError("rate limit")
            yield Text(text="ok")

    monkeypatch.setattr(agent_mod, "_backoff", lambda attempt: 0)
    monkeypatch.setattr(op_mod, "OpenAIProvider", FakeProvider)
    FakeProvider.calls = 0

    trace_file = tmp_path / "trace.jsonl"
    agent = Agent(provider="openai", max_retries=1, trace_file=trace_file)

    import asyncio

    result = asyncio.run(agent.run("ignored"))

    assert result.status == RunStatus.SUCCESS
    assert result.ended_reason == RunEndedReason.SUCCESS
    assert FakeProvider.calls == 2
    assert [event.event.type for event in result.events].count("warning") == 1
    assert len(trace_file.read_text().splitlines()) == len(result.events)


def test_run_does_not_retry_transient_after_partial_events(monkeypatch, tmp_path):
    from agent_sdk_wrapper.providers import base
    from agent_sdk_wrapper.providers import openai_provider as op_mod

    class FakeProvider(base.ProviderAdapter):
        name = "openai"
        calls = 0

        async def stream(self, req):  # type: ignore[override]
            FakeProvider.calls += 1
            yield Text(text="partial")
            raise TransientError("lost connection")

    monkeypatch.setattr(op_mod, "OpenAIProvider", FakeProvider)
    FakeProvider.calls = 0

    trace_file = tmp_path / "trace.jsonl"
    agent = Agent(provider="openai", max_retries=3, trace_file=trace_file)

    import asyncio

    result = asyncio.run(agent.run("ignored"))

    assert FakeProvider.calls == 1
    assert result.status == RunStatus.FAILURE
    assert result.ended_reason == RunEndedReason.ERROR
    assert result.final_text == "partial"
    assert result.error == "lost connection"
    assert any(
        isinstance(event.event, Error) and event.event.error_type == "TransientError"
        for event in result.events
    )
    assert len(trace_file.read_text().splitlines()) == len(result.events)


def test_run_records_timeout_in_result_and_trace(monkeypatch, tmp_path):
    from agent_sdk_wrapper.providers import base
    from agent_sdk_wrapper.providers import openai_provider as op_mod

    class FakeProvider(base.ProviderAdapter):
        name = "openai"

        async def stream(self, req):  # type: ignore[override]
            import asyncio

            await asyncio.sleep(1)
            yield Text(text="too late")

    monkeypatch.setattr(op_mod, "OpenAIProvider", FakeProvider)

    trace_file = tmp_path / "trace.jsonl"
    agent = Agent(provider="openai", timeout=0.001, trace_file=trace_file)

    import asyncio

    result = asyncio.run(agent.run("ignored"))

    assert result.status == RunStatus.TIMEOUT
    assert result.error == "timeout"
    assert any(
        isinstance(event.event, Error) and event.event.error_type == "timeout"
        for event in result.events
    )
    assert len(trace_file.read_text().splitlines()) == len(result.events)


def test_stream_marks_provider_error_as_failure_without_duplicate(monkeypatch):
    from agent_sdk_wrapper.providers import base
    from agent_sdk_wrapper.providers import openai_provider as op_mod

    class FakeProvider(base.ProviderAdapter):
        name = "openai"

        async def stream(self, req):  # type: ignore[override]
            yield Error(message="stream provider error", error_type="result_error")
            raise Exception("misleading cleanup error")

    monkeypatch.setattr(op_mod, "OpenAIProvider", FakeProvider)

    agent = Agent(provider="openai")

    import asyncio

    async def collect() -> list[EventEnvelope]:
        return [env async for env in agent.stream("ignored")]

    events = asyncio.run(collect())

    assert [event.event.message for event in events if isinstance(event.event, Error)] == [
        "stream provider error"
    ]
    finished = events[-1].event
    assert isinstance(finished, RunFinished)
    assert finished.status == RunStatus.FAILURE
