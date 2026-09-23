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
    FakeProvider,
    McpStdioServer,
    ProviderEventEnvelope,
    ProviderNotAvailableError,
    RunEndedReason,
    RunFinished,
    RunStatus,
    SessionInfo,
    StructuredOutput,
    SubagentDef,
    Text,
    TokenUsage,
    ToolCall,
    ToolResult,
    Usage,
    install_fake_providers,
)
from agent_sdk_wrapper.request import normalize_effort_for_provider, parse_model_spec


def test_token_usage_addition():
    a = TokenUsage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15)
    b = TokenUsage(requests=2, input_tokens=2, output_tokens=3, total_tokens=5)
    s = a + b
    assert (s.requests, s.input_tokens, s.output_tokens, s.total_tokens) == (3, 12, 8, 20)


def test_normalize_effort_for_provider():
    assert normalize_effort_for_provider("anthropic", "HIGH") == "high"
    assert normalize_effort_for_provider("anthropic", "max") == "max"
    assert normalize_effort_for_provider("openai", "max") == "max"
    assert normalize_effort_for_provider("openai", "minimal") == "minimal"

    with pytest.raises(ConfigError, match="not supported"):
        normalize_effort_for_provider("anthropic", "minimal")


def test_unknown_provider_raises():
    from agent_sdk_wrapper import ConfigError

    with pytest.raises(ConfigError):
        Agent(provider="bogus")  # type: ignore[arg-type]


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
    assert Agent(model="codex:gpt-5", effort="max").effort == "max"

    with pytest.raises(ConfigError, match="not supported"):
        Agent(model="anthropic:claude-haiku-4-5", effort="none")


def test_parse_model_spec_accepts_provider_prefixes():
    assert parse_model_spec("codex:gpt-5") == ("openai", "gpt-5")
    assert parse_model_spec("anthropic:claude-haiku-4-5") == (
        "anthropic",
        "claude-haiku-4-5",
    )
    assert parse_model_spec("gpt-5") == (None, "gpt-5")
    assert parse_model_spec(" Codex : gpt-5 ") == ("openai", "gpt-5")
    with pytest.raises(ConfigError, match="provider:model"):
        parse_model_spec("anthropic:")


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("anthropic", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"),
        ("anthropic", "arn:aws:bedrock:us-east-1:123456789012:inference-profile/x"),
        ("openai", "ft:gpt-4o:acme:custom:abc123"),
        ("openai", "qwen2.5-coder:7b"),
    ],
)
def test_model_ids_with_colons_are_not_provider_prefixes(provider, model):
    assert parse_model_spec(model) == (None, model)
    assert Agent(provider=provider, model=model).model == model
    assert Agent(provider=provider, model=f"{provider}:{model}").model == model


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
            yield Text(text="Checking.")
            yield Text(text="Hello world")
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
    # One envelope per JSONL line.
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
    agent = Agent(provider="openai", extra_options={"not_an_option": True})

    with pytest.raises(ConfigError, match="not_an_option"):
        agent.check_runtime()


def test_check_runtime_validates_anthropic_request_before_runtime_check():
    agent = Agent(
        provider="anthropic",
        extra_options={"not_an_option": True},
    )

    with pytest.raises(ConfigError, match="not_an_option"):
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
    assert saved_result["error"] == "run cancelled"
    assert trace_events[-1]["type"] == "run_finished"
    assert trace_events[-1]["status"] == "cancelled"
    assert trace_events[-1]["ended_reason"] == "cancelled"


def test_run_keeps_provider_error_when_sdk_raises_afterward(monkeypatch, caplog):
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
    logged = [record for record in caplog.records if "misleading cleanup" in record.message]
    assert [record.levelname for record in logged] == ["WARNING"]


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
    ("exc", "error_type"),
    [
        (ProviderNotAvailableError("missing runtime"), "runtime_unavailable"),
        (RuntimeError("API Error: 401 invalid x-api-key"), "authentication_failed"),
        (RuntimeError("sdk bug"), "provider_exception"),
    ],
)
def test_run_records_internal_errors_in_result_and_trace(
    monkeypatch, tmp_path, exc, error_type
):
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
    agent = Agent(provider="openai", trace_file=trace_file)

    import asyncio

    result = asyncio.run(agent.run("ignored"))

    assert result.status == RunStatus.FAILURE
    assert result.ended_reason == RunEndedReason.ERROR
    assert (result.error, result.error_type) == (str(exc), error_type)
    [error] = [event.event for event in result.events if isinstance(event.event, Error)]
    assert error.error_type == error_type
    assert len(trace_file.read_text().splitlines()) == len(result.events)


def test_a_transient_failure_after_text_keeps_the_text(monkeypatch, tmp_path):
    from agent_sdk_wrapper.providers import base
    from agent_sdk_wrapper.providers import openai_provider as op_mod

    class FakeProvider(base.ProviderAdapter):
        name = "openai"

        async def stream(self, req):  # type: ignore[override]
            yield Text(text="partial")
            raise ConnectionError("connection reset by peer")

    monkeypatch.setattr(op_mod, "OpenAIProvider", FakeProvider)

    trace_file = tmp_path / "trace.jsonl"
    agent = Agent(provider="openai", trace_file=trace_file)

    import asyncio

    result = asyncio.run(agent.run("ignored"))

    assert result.status == RunStatus.FAILURE
    assert result.ended_reason == RunEndedReason.ERROR
    assert result.final_text == "partial"
    assert result.error == "connection reset by peer"
    assert any(
        isinstance(event.event, Error) and event.event.error_type == "transient_api_error"
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
    assert result.error == "run timed out after 0.001s"
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


def _trace_events(path) -> list[dict]:
    return [json.loads(line)["event"] for line in path.read_text().splitlines()]


def test_stream_timeout_does_not_cancel_consumer(monkeypatch, tmp_path):
    import asyncio

    async def slow(req):
        yield Text(text="first")
        await asyncio.sleep(0.01)
        yield Text(text="second")

    install_fake_providers(monkeypatch, events=slow)
    trace_file = tmp_path / "trace.jsonl"
    # Margins leave room for a loaded machine: startup < deadline < consumer sleep.
    agent = Agent(provider="openai", timeout=0.5, trace_file=trace_file)

    async def consume() -> tuple[list[str], int]:
        seen = []
        async for env in agent.stream("hi"):
            seen.append(env.event.type)
            if env.event.type == "text":
                await asyncio.sleep(1.0)
                seen.append("consumer done")
        return seen, asyncio.current_task().cancelling()

    seen, cancelling = asyncio.run(consume())

    assert seen == ["run_started", "text", "consumer done", "error", "run_finished"]
    assert cancelling == 0
    events = _trace_events(trace_file)
    assert events[-2]["error_type"] == "timeout"
    assert (events[-1]["status"], events[-1]["ended_reason"]) == ("timeout", "timeout")


def test_a_ready_event_after_the_deadline_is_not_delivered(monkeypatch):
    import asyncio

    # Both texts are ready at once, so only the deadline check can stop the second.
    install_fake_providers(monkeypatch, events=[Text(text="first"), Text(text="ready")])

    async def consume() -> list:
        seen = []
        async for env in Agent(provider="openai", timeout=0.3).stream("hi"):
            seen.append(env.event)
            if env.event.type == "text":
                await asyncio.sleep(0.5)
        return seen

    events = asyncio.run(consume())

    assert [event.type for event in events] == ["run_started", "text", "error", "run_finished"]
    assert events[2].error_type == "timeout"


def test_provider_timeout_error_is_not_the_run_deadline(monkeypatch):
    async def raises_timeout(req):
        yield SessionInfo(id="s1")
        raise TimeoutError("socket read timed out")

    install_fake_providers(monkeypatch, events=raises_timeout)

    result = Agent(provider="openai", timeout=60).run_sync("hi")

    assert result.status == RunStatus.FAILURE
    assert result.error == "socket read timed out"


def test_stream_close_closes_provider_iterator(monkeypatch):
    import asyncio

    closed = []

    async def endless(req):
        try:
            while True:
                yield Text(text="tick")
        finally:
            closed.append(True)

    install_fake_providers(monkeypatch, events=endless)

    async def consume() -> list[bool]:
        stream = Agent(provider="openai").stream("hi")
        await anext(stream)
        await anext(stream)
        await stream.aclose()
        return list(closed)

    assert asyncio.run(consume()) == [True]


def test_stream_raises_config_error_before_iterating(tmp_path):
    trace_file = tmp_path / "trace.jsonl"
    agent = Agent(provider="openai", extra_options={"not_an_option": True}, trace_file=trace_file)

    with pytest.raises(ConfigError, match="not_an_option"):
        agent.stream("hi")
    with pytest.raises(ConfigError, match="not_an_option"):
        agent.run_sync("hi")
    assert not trace_file.exists()


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_unknown_provider_options_raise_config_error(provider):
    with pytest.raises(ConfigError, match="no_such_option"):
        Agent(provider=provider, provider_options={"no_such_option": 1})


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (lambda tmp: {"artifacts_dir": tmp / "file"}, "not a directory"),
        (lambda tmp: {"trace_file": tmp / "dir"}, "is a directory"),
        (lambda tmp: {"artifacts_dir": tmp / "dir-with-trace-dir"}, "is a directory"),
    ],
    ids=["artifacts_dir-is-a-file", "trace_file-is-a-dir", "artifacts-trace-is-a-dir"],
)
def test_output_paths_of_the_wrong_type_raise_config_error(
    monkeypatch, tmp_path, overrides, message
):
    install_fake_providers(monkeypatch)
    (tmp_path / "file").write_text("keep")
    (tmp_path / "dir").mkdir()
    (tmp_path / "dir-with-trace-dir" / "trace.jsonl").mkdir(parents=True)

    with pytest.raises(ConfigError, match=message):
        Agent(provider="openai").stream("hi", **overrides(tmp_path))
    assert (tmp_path / "file").read_text() == "keep"
    assert not (tmp_path / "dir-with-trace-dir" / "manifest.json").exists()


@pytest.mark.parametrize(
    "names", [["repo", "repo"], ["agent_sdk_wrapper_tools"]], ids=["duplicate", "reserved"]
)
def test_mcp_server_names_must_be_unique_and_not_reserved(monkeypatch, names):
    install_fake_providers(monkeypatch)
    servers = [McpStdioServer(name=name, command="true") for name in names]

    with pytest.raises(ConfigError, match="MCP server name"):
        Agent(provider="openai", mcp_servers=servers).stream("hi")


@pytest.mark.parametrize("cwd", ["missing", "file"])
def test_cwd_must_be_an_existing_directory(monkeypatch, tmp_path, cwd):
    install_fake_providers(monkeypatch)
    (tmp_path / "file").write_text("")

    with pytest.raises(ConfigError, match="cwd"):
        Agent(provider="openai").stream("hi", cwd=tmp_path / cwd)


@pytest.mark.parametrize("option", ["allowed_tools", "disallowed_tools", "setting_sources"])
def test_a_bare_string_is_not_a_list_of_strings(monkeypatch, option):
    install_fake_providers(monkeypatch)

    with pytest.raises(ConfigError, match=f"{option} must be a list of strings"):
        Agent(provider="anthropic", **{option: "Bash"})
    with pytest.raises(ConfigError, match=f"{option} must be a list of strings"):
        Agent(provider="anthropic").stream("hi", **{option: "Bash"})


def test_a_setting_the_adapter_cannot_read_is_a_config_error():
    server = McpStdioServer(name="s", command="c", tool_approval_modes="approve")  # type: ignore[arg-type]

    with pytest.raises(ConfigError, match="invalid settings"):
        Agent(provider="codex", mcp_servers=[server]).stream("hi")


def test_uncreatable_artifacts_dir_raises_config_error_before_any_event(monkeypatch, tmp_path):
    install_fake_providers(monkeypatch)
    (tmp_path / "file").write_text("")
    seen = []

    with pytest.raises(ConfigError, match="output files"):
        Agent(provider="openai", on_event=seen.append).run_sync(
            "hi", artifacts_dir=tmp_path / "file" / "artifacts"
        )
    assert seen == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"timeout": "30"},
        {"timeout": 0},
        {"max_turns": 0},
        {"max_turns": True},
        {"max_turns": 2.5},
        {"max_turns": float("nan")},
    ],
)
def test_run_rejects_invalid_run_limits(monkeypatch, overrides):
    # The fake accepts every request, so only the Agent's own checks can reject it.
    install_fake_providers(monkeypatch)

    with pytest.raises(ConfigError, match=next(iter(overrides))):
        Agent(provider="openai").stream("hi", **overrides)


def test_late_config_error_is_recorded_then_raised(monkeypatch, tmp_path):
    async def late(req):
        yield SessionInfo(id="s1")
        raise ConfigError("unsupported option discovered by the runtime")

    install_fake_providers(monkeypatch, events=late)
    trace_file = tmp_path / "trace.jsonl"

    with pytest.raises(ConfigError, match="discovered by the runtime"):
        Agent(provider="openai", trace_file=trace_file).run_sync("hi")

    events = _trace_events(trace_file)
    assert [event["type"] for event in events][-2:] == ["error", "run_finished"]
    assert events[-2]["error_type"] == "invalid_request"


def test_an_abandoned_stream_does_not_block_the_next_run(monkeypatch):
    import asyncio

    install_fake_providers(monkeypatch, events=[SessionInfo(id="s1"), Text(text="done")])
    agent = Agent(provider="openai", continue_session=True)

    async def scenario():
        stream = agent.stream("one")
        async for _ in stream:
            break
        return await agent.run("two")

    result = asyncio.run(scenario())
    assert result.ok and result.final_text == "done"


def test_concurrent_runs_without_continue_session_are_allowed(monkeypatch):
    import asyncio

    started = []
    both_started = asyncio.Event()

    async def waits(req):
        started.append(req.prompt)
        if len(started) == 2:
            both_started.set()
        await both_started.wait()
        yield Text(text=req.prompt)

    install_fake_providers(monkeypatch, events=waits)
    agent = Agent(provider="openai")

    async def scenario():
        return await asyncio.gather(agent.run("one"), agent.run("two"))

    results = asyncio.run(scenario())
    assert [result.final_text for result in results] == ["one", "two"]


def test_artifacts_round_trip_lone_surrogates(monkeypatch, tmp_path):
    from agent_sdk_wrapper.artifacts import ProviderEventLogger

    odd = "bad \ud800 and \udcff"

    async def surrogate_events(req):
        ProviderEventLogger("openai", req.artifacts_dir, run_id=req.run_id).write({"text": odd})
        yield Text(text=odd)
        yield ToolResult(id="t1", output=odd)
        yield Error(message=odd, error_type="execution_error")

    install_fake_providers(monkeypatch, events=surrogate_events)
    artifacts_dir = tmp_path / "artifacts"

    result = Agent(provider="openai", artifacts_dir=artifacts_dir).run_sync("hi")

    assert result.error == odd
    events = _trace_events(artifacts_dir / "trace.jsonl")
    assert [event["text"] for event in events if event["type"] == "text"] == [odd]
    assert [event["output"] for event in events if event["type"] == "tool_result"] == [odd]
    saved = json.loads((artifacts_dir / "result.json").read_text(encoding="utf-8"))
    assert saved["final_text"] == odd
    assert json.loads((artifacts_dir / "manifest.json").read_text())["error"] == odd
    native = json.loads((artifacts_dir / "provider-events.jsonl").read_text())
    assert native["message"] == {"text": odd}


def test_artifacts_run_start_drops_previous_result(monkeypatch, tmp_path):
    install_fake_providers(monkeypatch, events=[Text(text="ok")])
    artifacts_dir = tmp_path / "artifacts"
    Agent(provider="openai", artifacts_dir=artifacts_dir).run_sync("first")
    assert (artifacts_dir / "result.json").exists()
    at_start = {}

    def on_event(env):
        if env.event.type == "run_started":
            manifest = json.loads((artifacts_dir / "manifest.json").read_text())
            at_start.update(
                result_exists=(artifacts_dir / "result.json").exists(),
                manifest=(manifest["run_id"], manifest["status"]),
                trace_run_ids={
                    json.loads(line)["run_id"]
                    for line in (artifacts_dir / "trace.jsonl").read_text().splitlines()
                },
                run_id=env.run_id,
            )

    Agent(provider="openai", artifacts_dir=artifacts_dir, on_event=on_event).run_sync("second")

    assert at_start["result_exists"] is False
    assert at_start["manifest"] == (at_start["run_id"], "running")
    assert at_start["trace_run_ids"] == {at_start["run_id"]}


def test_a_trace_file_that_cannot_open_leaves_the_previous_artifacts(monkeypatch, tmp_path):
    install_fake_providers(monkeypatch, events=[Text(text="ok")])
    artifacts_dir = tmp_path / "artifacts"
    first = Agent(provider="openai", artifacts_dir=artifacts_dir).run_sync("first")
    (tmp_path / "file").write_text("")

    with pytest.raises(ConfigError, match="output files"):
        Agent(provider="openai", artifacts_dir=artifacts_dir).run_sync(
            "second", trace_file=tmp_path / "file" / "trace.jsonl"
        )

    manifest = json.loads((artifacts_dir / "manifest.json").read_text())
    assert (manifest["run_id"], manifest["status"]) == (first.run_id, "success")
    assert json.loads((artifacts_dir / "result.json").read_text())["run_id"] == first.run_id


def test_manifest_trace_path_outside_the_artifacts_dir_is_absolute(monkeypatch, tmp_path):
    install_fake_providers(monkeypatch, events=[Text(text="ok")])
    monkeypatch.chdir(tmp_path)

    Agent(provider="openai", artifacts_dir="out", trace_file="logs/t.jsonl").run_sync("hi")

    trace = json.loads((tmp_path / "out" / "manifest.json").read_text())["files"]["trace"]
    assert trace == (tmp_path / "logs" / "t.jsonl").resolve().as_posix()


def test_manifest_records_the_outcome_and_reported_model(monkeypatch, tmp_path):
    install_fake_providers(
        monkeypatch,
        events=[
            SessionInfo(id="s1", model="gpt-5-2026-01-01"),
            Error(message="hit the limit", error_type="max_turns"),
        ],
    )

    Agent(provider="openai", model="gpt-5", artifacts_dir=tmp_path).run_sync("hi")

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert {key: manifest[key] for key in ("model", "status", "ended_reason", "error_type")} == {
        "model": "gpt-5-2026-01-01",
        "status": "failure",
        "ended_reason": "max_turns",
        "error_type": "max_turns",
    }


def test_artifact_json_files_are_replaced_atomically(tmp_path):
    import threading

    from agent_sdk_wrapper.artifacts import manifest_file_for, write_manifest

    def write(index: int) -> None:
        write_manifest(
            tmp_path,
            run_id=f"run-{index}",
            provider="openai",
            model=None,
            status="running",
            trace_file=None,
            error="x" * 500_000,
        )

    write(0)
    path = manifest_file_for(tmp_path)
    stop = threading.Event()
    torn_reads = []

    def read() -> None:
        while not stop.is_set():
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, FileNotFoundError) as exc:
                torn_reads.append(type(exc).__name__)

    reader = threading.Thread(target=read)
    reader.start()
    try:
        for index in range(1, 60):
            write(index)
    finally:
        stop.set()
        reader.join()

    assert torn_reads == []
    assert [p.name for p in tmp_path.iterdir()] == ["manifest.json"]

