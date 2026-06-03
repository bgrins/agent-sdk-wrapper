"""Testing helpers for API-key-free wrapper tests."""

from __future__ import annotations

import json
from collections.abc import AsyncIterable, Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .events import (
    AgentEvent,
    AgentUpdated,
    Error,
    EventEnvelope,
    RunEndedReason,
    RunFinished,
    RunResult,
    RunStarted,
    RunStatus,
    SessionInfo,
    StructuredOutput,
    Text,
    Thinking,
    TokenUsage,
    ToolCall,
    ToolResult,
    Usage,
    WarningEvent,
)
from .providers.base import ProviderAdapter
from .request import Provider, RunRequest, normalize_provider

type EventSource = Iterable[AgentEvent] | AsyncIterable[AgentEvent]
type EventFactory = Callable[[RunRequest], EventSource]


@dataclass(frozen=True)
class TraceReplay:
    """A trace fixture prepared for replay through ``FakeProvider``."""

    path: Path
    provider: Provider
    model: str | None
    cwd: str | None
    events: list[AgentEvent]
    expected: dict[str, Any]


class FakeProvider(ProviderAdapter):
    """Provider adapter that yields caller-supplied events for offline tests."""

    name = "fake"

    def __init__(
        self,
        events: EventSource | EventFactory | None = None,
        *,
        seen_requests: list[RunRequest] | None = None,
        name: str = "fake",
        **_: object,
    ) -> None:
        self.name = name
        self.events = events
        self.seen_requests = seen_requests

    async def stream(self, req: RunRequest):
        if self.seen_requests is not None:
            self.seen_requests.append(req)

        source: EventSource
        if self.events is None:
            source = [Text(text="fake response")]
        elif callable(self.events):
            source = self.events(req)
        else:
            source = self.events

        if isinstance(source, AsyncIterable):
            async for event in source:
                yield event
            return

        for event in source:
            yield event


def install_fake_providers(
    monkeypatch: Any,
    *,
    fake: FakeProvider | None = None,
    events: EventSource | EventFactory | None = None,
    seen_requests: list[RunRequest] | None = None,
    providers: Iterable[Provider | str] = ("anthropic", "openai"),
) -> FakeProvider:
    """Patch provider adapter constructors to return a reusable fake provider.

    The ``monkeypatch`` argument is intentionally typed as ``Any`` so importing
    this module does not require pytest at runtime.
    """

    fake_provider = fake or FakeProvider(events, seen_requests=seen_requests)

    def factory(**_: object) -> FakeProvider:
        return fake_provider

    for provider_input in providers:
        provider = normalize_provider(provider_input)
        if provider == "anthropic":
            from .providers import anthropic_provider as module

            monkeypatch.setattr(module, "AnthropicProvider", factory)
        elif provider == "openai":
            from .providers import openai_provider as module

            monkeypatch.setattr(module, "OpenAIProvider", factory)
        else:
            raise ValueError(f"unknown provider {provider_input!r}")

    return fake_provider


def event_from_dict(payload: dict[str, Any]) -> AgentEvent:
    """Reconstruct a normalized event from its trace JSON object."""

    data = dict(payload)
    event_type = data.pop("type", None)
    if event_type == "run_started":
        return RunStarted(**data)
    if event_type == "text":
        return Text(**data)
    if event_type == "thinking":
        return Thinking(**data)
    if event_type == "tool_call":
        return ToolCall(**data)
    if event_type == "tool_result":
        return ToolResult(**data)
    if event_type == "agent_updated":
        return AgentUpdated(**data)
    if event_type == "usage":
        usage_payload = data.pop("usage", None) or {}
        return Usage(usage=TokenUsage(**usage_payload), **data)
    if event_type == "session_info":
        return SessionInfo(**data)
    if event_type == "structured_output":
        return StructuredOutput(**data)
    if event_type == "warning":
        return WarningEvent(**data)
    if event_type == "error":
        return Error(**data)
    if event_type == "run_finished":
        if "status" in data:
            data["status"] = RunStatus(data["status"])
        if "ended_reason" in data:
            data["ended_reason"] = RunEndedReason(data["ended_reason"])
        return RunFinished(**data)
    raise ValueError(f"unknown trace event type {event_type!r}")


def envelope_from_dict(payload: dict[str, Any]) -> EventEnvelope:
    """Reconstruct an ``EventEnvelope`` from trace JSON."""

    return EventEnvelope(
        run_id=payload["run_id"],
        sequence=payload["sequence"],
        timestamp=payload["timestamp"],
        event=event_from_dict(payload["event"]),
    )


def load_trace(path: str | Path) -> list[EventEnvelope]:
    """Load a normalized ``trace.jsonl`` file into event envelopes."""

    trace_path = Path(path)
    envelopes: list[EventEnvelope] = []
    for line_number, line in enumerate(trace_path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                envelopes.append(envelope_from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"could not parse {trace_path}:{line_number}: {exc}") from exc
    return envelopes


def load_trace_replay(path: str | Path) -> TraceReplay:
    """Load a trace fixture and prepare provider events plus expected result summary.

    ``TraceReplay.events`` excludes wrapper boundary events (`run_started` and
    `run_finished`) because ``Agent.run()`` will emit fresh boundaries during
    replay. The expected summary normalizes volatile run ids, timestamps, and
    durations while preserving the event payload sequence and aggregation
    fields.
    """

    trace_path = Path(path)
    envelopes = load_trace(trace_path)
    started = next((env.event for env in envelopes if isinstance(env.event, RunStarted)), None)
    if started is None:
        raise ValueError(f"{trace_path} has no run_started event")
    provider = normalize_provider(started.provider)
    if provider is None:
        raise ValueError(f"{trace_path} has invalid provider {started.provider!r}")
    events = [
        env.event
        for env in envelopes
        if not isinstance(env.event, RunStarted | RunFinished)
    ]
    return TraceReplay(
        path=trace_path,
        provider=provider,
        model=started.model,
        cwd=started.cwd,
        events=events,
        expected=trace_summary(envelopes),
    )


def trace_summary(envelopes: Iterable[EventEnvelope]) -> dict[str, Any]:
    """Return the stable result summary represented by trace envelopes."""

    text_parts: list[str] = []
    usage: TokenUsage | None = None
    cost_usd: float | None = None
    structured_output: Any = None
    session_id: str | None = None
    provider = ""
    model: str | None = None
    status = RunStatus.SUCCESS
    ended_reason = RunEndedReason.SUCCESS
    error: str | None = None
    event_payloads: list[dict[str, Any]] = []

    for env in envelopes:
        event = env.event
        event_payloads.append(_stable_event_payload(event))
        if isinstance(event, RunStarted):
            provider = event.provider
            model = event.model
        elif isinstance(event, Text):
            text_parts.append(event.text)
        elif isinstance(event, Usage):
            usage = event.usage if usage is None else usage + event.usage
            if event.cost_usd is not None:
                cost_usd = event.cost_usd if cost_usd is None else cost_usd + event.cost_usd
        elif isinstance(event, StructuredOutput):
            structured_output = event.value
        elif isinstance(event, SessionInfo):
            session_id = event.id
        elif isinstance(event, Error) and error is None:
            error = event.message
        elif isinstance(event, RunFinished):
            status = event.status
            ended_reason = event.ended_reason

    return {
        "provider": provider,
        "model": model,
        "status": status.value,
        "ended_reason": ended_reason.value,
        "final_text": "".join(text_parts),
        "structured_output": structured_output,
        "usage": _usage_dict(usage),
        "cost_usd": cost_usd,
        "session_id": session_id,
        "error": error if status != RunStatus.SUCCESS else None,
        "events": event_payloads,
    }


def run_result_summary(result: RunResult) -> dict[str, Any]:
    """Return the same stable summary for a replayed ``RunResult``."""

    return {
        "provider": result.provider,
        "model": result.model,
        "status": result.status.value,
        "ended_reason": result.ended_reason.value,
        "final_text": result.final_text,
        "structured_output": result.structured_output,
        "usage": _usage_dict(result.usage),
        "cost_usd": result.cost_usd,
        "session_id": result.session_id,
        "error": result.error,
        "events": [_stable_event_payload(env.event) for env in result.events],
    }


def _stable_event_payload(event: AgentEvent) -> dict[str, Any]:
    payload = event.to_dict()
    if payload.get("type") == "run_finished":
        payload["duration_ms"] = 0
    return payload


def _usage_dict(usage: TokenUsage | None) -> dict[str, Any] | None:
    if usage is None:
        return None
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
        "reasoning_output_tokens": usage.reasoning_output_tokens,
        "requests": usage.requests,
    }


__all__ = [
    "EventFactory",
    "EventSource",
    "FakeProvider",
    "TraceReplay",
    "envelope_from_dict",
    "event_from_dict",
    "install_fake_providers",
    "load_trace",
    "load_trace_replay",
    "run_result_summary",
    "trace_summary",
]
