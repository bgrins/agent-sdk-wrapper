"""Agent configuration, provider dispatch and event collection."""

from __future__ import annotations

import asyncio
import contextlib
import math
import random
import time
import uuid
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import (
    ProviderEventCallback,
    collect_side_files,
    normalize_artifacts_dir,
    provider_events_file_for,
    trace_file_for,
    write_manifest,
    write_result_artifact,
)
from .errors import (
    ConfigError,
    ProcessTerminatedError,
    ProviderNotAvailableError,
    RunFailedError,
    TransientError,
)
from .events import (
    AgentEvent,
    AgentUpdated,
    ContextCompacted,
    Error,
    EventEnvelope,
    RunEndedReason,
    RunFinished,
    RunResult,
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
    utcnow_iso,
)
from .logging import TraceWriter, get_logger
from .mcp import McpServer
from .providers import build_provider
from .request import (
    Provider,
    ProviderInput,
    RunRequest,
    SubagentDef,
    normalize_builtin_tools,
    normalize_effort_for_provider,
    normalize_model_for_provider,
    normalize_subagents_for_provider,
    resolve_provider,
)

DEFAULT_CONTEXT_DUMP_PROMPT = (
    "Summarize the current conversation as durable context for a future run. "
    "Include goals, decisions, important files, open questions, and next steps."
)

_ALLOWED_OVERRIDES = frozenset({
    "allowed_tools",
    "artifacts_dir",
    "builtin_tools",
    "continue_session",
    "cwd",
    "disallowed_tools",
    "effort",
    "env",
    "extra_options",
    "include_events_in_result",
    "include_raw",
    "max_retries",
    "max_turns",
    "mcp_servers",
    "model",
    "on_event",
    "on_provider_event",
    "output_schema",
    "permission_mode",
    "raise_on_error",
    "session_id",
    "setting_sources",
    "subagents",
    "system_prompt",
    "timeout",
    "tools",
    "trace_file",
    "web_tools",
})

# Events that show an attempt did work; after one, the attempt is never retried.
_PROGRESS_EVENTS = (
    Text,
    Thinking,
    ToolCall,
    ToolResult,
    StructuredOutput,
    SubagentStarted,
    SubagentEnded,
    ContextCompacted,
    AgentUpdated,
)

_ENDED_REASONS = {
    "max_turns": RunEndedReason.MAX_TURNS,
    "timeout": RunEndedReason.TIMEOUT,
    "cancelled": RunEndedReason.CANCELLED,
    "refused": RunEndedReason.REFUSED,
}


class _DeadlineExceeded(Exception):
    """The wrapper deadline passed while waiting on the provider."""


_END = object()


@dataclass
class _Run:
    req: RunRequest
    session_overridden: bool
    trace_path: str | Path | None
    on_event: Callable[[EventEnvelope], None] | None
    raise_on_error: bool
    result: RunResult | None = None


class Agent:
    """Run or stream calls through Claude or Codex.

    Constructor arguments are defaults; per-call arguments override them.
    """

    def __init__(
        self,
        *,
        provider: ProviderInput = None,
        model: str | None = None,
        system_prompt: str | None = None,
        tools: Sequence[Callable[..., Any]] | None = None,
        subagents: Mapping[str, SubagentDef] | None = None,
        mcp_servers: Sequence[McpServer] | None = None,
        output_schema: type | None = None,
        max_turns: int | None = None,
        effort: str | None = None,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int = 2,
        include_raw: bool = False,
        include_events_in_result: bool = True,
        builtin_tools: Sequence[str] | str | None = None,
        web_tools: bool | None = None,
        allowed_tools: Sequence[str] | None = None,
        disallowed_tools: Sequence[str] | None = None,
        session_id: str | None = None,
        continue_session: bool = False,
        permission_mode: str | None = None,
        setting_sources: list[str] | None = None,
        extra_options: dict[str, Any] | None = None,
        provider_options: dict[str, Any] | None = None,
        trace_file: str | Path | None = None,
        artifacts_dir: str | Path | None = None,
        on_event: Callable[[EventEnvelope], None] | None = None,
        on_provider_event: ProviderEventCallback | None = None,
        raise_on_error: bool = False,
    ) -> None:
        self.provider: Provider = resolve_provider(provider, model)
        self.model = normalize_model_for_provider(self.provider, model)
        self.system_prompt = system_prompt
        self.tools = list(tools or [])
        self.subagents = normalize_subagents_for_provider(
            self.provider, dict(subagents or {})
        )
        self.mcp_servers = list(mcp_servers or [])
        self.output_schema = output_schema
        self.max_turns = max_turns
        self.effort = normalize_effort_for_provider(self.provider, effort)
        self.cwd = cwd
        self.env = dict(env or {})
        self.timeout = timeout
        self.max_retries = max_retries
        self.include_raw = include_raw
        self.include_events_in_result = include_events_in_result
        self.builtin_tools = normalize_builtin_tools(builtin_tools)
        self.web_tools = web_tools
        self.allowed_tools = list(allowed_tools or [])
        self.disallowed_tools = list(disallowed_tools or [])
        self.session_id = session_id
        self.continue_session = continue_session
        self.permission_mode = permission_mode
        self.setting_sources = setting_sources
        self.extra_options = dict(extra_options or {})
        self.trace_file = trace_file
        self.artifacts_dir = artifacts_dir
        self.on_event = on_event
        self.on_provider_event = on_provider_event
        self.raise_on_error = raise_on_error

        self._provider = build_provider(self.provider, **(provider_options or {}))

    def check_runtime(self) -> None:
        """Validate the request before checking runtime availability."""

        req = self._build_request("", {})
        self._provider.validate_request(req)
        self._provider.ensure_available()

    def stream(self, prompt: str, **overrides: Any) -> AsyncIterator[EventEnvelope]:
        """Stream envelopes for one run. Invalid settings raise ``ConfigError`` here."""

        return self._events(self._prepare(prompt, overrides))

    async def run(self, prompt: str, **overrides: Any) -> RunResult:
        """Collect :meth:`stream` into a ``RunResult``."""

        run = self._prepare(prompt, overrides)
        async with contextlib.aclosing(self._events(run)) as events:
            async for _ in events:
                pass
        result = run.result
        assert result is not None
        if not result.ok and run.raise_on_error:
            raise RunFailedError(result.error or "run failed", status=result.status.value)
        return result

    def run_sync(self, prompt: str, **overrides: Any) -> RunResult:
        return asyncio.run(self.run(prompt, **overrides))

    async def dump_context(
        self,
        path: str | Path,
        *,
        prompt: str = DEFAULT_CONTEXT_DUMP_PROMPT,
        **overrides: Any,
    ) -> RunResult:
        """Ask the current provider session for a summary and write it to ``path``."""

        result = await self.run(prompt, **overrides)
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(result.final_text, encoding="utf-8")
        return result

    def dump_context_sync(
        self,
        path: str | Path,
        *,
        prompt: str = DEFAULT_CONTEXT_DUMP_PROMPT,
        **overrides: Any,
    ) -> RunResult:
        return asyncio.run(self.dump_context(path, prompt=prompt, **overrides))

    def _prepare(self, prompt: str, overrides: dict[str, Any]) -> _Run:
        _check_overrides(overrides)
        req = self._build_request(prompt, overrides)
        self._provider.validate_request(req)
        return _Run(
            req=req,
            session_overridden="session_id" in overrides,
            trace_path=overrides.get("trace_file", self.trace_file),
            on_event=overrides.get("on_event", self.on_event),
            raise_on_error=bool(overrides.get("raise_on_error", self.raise_on_error)),
        )

    def _build_request(self, prompt: str, overrides: dict[str, Any]) -> RunRequest:
        def pick(name: str, default: Any) -> Any:
            return overrides[name] if name in overrides else default

        timeout = pick("timeout", self.timeout)
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, int | float)
            or not (timeout > 0 and math.isfinite(timeout))
        ):
            raise ConfigError(f"timeout must be a positive number of seconds, got {timeout!r}")
        max_retries = pick("max_retries", self.max_retries)
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ConfigError(f"max_retries must be a non-negative integer, got {max_retries!r}")

        return RunRequest(
            provider=self.provider,
            prompt=prompt,
            model=normalize_model_for_provider(self.provider, pick("model", self.model)),
            system_prompt=pick("system_prompt", self.system_prompt),
            tools=list(pick("tools", self.tools)),
            subagents=normalize_subagents_for_provider(
                self.provider, dict(pick("subagents", self.subagents))
            ),
            mcp_servers=list(pick("mcp_servers", self.mcp_servers)),
            output_schema=pick("output_schema", self.output_schema),
            max_turns=pick("max_turns", self.max_turns),
            effort=normalize_effort_for_provider(
                self.provider, pick("effort", self.effort)
            ),
            cwd=pick("cwd", self.cwd),
            env=dict(pick("env", self.env)),
            timeout=timeout,
            max_retries=max_retries,
            include_raw=bool(pick("include_raw", self.include_raw)),
            include_events_in_result=bool(
                pick("include_events_in_result", self.include_events_in_result)
            ),
            artifacts_dir=pick("artifacts_dir", self.artifacts_dir),
            on_provider_event=pick("on_provider_event", self.on_provider_event),
            builtin_tools=normalize_builtin_tools(
                pick("builtin_tools", self.builtin_tools)
            ),
            web_tools=pick("web_tools", self.web_tools),
            allowed_tools=list(pick("allowed_tools", self.allowed_tools)),
            disallowed_tools=list(pick("disallowed_tools", self.disallowed_tools)),
            session_id=pick("session_id", self.session_id),
            continue_session=bool(pick("continue_session", self.continue_session)),
            permission_mode=pick("permission_mode", self.permission_mode),
            setting_sources=pick("setting_sources", self.setting_sources),
            extra_options=dict(pick("extra_options", self.extra_options)),
        )

    async def _events(self, run: _Run) -> AsyncIterator[EventEnvelope]:
        req = run.req
        run_id = uuid.uuid4().hex
        req.run_id = run_id
        if not run.session_overridden:
            req.session_id = self.session_id
        original_session_id = req.session_id
        on_event = run.on_event
        include_events = req.include_events_in_result
        seq = _SeqGen()
        state = _ResultState()
        result_events: list[EventEnvelope] = []
        # Retryable errors wait here until the attempt either retries or makes progress.
        held: list[Error] = []
        start = time.monotonic()
        deadline = None if req.timeout is None else start + req.timeout
        status = RunStatus.SUCCESS
        ended_reason = RunEndedReason.SUCCESS
        error_msg: str | None = None
        duration_ms = 0
        finished = False
        writer: TraceWriter | None = None
        artifacts_dir: Path | None = None
        trace_path = run.trace_path

        def record(event: AgentEvent) -> EventEnvelope:
            nonlocal status, ended_reason, error_msg
            assert writer is not None
            if req.continue_session and isinstance(event, SessionInfo) and event.id:
                self.session_id = event.id
            if isinstance(event, Error) and error_msg is None:
                error_msg = event.message or "provider reported an error"
                ended_reason = _ended_reason_from_error_type(event.error_type)
                status = _status_for(ended_reason)
            env = EventEnvelope(run_id, seq.next(), utcnow_iso(), event)
            writer.write(env)
            state.record(env)
            if include_events:
                result_events.append(env)
            if on_event is not None:
                try:
                    on_event(env)
                except Exception:
                    get_logger().exception("on_event callback raised; continuing run")
            return env

        def flush_held() -> list[EventEnvelope]:
            envs = [record(event) for event in held]
            held.clear()
            return envs

        def finish() -> EventEnvelope:
            nonlocal duration_ms, finished
            duration_ms = _elapsed_ms(start)
            env = record(
                RunFinished(status=status, duration_ms=duration_ms, ended_reason=ended_reason)
            )
            finished = True
            return env

        try:
            artifacts_dir = normalize_artifacts_dir(req.artifacts_dir)
            trace_path = _resolve_trace_path(trace_path, artifacts_dir)
            if artifacts_dir is not None:
                provider_events_file_for(artifacts_dir).unlink(missing_ok=True)
            writer = TraceWriter(trace_path)
            yield record(
                RunStarted(
                    provider=req.provider,
                    model=req.model,
                    cwd=_as_str(req.cwd),
                    prompt=req.prompt or None,
                    system_prompt=req.system_prompt,
                )
            )
            if artifacts_dir is not None:
                write_manifest(
                    artifacts_dir,
                    run_id=run_id,
                    provider=req.provider,
                    model=req.model,
                    status="running",
                    trace_file=trace_path,
                )

            try:
                attempt = 0
                while True:
                    req.attempt = attempt
                    req.session_id = original_session_id
                    progressed = failed = False
                    error: Exception | None = None
                    async with contextlib.aclosing(self._provider.stream(req)) as native:
                        while True:
                            try:
                                ev = await _next_event(native, deadline)
                            except (ProcessTerminatedError, ConfigError, _DeadlineExceeded):
                                raise
                            except Exception as exc:
                                error = exc
                                break
                            if ev is _END:
                                break
                            if isinstance(ev, Error) and ev.retryable and not (
                                progressed or failed
                            ):
                                held.append(ev)
                                continue
                            if held and isinstance(ev, (*_PROGRESS_EVENTS, Error)):
                                failed = True
                                for env in flush_held():
                                    yield env
                            progressed = progressed or isinstance(ev, _PROGRESS_EVENTS)
                            failed = failed or isinstance(ev, Error)
                            yield record(ev)

                    retryable = bool(held) or isinstance(error, TransientError)
                    if retryable and not (progressed or failed) and attempt < req.max_retries:
                        reason = held[0].message if held else str(error)
                        held.clear()
                        delay = _backoff(attempt)
                        yield record(
                            WarningEvent(
                                message=(
                                    f"transient error, retrying in {delay:.1f}s "
                                    f"(attempt {attempt + 1}/{req.max_retries}): {reason}"
                                )
                            )
                        )
                        await _retry_sleep(delay, deadline)
                        attempt += 1
                        continue

                    if held:
                        failed = True
                        for env in flush_held():
                            yield env
                    if error is not None and not failed:
                        yield record(_error_event(error))
                    break
            except _DeadlineExceeded:
                for env in flush_held():
                    yield env
                yield record(
                    Error(message=f"run timed out after {req.timeout}s", error_type="timeout")
                )
                status = RunStatus.TIMEOUT
                ended_reason = RunEndedReason.TIMEOUT
                error_msg = "timeout"
            except (ProcessTerminatedError, ConfigError) as exc:
                for env in flush_held():
                    yield env
                yield record(_error_event(exc))
                yield finish()
                raise

            yield finish()
        except GeneratorExit:
            if writer is not None and not finished:
                flush_held()
                if error_msg is None:
                    record(Error(message="stream closed before completion", error_type="cancelled"))
                finish()
            raise
        except asyncio.CancelledError:
            if writer is not None and not finished:
                flush_held()
                record(Error(message="run cancelled", error_type="cancelled"))
                status = RunStatus.CANCELLED
                ended_reason = RunEndedReason.CANCELLED
                error_msg = "cancelled"
                finish()
            raise
        finally:
            if writer is not None:
                writer.close()
                if not finished:
                    duration_ms = _elapsed_ms(start)
                run.result = state.to_result(
                    run_id=run_id,
                    provider=req.provider,
                    model=req.model,
                    status=status,
                    ended_reason=ended_reason,
                    events=result_events,
                    duration_ms=duration_ms,
                    artifacts_dir=_as_str(artifacts_dir),
                    error_msg=error_msg,
                )
                if artifacts_dir is not None:
                    result_path = write_result_artifact(artifacts_dir, run.result)
                    write_manifest(
                        artifacts_dir,
                        run_id=run_id,
                        provider=req.provider,
                        model=req.model,
                        status=status.value,
                        trace_file=trace_path,
                        result_file=result_path,
                        duration_ms=duration_ms,
                        error=error_msg,
                        extra_files=collect_side_files(artifacts_dir),
                    )


class _SeqGen:
    __slots__ = ("_n",)

    def __init__(self) -> None:
        self._n = 0

    def next(self) -> int:
        n = self._n
        self._n += 1
        return n


def _backoff(attempt: int, *, base: float = 0.5, cap: float = 8.0) -> float:
    """Decorrelated exponential backoff with jitter."""
    return min(cap, base * (2**attempt)) * (0.5 + random.random() / 2)


def _elapsed_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


async def _next_event(native: AsyncIterator[AgentEvent], deadline: float | None) -> Any:
    """Await the next provider event, applying the run deadline to this wait only."""

    try:
        if deadline is None:
            return await anext(native)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _DeadlineExceeded
        scope = asyncio.timeout(remaining)
        try:
            async with scope:
                return await anext(native)
        except TimeoutError:
            if scope.expired():
                raise _DeadlineExceeded from None
            raise
    except StopAsyncIteration:
        return _END


async def _retry_sleep(delay: float, deadline: float | None) -> None:
    if deadline is not None and time.monotonic() + delay >= deadline:
        await asyncio.sleep(max(0.0, deadline - time.monotonic()))
        raise _DeadlineExceeded
    await asyncio.sleep(delay)


def _error_event(exc: BaseException) -> Error:
    message = str(exc) or type(exc).__name__
    if isinstance(exc, TransientError):
        return Error(message=message, error_type="transient_api_error", retryable=True)
    if isinstance(exc, ProcessTerminatedError):
        return Error(message=message, error_type="process_terminated")
    if isinstance(exc, ProviderNotAvailableError):
        return Error(message=message, error_type="runtime_unavailable")
    if isinstance(exc, ConfigError):
        return Error(message=message, error_type="invalid_request")
    return Error(message=message, error_type="provider_exception")


def _check_overrides(overrides: dict[str, Any]) -> None:
    unknown = sorted(set(overrides) - _ALLOWED_OVERRIDES)
    if unknown:
        raise ConfigError(
            f"unknown Agent.run/stream override(s): {', '.join(unknown)}"
        )


def _as_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _resolve_trace_path(trace_path: Any, artifacts_dir: Path | None) -> str | Path | None:
    if trace_path is not None:
        return trace_path
    if artifacts_dir is not None:
        return trace_file_for(artifacts_dir)
    return None


def _ended_reason_from_error_type(error_type: str | None) -> RunEndedReason:
    return _ENDED_REASONS.get(error_type or "", RunEndedReason.ERROR)


def _status_for(reason: RunEndedReason) -> RunStatus:
    if reason == RunEndedReason.CANCELLED:
        return RunStatus.CANCELLED
    if reason == RunEndedReason.TIMEOUT:
        return RunStatus.TIMEOUT
    return RunStatus.FAILURE


class _ResultState:
    def __init__(self) -> None:
        self.final_text = ""
        self.structured: Any = None
        self.usage: TokenUsage | None = None
        self.cost: float | None = None
        self.session_id: str | None = None
        self.model: str | None = None

    def record(self, env: EventEnvelope) -> None:
        ev = env.event
        if isinstance(ev, Text):
            self.final_text = ev.text
        elif isinstance(ev, Usage):
            self.usage = ev.usage if self.usage is None else self.usage + ev.usage
            if ev.cost_usd is not None:
                self.cost = ev.cost_usd if self.cost is None else self.cost + ev.cost_usd
        elif isinstance(ev, StructuredOutput):
            self.structured = ev.value
        elif isinstance(ev, SessionInfo):
            self.session_id = ev.id
            if ev.model:
                self.model = ev.model

    def to_result(
        self,
        *,
        run_id: str,
        provider: str,
        model: str | None,
        status: RunStatus,
        ended_reason: RunEndedReason,
        events: list[EventEnvelope],
        duration_ms: int,
        artifacts_dir: str | None,
        error_msg: str | None,
    ) -> RunResult:
        return RunResult(
            run_id=run_id,
            provider=provider,
            model=self.model or model,
            status=status,
            ended_reason=ended_reason,
            final_text=self.final_text,
            structured_output=self.structured,
            usage=self.usage,
            cost_usd=self.cost,
            duration_ms=duration_ms,
            session_id=self.session_id,
            artifacts_dir=artifacts_dir,
            error=error_msg,
            events=events,
        )


__all__ = [
    "Agent",
    "AgentUpdated",
    "Error",
    "EventEnvelope",
    "Text",
    "RunEndedReason",
    "RunFinished",
    "RunResult",
    "RunStarted",
    "RunStatus",
    "SessionInfo",
    "StructuredOutput",
    "Thinking",
    "ToolCall",
    "ToolResult",
    "Usage",
]
