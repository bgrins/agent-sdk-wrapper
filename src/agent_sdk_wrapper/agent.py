"""The unified ``Agent`` — one surface over both backend SDKs.

``Agent`` holds the persistent configuration (provider, model, system prompt,
tools, subagents, output schema). Each call to :meth:`Agent.run` or
:meth:`Agent.stream` builds a :class:`RunRequest`, drives the provider's event
stream, frames every event in an :class:`EventEnvelope`, logs it, and either
collects a :class:`RunResult` or yields envelopes to the caller.
"""

from __future__ import annotations

import asyncio
import random
import uuid
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .artifacts import (
    ProviderEventCallback,
    collect_side_files,
    normalize_artifacts_dir,
    trace_file_for,
    write_manifest,
    write_result_artifact,
)
from .errors import ConfigError, ProviderNotAvailableError, RunFailedError, TransientError
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


class Agent:
    """A unified agent over the Claude Agent SDK or the OpenAI Codex SDK.

    The same ``run()`` / ``stream()`` surface works for either backend. Keyword
    arguments to the constructor are defaults; the same names on ``run`` /
    ``stream`` override them per call.
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

    # ── public API ──────────────────────────────────────────────────────

    def check_runtime(self) -> None:
        """Validate the resolved request, then raise if runtime is unavailable.

        This runs provider-specific request validation first, so most
        ``ConfigError`` cases surface before provider I/O or the first run.
        """

        req = self._build_request("", {})
        self._provider.validate_request(req)
        self._provider.ensure_available()

    def stream(self, prompt: str, **overrides: Any) -> AsyncIterator[EventEnvelope]:
        _check_overrides(overrides)
        req = self._build_request(prompt, overrides)
        trace_path = overrides.get("trace_file", self.trace_file)
        on_event = overrides.get("on_event", self.on_event)
        return self._stream(req, trace_path, on_event)

    async def run(self, prompt: str, **overrides: Any) -> RunResult:
        _check_overrides(overrides)
        req = self._build_request(prompt, overrides)
        artifacts_dir = normalize_artifacts_dir(req.artifacts_dir)
        trace_path = _resolve_trace_path(
            overrides.get("trace_file", self.trace_file), artifacts_dir
        )
        on_event = overrides.get("on_event", self.on_event)
        raise_on_error = bool(overrides.get("raise_on_error", self.raise_on_error))

        run_id = uuid.uuid4().hex
        writer = TraceWriter(trace_path)
        seq = _SeqGen()
        loop_start = asyncio.get_event_loop().time()
        result_events: list[EventEnvelope] = []
        result_state = _ResultState()
        result: RunResult | None = None
        include_events = req.include_events_in_result
        finished = False
        duration_ms = 0
        status = RunStatus.SUCCESS
        ended_reason = RunEndedReason.SUCCESS
        error_msg: str | None = None
        attempt_events: list[EventEnvelope] = []

        def emit(event: AgentEvent) -> EventEnvelope:
            self._record_session(req, event)
            env = EventEnvelope(run_id, seq.next(), utcnow_iso(), event)
            writer.write(env)
            if on_event is not None:
                try:
                    on_event(env)
                except Exception:
                    get_logger().exception("on_event callback raised; continuing run")
            return env

        def record(event: AgentEvent) -> EventEnvelope:
            env = emit(event)
            result_state.record(env)
            if include_events:
                result_events.append(env)
            return env

        try:
            record(RunStarted(provider=req.provider, model=req.model, cwd=_as_str(req.cwd)))
            if artifacts_dir is not None:
                write_manifest(
                    artifacts_dir,
                    run_id=run_id,
                    provider=req.provider,
                    model=req.model,
                    status="running",
                    trace_file=trace_path,
                )

            attempt = 0
            while True:
                attempt_events = []
                attempt_had_events = False
                attempt_error_msg: str | None = None
                attempt_error_type: str | None = None
                try:
                    async for ev in self._iterate_with_timeout(req):
                        env = emit(ev)
                        result_state.record(env)
                        attempt_had_events = True
                        if include_events:
                            attempt_events.append(env)
                        if isinstance(ev, Error) and attempt_error_msg is None:
                            attempt_error_msg = ev.message or "provider reported an error"
                            attempt_error_type = ev.error_type
                    if attempt_error_msg is None:
                        status = RunStatus.SUCCESS
                        ended_reason = RunEndedReason.SUCCESS
                    else:
                        status = RunStatus.FAILURE
                        ended_reason = _ended_reason_from_error_type(attempt_error_type)
                        error_msg = attempt_error_msg
                    if include_events:
                        result_events.extend(attempt_events)
                    break
                except TransientError as exc:
                    if attempt_had_events:
                        if include_events:
                            result_events.extend(attempt_events)
                        if attempt_error_msg is None:
                            record(Error(message=str(exc), error_type=type(exc).__name__))
                        status = RunStatus.FAILURE
                        ended_reason = _ended_reason_from_error_type(attempt_error_type)
                        error_msg = attempt_error_msg or str(exc)
                        break
                    if attempt < req.max_retries:
                        delay = _backoff(attempt)
                        record(
                            WarningEvent(
                                message=(
                                    f"transient error, retrying in {delay:.1f}s "
                                    f"(attempt {attempt + 1}/{req.max_retries}): {exc}"
                                )
                            )
                        )
                        await asyncio.sleep(delay)
                        attempt += 1
                        continue
                    record(Error(message=str(exc), error_type=type(exc).__name__))
                    status = RunStatus.FAILURE
                    ended_reason = RunEndedReason.ERROR
                    error_msg = str(exc)
                    break
                except TimeoutError:
                    if include_events:
                        result_events.extend(attempt_events)
                    record(
                        Error(message=f"run timed out after {req.timeout}s", error_type="timeout")
                    )
                    status = RunStatus.TIMEOUT
                    ended_reason = RunEndedReason.TIMEOUT
                    error_msg = "timeout"
                    break
                except (ProviderNotAvailableError, ConfigError) as exc:
                    if include_events:
                        result_events.extend(attempt_events)
                    record(Error(message=str(exc), error_type=type(exc).__name__))
                    status = RunStatus.FAILURE
                    ended_reason = RunEndedReason.ERROR
                    error_msg = str(exc)
                    break
                except Exception as exc:
                    if attempt_error_msg is not None:
                        status = RunStatus.FAILURE
                        ended_reason = _ended_reason_from_error_type(attempt_error_type)
                        error_msg = attempt_error_msg
                        if include_events:
                            result_events.extend(attempt_events)
                        break
                    if include_events:
                        result_events.extend(attempt_events)
                    record(Error(message=str(exc), error_type=type(exc).__name__))
                    status = RunStatus.FAILURE
                    ended_reason = RunEndedReason.ERROR
                    error_msg = str(exc)
                    break
        except asyncio.CancelledError:
            if include_events:
                result_events.extend(attempt_events)
            status = RunStatus.CANCELLED
            ended_reason = RunEndedReason.CANCELLED
            error_msg = "cancelled"
            try:
                record(Error(message="run cancelled", error_type="cancelled"))
            except Exception:
                pass
            raise
        finally:
            duration_ms = int((asyncio.get_event_loop().time() - loop_start) * 1000)
            try:
                if not finished:
                    record(
                        RunFinished(
                            status=status,
                            duration_ms=duration_ms,
                            ended_reason=ended_reason,
                        )
                    )
                    finished = True
            finally:
                writer.close()

            result = result_state.to_result(
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
                result_path = write_result_artifact(artifacts_dir, result)
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
        assert result is not None
        if not result.ok and raise_on_error:
            raise RunFailedError(error_msg or "run failed", status=status.value)
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

    # ── internals ───────────────────────────────────────────────────────

    def _build_request(self, prompt: str, overrides: dict[str, Any]) -> RunRequest:
        def pick(name: str, default: Any) -> Any:
            return overrides[name] if name in overrides else default

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
            timeout=pick("timeout", self.timeout),
            max_retries=int(pick("max_retries", self.max_retries)),
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

    async def _iterate_with_timeout(self, req: RunRequest) -> AsyncIterator[AgentEvent]:
        if req.timeout is None:
            async for ev in self._provider.stream(req):
                yield ev
            return
        async with asyncio.timeout(req.timeout):
            async for ev in self._provider.stream(req):
                yield ev

    async def _stream(
        self,
        req: RunRequest,
        trace_path: str | Path | None,
        on_event: Callable[[EventEnvelope], None] | None,
    ) -> AsyncIterator[EventEnvelope]:
        run_id = uuid.uuid4().hex
        artifacts_dir = normalize_artifacts_dir(req.artifacts_dir)
        trace_path = _resolve_trace_path(trace_path, artifacts_dir)
        writer = TraceWriter(trace_path)
        seq = _SeqGen()
        loop_start = asyncio.get_event_loop().time()
        result_events: list[EventEnvelope] = []
        result_state = _ResultState()
        include_events = req.include_events_in_result
        finished = False
        duration_ms = 0

        def emit(event: AgentEvent) -> EventEnvelope:
            self._record_session(req, event)
            env = EventEnvelope(run_id, seq.next(), utcnow_iso(), event)
            writer.write(env)
            if on_event is not None:
                try:
                    on_event(env)
                except Exception:
                    get_logger().exception("on_event callback raised; continuing run")
            return env

        def record(event: AgentEvent) -> EventEnvelope:
            env = emit(event)
            result_state.record(env)
            if include_events:
                result_events.append(env)
            return env

        status = RunStatus.SUCCESS
        ended_reason = RunEndedReason.SUCCESS
        error_msg: str | None = None

        def finish() -> EventEnvelope:
            nonlocal duration_ms, finished
            duration_ms = int((asyncio.get_event_loop().time() - loop_start) * 1000)
            env = record(
                RunFinished(
                    status=status,
                    duration_ms=duration_ms,
                    ended_reason=ended_reason,
                )
            )
            finished = True
            return env

        try:
            yield record(RunStarted(provider=req.provider, model=req.model, cwd=_as_str(req.cwd)))
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
                async for ev in self._iterate_with_timeout(req):
                    if isinstance(ev, Error):
                        status = RunStatus.FAILURE
                        ended_reason = _ended_reason_from_error_type(ev.error_type)
                        if error_msg is None:
                            error_msg = ev.message or "provider reported an error"
                    yield record(ev)
            except TimeoutError:
                status = RunStatus.TIMEOUT
                ended_reason = RunEndedReason.TIMEOUT
                error_msg = "timeout"
                yield record(
                    Error(message=f"run timed out after {req.timeout}s", error_type="timeout")
                )
            except (TransientError, ProviderNotAvailableError, ConfigError) as exc:
                status = RunStatus.FAILURE
                ended_reason = RunEndedReason.ERROR
                if error_msg is None:
                    error_msg = str(exc)
                    yield record(Error(message=str(exc), error_type=type(exc).__name__))
            except Exception as exc:
                status = RunStatus.FAILURE
                ended_reason = RunEndedReason.ERROR
                if error_msg is None:
                    error_msg = str(exc)
                    yield record(Error(message=str(exc), error_type=type(exc).__name__))

            yield finish()
        except GeneratorExit:
            if not finished:
                if error_msg is None and status == RunStatus.SUCCESS:
                    status = RunStatus.CANCELLED
                    ended_reason = RunEndedReason.CANCELLED
                    error_msg = "stream closed before completion"
                    record(Error(message=error_msg, error_type="cancelled"))
                finish()
            raise
        except asyncio.CancelledError:
            if not finished:
                status = RunStatus.CANCELLED
                ended_reason = RunEndedReason.CANCELLED
                error_msg = "cancelled"
                record(Error(message="stream cancelled", error_type="cancelled"))
                finish()
            raise
        finally:
            writer.close()
            if artifacts_dir is not None:
                result = result_state.to_result(
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
                result_path = write_result_artifact(artifacts_dir, result)
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

    def _record_session(self, req: RunRequest, event: AgentEvent) -> None:
        if req.continue_session and isinstance(event, SessionInfo) and event.id:
            self.session_id = event.id
            req.session_id = event.id


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
    normalized = (error_type or "").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized == "max_turns":
        return RunEndedReason.MAX_TURNS
    if normalized == "timeout":
        return RunEndedReason.TIMEOUT
    if normalized == "cancelled":
        return RunEndedReason.CANCELLED
    if normalized in {"refused", "refusal", "policy_refusal", "safety_refusal"}:
        return RunEndedReason.REFUSED
    return RunEndedReason.ERROR


class _ResultState:
    def __init__(self) -> None:
        self.text_parts: list[str] = []
        self.structured: Any = None
        self.usage: TokenUsage | None = None
        self.cost: float | None = None
        self.session_id: str | None = None

    def record(self, env: EventEnvelope) -> None:
        ev = env.event
        if isinstance(ev, Text):
            self.text_parts.append(ev.text)
        elif isinstance(ev, Usage):
            self.usage = ev.usage if self.usage is None else self.usage + ev.usage
            if ev.cost_usd is not None:
                self.cost = ev.cost_usd if self.cost is None else self.cost + ev.cost_usd
        elif isinstance(ev, StructuredOutput):
            self.structured = ev.value
        elif isinstance(ev, SessionInfo):
            self.session_id = ev.id

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
            model=model,
            status=status,
            ended_reason=ended_reason,
            final_text="".join(self.text_parts),
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
