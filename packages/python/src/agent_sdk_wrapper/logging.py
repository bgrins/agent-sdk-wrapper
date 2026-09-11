"""Logging: a JSONL trace writer plus structured stdlib logging.

Every event a run produces is written to the trace file (full fidelity) and
emitted through the stdlib ``agent_sdk_wrapper`` logger. Lifecycle events
(run_started, tool_call, error, run_finished) log at INFO; everything else at
DEBUG, so the default console stays quiet while the JSONL keeps everything.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TextIO

from .events import (
    Error,
    EventEnvelope,
    RunFinished,
    RunStarted,
    Text,
    Thinking,
    ToolCall,
    ToolResult,
    Usage,
    WarningEvent,
)

LOGGER_NAME = "agent_sdk_wrapper"
_INFO_EVENTS = (RunStarted, ToolCall, ToolResult, Error, WarningEvent, RunFinished)


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def _truncate(text: str, limit: int = 200) -> str:
    text = text.replace("\n", "\\n")
    return text if len(text) <= limit else text[:limit] + f"...(+{len(text) - limit} chars)"


def summarize(env: EventEnvelope) -> str:
    """A concise one-line human description of an event."""
    ev = env.event
    if isinstance(ev, RunStarted):
        return f"run_started provider={ev.provider} model={ev.model}"
    if isinstance(ev, Text):
        return f"text {_truncate(ev.text)!r}"
    if isinstance(ev, Thinking):
        return f"thinking {_truncate(ev.text)!r}"
    if isinstance(ev, ToolCall):
        return f"tool_call {ev.name} input={_truncate(str(ev.input))}"
    if isinstance(ev, ToolResult):
        flag = " (error)" if ev.is_error else ""
        return f"tool_result{flag} {_truncate(ev.output or '')!r}"
    if isinstance(ev, Usage):
        u = ev.usage
        return f"usage in={u.input_tokens} out={u.output_tokens} cost={ev.cost_usd}"
    if isinstance(ev, Error):
        return f"error [{ev.error_type}] {ev.message}"
    if isinstance(ev, WarningEvent):
        return f"warning {ev.message}"
    if isinstance(ev, RunFinished):
        return f"run_finished status={ev.status.value} duration_ms={ev.duration_ms}"
    return ev.type


class TraceWriter:
    """Writes every event to a JSONL file and the stdlib logger.

    Use as a context manager or call :meth:`close` when done.
    """

    def __init__(
        self,
        trace_file: str | Path | None = None,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self._logger = logger or get_logger()
        self._fh: TextIO | None = None
        self._path = Path(trace_file) if trace_file else None
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self._path.open("w", encoding="utf-8")

    def write(self, env: EventEnvelope) -> None:
        if self._fh is not None:
            self._fh.write(env.to_json())
            self._fh.write("\n")
            self._fh.flush()
        level = logging.INFO if isinstance(env.event, _INFO_EVENTS) else logging.DEBUG
        self._logger.log(level, "[%s] %s", env.run_id[:8], summarize(env))

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> TraceWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
