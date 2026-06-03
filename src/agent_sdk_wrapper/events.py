"""Normalized event model shared by every provider.

Both the Claude Agent SDK and the OpenAI Codex SDK emit their own streaming
event shapes. Each provider adapter translates those into the small, stable set
of events defined here, so callers see one vocabulary regardless of backend.

Every event a run produces is wrapped in an :class:`EventEnvelope` (run id,
monotonic sequence, timestamp) and can be serialized to a single JSONL line.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, StrEnum
from typing import Any, ClassVar


def utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


class RunStatus(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class RunEndedReason(StrEnum):
    UNKNOWN = "unknown"
    SUCCESS = "success"
    MAX_TURNS = "max_turns"
    TIMEOUT = "timeout"
    REFUSED = "refused"
    ERROR = "error"
    CANCELLED = "cancelled"


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of arbitrary values into JSON-serializable form."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "model_dump"):  # pydantic BaseModel
        try:
            return value.model_dump(mode="json")
        except Exception:
            return value.model_dump()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_output_tokens: int = 0
    requests: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            reasoning_output_tokens=(
                self.reasoning_output_tokens + other.reasoning_output_tokens
            ),
            requests=self.requests + other.requests,
        )


class _EventBase:
    """Mixin giving every event a stable ``type`` tag and ``to_dict``."""

    type: ClassVar[str]

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.type}
        for f in dataclasses.fields(self):  # type: ignore[arg-type]
            value = getattr(self, f.name)
            if value is None:
                continue
            out[f.name] = _jsonable(value)
        return out


@dataclass
class RunStarted(_EventBase):
    type: ClassVar[str] = "run_started"
    provider: str = ""
    model: str | None = None
    cwd: str | None = None


@dataclass
class Text(_EventBase):
    """A completed assistant text item.

    One event per finalized assistant message item. Codex deltas are buffered
    inside the adapter; Anthropic partial frames are explicitly disabled. The
    name parallels the sibling :class:`Thinking` event.
    """

    type: ClassVar[str] = "text"
    text: str = ""
    raw: dict[str, Any] | None = None


@dataclass
class Thinking(_EventBase):
    """A completed model reasoning / extended-thinking item."""

    type: ClassVar[str] = "thinking"
    text: str = ""
    raw: dict[str, Any] | None = None


@dataclass
class ToolCall(_EventBase):
    type: ClassVar[str] = "tool_call"
    id: str | None = None
    name: str | None = None
    input: dict[str, Any] | None = None
    raw: dict[str, Any] | None = None


@dataclass
class ToolResult(_EventBase):
    type: ClassVar[str] = "tool_result"
    id: str | None = None
    output: str | None = None
    is_error: bool = False
    raw: dict[str, Any] | None = None


@dataclass
class AgentUpdated(_EventBase):
    """The active (sub)agent changed."""

    type: ClassVar[str] = "agent_updated"
    name: str = ""


@dataclass
class Usage(_EventBase):
    type: ClassVar[str] = "usage"
    usage: TokenUsage = field(default_factory=TokenUsage)
    cost_usd: float | None = None
    raw: dict[str, Any] | None = None


@dataclass
class SessionInfo(_EventBase):
    """An underlying provider session/thread identifier."""

    type: ClassVar[str] = "session_info"
    id: str = ""


@dataclass
class StructuredOutput(_EventBase):
    """The validated structured result, when an output schema was requested."""

    type: ClassVar[str] = "structured_output"
    value: Any = None


@dataclass
class WarningEvent(_EventBase):
    type: ClassVar[str] = "warning"
    message: str = ""
    raw: dict[str, Any] | None = None


@dataclass
class Error(_EventBase):
    type: ClassVar[str] = "error"
    message: str = ""
    error_type: str | None = None
    raw: dict[str, Any] | None = None


@dataclass
class RunFinished(_EventBase):
    type: ClassVar[str] = "run_finished"
    status: RunStatus = RunStatus.SUCCESS
    duration_ms: int = 0
    ended_reason: RunEndedReason = RunEndedReason.UNKNOWN


AgentEvent = (
    RunStarted
    | Text
    | Thinking
    | ToolCall
    | ToolResult
    | AgentUpdated
    | Usage
    | SessionInfo
    | StructuredOutput
    | WarningEvent
    | Error
    | RunFinished
)


@dataclass
class EventEnvelope:
    run_id: str
    sequence: int
    timestamp: str
    event: AgentEvent

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "event": self.event.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class RunResult:
    """The collected outcome of a completed (non-streamed) run."""

    run_id: str
    provider: str
    status: RunStatus
    ended_reason: RunEndedReason = RunEndedReason.UNKNOWN
    final_text: str = ""
    model: str | None = None
    structured_output: Any = None
    usage: TokenUsage | None = None
    cost_usd: float | None = None
    duration_ms: int = 0
    session_id: str | None = None
    artifacts_dir: str | None = None
    error: str | None = None
    events: list[EventEnvelope] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == RunStatus.SUCCESS

    def tool_calls(self) -> list[ToolCall]:
        return [e.event for e in self.events if isinstance(e.event, ToolCall)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "provider": self.provider,
            "model": self.model,
            "status": self.status.value,
            "ended_reason": self.ended_reason.value,
            "final_text": self.final_text,
            "structured_output": _jsonable(self.structured_output),
            "usage": _jsonable(self.usage),
            "cost_usd": self.cost_usd,
            "duration_ms": self.duration_ms,
            "session_id": self.session_id,
            "artifacts_dir": self.artifacts_dir,
            "error": self.error,
            "events": [e.to_dict() for e in self.events],
        }

    def __str__(self) -> str:
        return self.final_text
