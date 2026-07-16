"""Control events for the remote-control protocol.

Normalized agent output reuses the :mod:`agent_sdk_wrapper.events` vocabulary.
The types here are session-control events that only exist in the remote layer:
state transitions, user-message echoes, and permission prompts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar

from ..events import _EventBase


class SessionState(StrEnum):
    STARTING = "starting"
    IDLE = "idle"
    RUNNING = "running"
    AWAITING_PERMISSION = "awaiting_permission"
    CLOSED = "closed"


@dataclass
class StateChanged(_EventBase):
    type: ClassVar[str] = "state_changed"
    state: SessionState = SessionState.STARTING


@dataclass
class UserMessage(_EventBase):
    """Echo of a user message accepted by the session (multi-client sync)."""

    type: ClassVar[str] = "user_message"
    text: str = ""
    steered: bool = False


@dataclass
class PermissionRequest(_EventBase):
    type: ClassVar[str] = "permission_request"
    request_id: str = ""
    tool: str | None = None
    input: dict[str, Any] | None = None
    title: str | None = None


@dataclass
class PermissionResolved(_EventBase):
    type: ClassVar[str] = "permission_resolved"
    request_id: str = ""
    behavior: str = "deny"
    message: str | None = None
