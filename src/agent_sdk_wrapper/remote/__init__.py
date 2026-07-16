"""Remote-control layer: long-lived agent sessions behind an HTTP+SSE protocol.

See ``docs/remote-control.md`` for the protocol and provider mapping.
"""

from .protocol import (
    PermissionRequest,
    PermissionResolved,
    SessionState,
    StateChanged,
    UserMessage,
)
from .sessions import BaseRemoteSession, RemoteSessionError, SessionManager

__all__ = [
    "BaseRemoteSession",
    "PermissionRequest",
    "PermissionResolved",
    "RemoteSessionError",
    "SessionManager",
    "SessionState",
    "StateChanged",
    "UserMessage",
]
