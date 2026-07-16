"""Session base class and manager for the remote-control layer.

A :class:`BaseRemoteSession` owns an append-only event log (replayable by
sequence number, so SSE clients can reconnect with ``?since=N``), the session
state machine, and pending permission requests. Provider adapters subclass it
in ``agent_sdk_wrapper.providers`` and translate native SDK streams into
normalized events via :meth:`_emit`.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import AgentSdkWrapperError
from ..events import _EventBase, utcnow_iso
from .protocol import PermissionRequest, PermissionResolved, SessionState, StateChanged


class RemoteSessionError(AgentSdkWrapperError):
    pass


@dataclass
class PermissionDecision:
    behavior: str  # "allow" | "deny"
    message: str | None = None

    @property
    def allowed(self) -> bool:
        return self.behavior == "allow"


class BaseRemoteSession:
    provider = "base"

    def __init__(
        self,
        session_id: str,
        *,
        model: str | None = None,
        cwd: str | None = None,
        resume: str | None = None,
        permission_mode: str | None = None,
        state_dir: str | None = None,
    ) -> None:
        self.id = session_id
        self.model = model
        self.cwd = cwd
        self.resume = resume
        self.permission_mode = permission_mode
        self.state_dir = state_dir
        self.created_at = utcnow_iso()
        self.state = SessionState.STARTING
        self.native_session_id: str | None = resume
        self._events: list[dict[str, Any]] = []
        self._waiters: set[asyncio.Future[None]] = set()
        self._pending_permissions: dict[str, asyncio.Future[PermissionDecision]] = {}

    # -- lifecycle (implemented by provider adapters) -------------------------

    async def start(self) -> None:
        raise NotImplementedError

    async def send(self, text: str) -> None:
        """Send a user message: starts a turn when idle, steers when running."""
        raise NotImplementedError

    async def interrupt(self) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        raise NotImplementedError

    # -- event log -------------------------------------------------------------

    def _emit(self, event: _EventBase | dict[str, Any]) -> None:
        """Append an event and wake SSE subscribers. Event-loop thread only."""
        payload = event if isinstance(event, dict) else event.to_dict()
        self._events.append(
            {
                "seq": len(self._events) + 1,
                "timestamp": utcnow_iso(),
                "event": payload,
            }
        )
        for waiter in self._waiters:
            if not waiter.done():
                waiter.set_result(None)
        self._waiters.clear()

    def _set_state(self, state: SessionState) -> None:
        if state == self.state:
            return
        self.state = state
        self._emit(StateChanged(state=state))

    @property
    def events(self) -> list[dict[str, Any]]:
        return self._events

    async def events_since(self, since: int = 0):
        """Yield event entries with ``seq > since``; live-follows until closed."""
        index = max(since, 0)
        while True:
            while index < len(self._events):
                entry = self._events[index]
                index += 1
                yield entry
            if self.state == SessionState.CLOSED:
                return
            waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            self._waiters.add(waiter)
            try:
                await waiter
            finally:
                self._waiters.discard(waiter)

    # -- permissions -----------------------------------------------------------

    def _create_permission_request(
        self,
        *,
        tool: str | None,
        tool_input: dict[str, Any] | None,
        title: str | None = None,
    ) -> tuple[str, asyncio.Future[PermissionDecision]]:
        request_id = uuid.uuid4().hex[:12]
        future: asyncio.Future[PermissionDecision] = (
            asyncio.get_running_loop().create_future()
        )
        self._pending_permissions[request_id] = future
        self._emit(
            PermissionRequest(
                request_id=request_id, tool=tool, input=tool_input, title=title
            )
        )
        self._set_state(SessionState.AWAITING_PERMISSION)
        return request_id, future

    def respond_permission(
        self, request_id: str, behavior: str, message: str | None = None
    ) -> None:
        if behavior not in ("allow", "deny"):
            raise RemoteSessionError(f"invalid permission behavior: {behavior!r}")
        future = self._pending_permissions.pop(request_id, None)
        if future is None or future.done():
            raise RemoteSessionError(f"no pending permission request {request_id!r}")
        future.set_result(PermissionDecision(behavior=behavior, message=message))
        self._emit(
            PermissionResolved(request_id=request_id, behavior=behavior, message=message)
        )
        # Only resume RUNNING from AWAITING_PERMISSION; a session that closed
        # (e.g. turn errored) while a permission was pending must stay closed.
        if not self._pending_permissions and self.state == SessionState.AWAITING_PERMISSION:
            self._set_state(SessionState.RUNNING)

    def _deny_pending_permissions(self, message: str) -> None:
        for request_id in list(self._pending_permissions):
            try:
                self.respond_permission(request_id, "deny", message)
            except RemoteSessionError:
                pass

    def pending_permission_ids(self) -> list[str]:
        return list(self._pending_permissions)

    # -- state dir ---------------------------------------------------------------

    def _state_env(self, subdir: str, var: str) -> dict[str, str]:
        """Env override pointing the provider runtime at <state_dir>/<subdir>."""
        if not self.state_dir:
            return {}
        path = Path(self.state_dir).expanduser().resolve() / subdir
        path.mkdir(parents=True, exist_ok=True)
        return {var: str(path)}

    # -- description -----------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        return {
            "session_id": self.id,
            "provider": self.provider,
            "model": self.model,
            "cwd": self.cwd,
            "state": self.state.value,
            "native_session_id": self.native_session_id,
            "created_at": self.created_at,
            "num_events": len(self._events),
            "pending_permissions": self.pending_permission_ids(),
        }


def _default_factories() -> dict[str, Any]:
    from ..providers.anthropic_remote import AnthropicRemoteSession
    from ..providers.openai_remote import OpenAIRemoteSession

    return {"anthropic": AnthropicRemoteSession, "openai": OpenAIRemoteSession}


class SessionManager:
    def __init__(
        self,
        factories: dict[str, Any] | None = None,
        *,
        state_dir: str | None = None,
    ) -> None:
        self._factories = factories
        self._sessions: dict[str, BaseRemoteSession] = {}
        self.state_dir = state_dir

    @property
    def factories(self) -> dict[str, Any]:
        if self._factories is None:
            self._factories = _default_factories()
        return self._factories

    async def create(
        self,
        provider: str,
        *,
        model: str | None = None,
        cwd: str | None = None,
        resume: str | None = None,
        permission_mode: str | None = None,
    ) -> BaseRemoteSession:
        from ..request import normalize_provider

        if provider in self.factories:
            name = provider
        else:
            name = normalize_provider(provider)
        factory = self.factories.get(name)
        if factory is None:
            raise RemoteSessionError(f"no remote session factory for provider {name!r}")
        session_id = uuid.uuid4().hex[:12]
        session = factory(
            session_id,
            model=model,
            cwd=cwd,
            resume=resume,
            permission_mode=permission_mode,
            state_dir=self.state_dir,
        )
        try:
            await session.start()
        except Exception:
            with contextlib.suppress(Exception):
                await session.close()
            raise
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> BaseRemoteSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    def list(self) -> list[dict[str, Any]]:
        return [s.describe() for s in self._sessions.values()]

    async def close(self, session_id: str) -> None:
        session = self.get(session_id)
        await session.close()
        del self._sessions[session_id]

    async def close_all(self) -> None:
        for session_id in list(self._sessions):
            try:
                await self.close(session_id)
            except Exception:
                self._sessions.pop(session_id, None)
