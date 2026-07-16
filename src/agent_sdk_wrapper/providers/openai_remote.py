"""Long-lived Codex session for the remote-control layer.

Uses ``AsyncCodex`` (app-server over stdio): ``turn/start`` sends,
``turn/steer`` steers mid-turn, ``turn/interrupt`` aborts, and
``thread/resume`` restores a thread across process exits.

Approvals: the app-server sends ``item/*/requestApproval`` server requests,
which the SDK dispatches to a sync ``approval_handler`` **on its reader
thread**. While that handler blocks, no notifications or RPC responses flow —
so interrupt() resolves pending approvals as "deny" first, otherwise waiting
on the ``turn/interrupt`` response would deadlock. The async client does not
expose ``approval_handler``, so this adapter installs one on the underlying
sync client (private API; see docs/remote-control.md gaps #1-#3).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import os
from typing import Any

from ..events import Error, SessionInfo, Text, Thinking
from ..remote.protocol import SessionState, UserMessage
from ..remote.sessions import BaseRemoteSession, RemoteSessionError
from .openai_provider import (
    _reasoning_text,
    _tool_events,
    _turn_error_message,
    _turn_failed,
    _usage_event,
)

_APPROVAL_TOOLS = {
    "item/commandExecution/requestApproval": "commandExecution",
    "item/fileChange/requestApproval": "fileChange",
}
_APPROVAL_TIMEOUT_SEC = 1800.0


class OpenAIRemoteSession(BaseRemoteSession):
    provider = "openai"

    def __init__(self, session_id: str, **kwargs: Any) -> None:
        super().__init__(session_id, **kwargs)
        self._codex: Any = None
        self._thread: Any = None
        self._turn: Any = None
        self._turn_task: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def start(self) -> None:
        from openai_codex import AsyncCodex
        from openai_codex.api import AsyncThread
        from openai_codex.client import CodexConfig
        from openai_codex.generated.v2_all import (
            AskForApproval,
            AskForApprovalValue,
            SandboxMode,
            ThreadResumeParams,
            ThreadStartParams,
        )

        self._loop = asyncio.get_running_loop()
        state_env = self._state_env("codex", "CODEX_HOME")
        self._codex = AsyncCodex(
            config=CodexConfig(cwd=self.cwd, env=state_env or None)
        )
        # The async client does not expose approval_handler; install ours on
        # the wrapped sync client so approvals reach the remote UI.
        self._codex._client._sync._approval_handler = self._approval_handler
        await self._codex._ensure_initialized()

        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            await self._codex.login_api_key(api_key)

        # ApprovalMode can only express auto_review/deny_all; build the
        # generated params directly to get human-in-the-loop approvals.
        # on-request only escalates commands that leave the sandbox;
        # "untrusted" prompts for anything not on the trusted list.
        policy = {
            "bypassPermissions": AskForApprovalValue.never,
            "untrusted": AskForApprovalValue.untrusted,
        }.get(self.permission_mode or "", AskForApprovalValue.on_request)
        approval = AskForApproval(root=policy)
        client = self._codex._client
        if self.resume:
            params = ThreadResumeParams(
                thread_id=self.resume,
                approval_policy=approval,
                cwd=self.cwd,
                model=self.model,
                sandbox=SandboxMode.workspace_write,
            )
            resumed = await client.thread_resume(self.resume, params)
            thread_id = resumed.thread.id
        else:
            params = ThreadStartParams(
                approval_policy=approval,
                cwd=self.cwd,
                model=self.model,
                sandbox=SandboxMode.workspace_write,
            )
            started = await client.thread_start(params)
            thread_id = started.thread.id
        self._thread = AsyncThread(self._codex, thread_id)
        self.native_session_id = thread_id
        self._emit(SessionInfo(id=thread_id))
        self._set_state(SessionState.IDLE)

    async def send(self, text: str) -> None:
        if self._thread is None or self.state == SessionState.CLOSED:
            raise RemoteSessionError("session is not running")
        turn_active = self._turn_task is not None and not self._turn_task.done()
        if turn_active and self._turn is not None:
            self._emit(UserMessage(text=text, steered=True))
            try:
                await self._turn.steer(text)
                return
            except Exception:
                # The turn may have completed while steering; fall through and
                # start a new turn instead.
                pass
        self._emit(UserMessage(text=text, steered=False))
        self._turn = await self._thread.turn(text)
        self._turn_task = asyncio.create_task(self._consume_turn(self._turn))
        self._set_state(SessionState.RUNNING)

    async def interrupt(self) -> None:
        if self._thread is None or self.state == SessionState.CLOSED:
            raise RemoteSessionError("session is not running")
        # Unblock the SDK reader thread first: while an approval is pending it
        # cannot deliver the turn/interrupt response (see module docstring).
        self._deny_pending_permissions("interrupted by user")
        if self._turn is not None and self._turn_task is not None and not self._turn_task.done():
            await self._turn.interrupt()

    async def close(self) -> None:
        self._deny_pending_permissions("session closed")
        if self._turn_task is not None:
            self._turn_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._turn_task
            self._turn_task = None
        if self._codex is not None:
            with contextlib.suppress(Exception):
                await self._codex.close()
            self._codex = None
        self._set_state(SessionState.CLOSED)

    async def _consume_turn(self, turn: Any) -> None:
        last_usage: Any = None
        try:
            async for notification in turn.stream():
                method = getattr(notification, "method", "")
                payload = getattr(notification, "payload", None)
                if method == "item/completed":
                    item = getattr(payload, "item", None)
                    root = getattr(item, "root", item)
                    root_type = getattr(root, "type", "")
                    if root_type == "agentMessage":
                        text = getattr(root, "text", "") or ""
                        if text:
                            self._emit(Text(text=text))
                    elif root_type == "reasoning":
                        text = _reasoning_text(root)
                        if text:
                            self._emit(Thinking(text=text))
                    elif root_type == "plan":
                        text = getattr(root, "text", "") or ""
                        if text:
                            self._emit(Thinking(text=text))
                    else:
                        for event in _tool_events(root, notification, False):
                            self._emit(event)
                elif method == "thread/tokenUsage/updated":
                    last_usage = getattr(payload, "token_usage", None) or getattr(
                        payload, "tokenUsage", None
                    )
                elif method == "turn/completed":
                    if last_usage is not None:
                        self._emit(_usage_event(last_usage, False))
                    turn_info = getattr(payload, "turn", None)
                    if _turn_failed(turn_info):
                        self._emit(
                            Error(
                                message=_turn_error_message(turn_info),
                                error_type="turn_failed",
                            )
                        )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._emit(Error(message=str(exc), error_type=type(exc).__name__))
        finally:
            if self._turn is turn:
                self._turn = None
            if self.state != SessionState.CLOSED:
                self._set_state(SessionState.IDLE)

    def _approval_handler(self, method: str, params: dict[str, Any] | None) -> dict[str, Any]:
        """Runs on the SDK reader thread; bridges into the event loop."""
        tool = _APPROVAL_TOOLS.get(method)
        if tool is None:
            return {}
        loop = self._loop
        if loop is None or loop.is_closed():
            return {"decision": "decline"}
        bridge: concurrent.futures.Future[Any] = concurrent.futures.Future()
        holder: dict[str, str] = {}

        def _register() -> None:
            request_id, future = self._create_permission_request(
                tool=tool, tool_input=params
            )
            holder["request_id"] = request_id

            def _done(f: asyncio.Future[Any]) -> None:
                if f.cancelled():
                    bridge.cancel()
                elif f.exception() is not None:
                    bridge.set_exception(f.exception())
                else:
                    bridge.set_result(f.result())

            future.add_done_callback(_done)

        def _cleanup_stale() -> None:
            request_id = holder.get("request_id")
            if request_id and request_id in self._pending_permissions:
                with contextlib.suppress(RemoteSessionError):
                    self.respond_permission(request_id, "deny", "approval timed out")

        try:
            loop.call_soon_threadsafe(_register)
            decision = bridge.result(timeout=_APPROVAL_TIMEOUT_SEC)
        except Exception:
            # Timed out or loop shut down: clear the dangling request so the
            # session does not stay stuck in awaiting_permission.
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(_cleanup_stale)
            return {"decision": "decline"}
        return {"decision": "accept" if decision.allowed else "decline"}
