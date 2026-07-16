"""Long-lived Claude session for the remote-control layer.

Uses ``ClaudeSDKClient`` in streaming-input mode: ``query()`` sends/steers,
``interrupt()`` aborts the turn, ``can_use_tool`` proxies permission prompts
to the remote UI, and ``resume=`` restores a session across process exits.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from ..events import Error, SessionInfo, Text, Thinking, ToolCall, ToolResult
from ..remote.protocol import SessionState, UserMessage
from ..remote.sessions import BaseRemoteSession, RemoteSessionError
from .anthropic_provider import _stringify, _usage_event


class AnthropicRemoteSession(BaseRemoteSession):
    provider = "anthropic"

    def __init__(self, session_id: str, **kwargs: Any) -> None:
        super().__init__(session_id, **kwargs)
        self._client: Any = None
        self._reader_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

        kwargs: dict[str, Any] = {
            "permission_mode": self.permission_mode or "default",
            "can_use_tool": self._can_use_tool,
        }
        state_env = self._state_env("claude", "CLAUDE_CONFIG_DIR")
        if state_env:
            kwargs["env"] = state_env
        if self.model:
            kwargs["model"] = self.model
        if self.cwd:
            kwargs["cwd"] = self.cwd
        if self.resume:
            kwargs["resume"] = self.resume
        self._client = ClaudeSDKClient(options=ClaudeAgentOptions(**kwargs))
        await self._client.connect()
        self._reader_task = asyncio.create_task(self._read_loop())
        self._set_state(SessionState.IDLE)

    async def send(self, text: str) -> None:
        if self._client is None or self.state == SessionState.CLOSED:
            raise RemoteSessionError("session is not running")
        steered = self.state in (SessionState.RUNNING, SessionState.AWAITING_PERMISSION)
        self._emit(UserMessage(text=text, steered=steered))
        await self._client.query(text)
        if self.state == SessionState.IDLE:
            self._set_state(SessionState.RUNNING)

    async def interrupt(self) -> None:
        if self._client is None or self.state == SessionState.CLOSED:
            raise RemoteSessionError("session is not running")
        self._deny_pending_permissions("interrupted by user")
        await self._client.interrupt()

    async def close(self) -> None:
        self._deny_pending_permissions("session closed")
        if self._reader_task is not None:
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task
            self._reader_task = None
        if self._client is not None:
            with contextlib.suppress(Exception):
                await self._client.disconnect()
            self._client = None
        self._set_state(SessionState.CLOSED)

    async def _read_loop(self) -> None:
        try:
            async for message in self._client.receive_messages():
                self._handle_message(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._emit(Error(message=str(exc), error_type=type(exc).__name__))
        finally:
            # Runs on error, on clean CLI exit, and on cancellation via
            # close(); in every case the session is dead, so unblock pending
            # permission callbacks and let SSE followers terminate.
            self._deny_pending_permissions("session ended")
            self._set_state(SessionState.CLOSED)

    def _handle_message(self, message: Any) -> None:
        from claude_agent_sdk import (
            AssistantMessage,
            ResultMessage,
            SystemMessage,
            TextBlock,
            ThinkingBlock,
            ToolResultBlock,
            ToolUseBlock,
        )
        from claude_agent_sdk import (
            UserMessage as SdkUserMessage,
        )

        if isinstance(message, AssistantMessage):
            if self.state == SessionState.IDLE:
                self._set_state(SessionState.RUNNING)
            for block in message.content:
                if isinstance(block, TextBlock):
                    self._emit(Text(text=block.text))
                elif isinstance(block, ThinkingBlock):
                    self._emit(Thinking(text=block.thinking))
                elif isinstance(block, ToolUseBlock):
                    self._emit(ToolCall(id=block.id, name=block.name, input=block.input))
        elif isinstance(message, SdkUserMessage):
            content = message.content
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, ToolResultBlock):
                        self._emit(
                            ToolResult(
                                id=block.tool_use_id,
                                output=_stringify(block.content),
                                is_error=bool(block.is_error),
                            )
                        )
        elif isinstance(message, SystemMessage):
            if isinstance(message.data, dict):
                sid = message.data.get("session_id")
                if sid and sid != self.native_session_id:
                    self.native_session_id = sid
                    self._emit(SessionInfo(id=sid))
        elif isinstance(message, ResultMessage):
            if message.session_id and message.session_id != self.native_session_id:
                self.native_session_id = message.session_id
                self._emit(SessionInfo(id=message.session_id))
            if message.usage:
                self._emit(_usage_event(message.usage, message.total_cost_usd))
            if message.is_error:
                self._emit(
                    Error(
                        message=message.result or "run reported an error",
                        error_type="result_error",
                    )
                )
            self._set_state(SessionState.IDLE)

    async def _can_use_tool(self, tool_name: str, tool_input: dict[str, Any], context: Any):
        from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

        _, future = self._create_permission_request(
            tool=tool_name,
            tool_input=tool_input,
            title=getattr(context, "title", None),
        )
        decision = await future
        if decision.allowed:
            return PermissionResultAllow()
        return PermissionResultDeny(message=decision.message or "denied via remote control")
