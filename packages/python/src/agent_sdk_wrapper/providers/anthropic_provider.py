"""Claude Agent SDK adapter. The SDK manages the runtime and API retries."""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import functools
import json
import os
import platform
import shutil
import sys
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import claude_agent_sdk
from claude_agent_sdk import (
    TERMINAL_TASK_STATUSES,
    AgentDefinition,
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKError,
    CLIConnectionError,
    CLIJSONDecodeError,
    CLINotFoundError,
    ProcessError,
    RateLimitEvent,
    ResultMessage,
    ServerToolResultBlock,
    ServerToolUseBlock,
    StreamEvent,
    SystemMessage,
    TaskNotificationMessage,
    TaskStartedMessage,
    TaskUpdatedMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from ..artifacts import ProviderEventLogger
from ..classify import TRANSIENT, classify
from ..errors import (
    AgentSdkWrapperError,
    ConfigError,
    ProcessTerminatedError,
    ProviderNotAvailableError,
    TransientError,
)
from ..events import (
    AgentEvent,
    ContextCompacted,
    Error,
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
)
from ..mcp import McpHttpServer, McpServer, McpStdioServer, stdio_server_env
from ..request import RunRequest, normalize_effort_for_provider
from ..structured import json_schema_of_type, validate_output
from ..tools import json_schema_for, to_anthropic_tools, validate_tool_names
from .base import ProviderAdapter

_DEFAULT_THINKING: dict[str, str] = {"type": "adaptive", "display": "summarized"}
_WEB_TOOL_NAMES: tuple[str, ...] = ("WebSearch", "WebFetch")
_SETTING_SOURCES = frozenset({"user", "project", "local"})
# The SDK's 1 MiB default fails on single frames such as a base64 image read.
_DEFAULT_MAX_BUFFER_SIZE = 16 * 1024 * 1024
_STDERR_TAIL_LINES = 50
# The CLI ranks this above --effort, so an inherited value would override the request.
_EFFORT_ENV = "CLAUDE_CODE_EFFORT_LEVEL"
# Background subagents add a follow-up turn and a second result frame.
_BACKGROUND_TASKS_ENV = "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"
# claude.ai login tokens the CLI reads from the env; it treats empty values as unset.
_LOGIN_TOKEN_ENV = (
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR",
    "CLAUDE_CODE_OAUTH_REFRESH_TOKEN",
    "CLAUDE_CODE_SESSION_ACCESS_TOKEN",
)
_API_KEY_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
# The CLI enables CLAUDE_CODE_USE_* provider flags only for these values.
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
# The CLI ranks these above every stored login (claude.ai, OAuth token, Console profile).
_PROVIDER_FLAG_ENV = (
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
    "CLAUDE_CODE_USE_ANTHROPIC_AWS",
    "CLAUDE_CODE_USE_ANTHROPIC_GOOGLE_CLOUD",
    "CLAUDE_CODE_USE_MANTLE",
    "CLAUDE_CODE_USE_GATEWAY",
)
_SYNTHETIC_MODEL = "<synthetic>"
_SUBAGENT_TASK_TYPES = frozenset({"local_agent", "remote_agent"})
# Native keys first-class options compute; extra_options may set one only when
# its option is unused. The wrapper always computes env.
_WRAPPER_OWNED_OPTIONS: dict[str, Callable[[RunRequest], bool]] = {
    "agents": lambda req: bool(req.subagents),
    "allowed_tools": lambda req: bool(
        req.allowed_tools or req.tools or req.mcp_servers or req.subagents
    ),
    "cwd": lambda req: req.cwd is not None,
    "disallowed_tools": lambda req: bool(
        req.disallowed_tools or req.mcp_servers or req.web_tools is False
    ),
    "effort": lambda req: req.effort is not None,
    "env": lambda req: True,
    "max_turns": lambda req: req.max_turns is not None,
    "mcp_servers": lambda req: bool(req.tools or req.mcp_servers),
    "model": lambda req: req.model is not None,
    "output_format": lambda req: req.output_schema is not None,
    "permission_mode": lambda req: req.permission_mode is not None,
    "resume": lambda req: bool(req.session_id),
    "setting_sources": lambda req: req.setting_sources is not None,
    "system_prompt": lambda req: req.system_prompt is not None,
}

_SUBTYPE_ERRORS = {
    "error_max_turns": "max_turns",
    "error_max_budget_usd": "max_budget",
    "error_max_structured_output_retries": "structured_output_failed",
    "error_during_execution": "execution_error",
}
_TERMINAL_REASON_ERRORS = {
    "max_turns": "max_turns",
    "budget_exhausted": "max_budget",
    "structured_output_retry_exhausted": "structured_output_failed",
    "prompt_too_long": "context_window_exceeded",
    "aborted_streaming": "cancelled",
    "aborted_tools": "cancelled",
    # The CLI groups these with prompt_too_long as context limits.
    "blocking_limit": "context_window_exceeded",
    "rapid_refill_breaker": "context_window_exceeded",
}
# Other reasons (hook stops, deferred tools) end runs the CLI reports as successful.
_FAILURE_TERMINAL_REASONS = frozenset({
    "api_error",
    "image_error",
    "malformed_tool_use_exhausted",
    "model_error",
    "tool_deferred_unavailable",
    "turn_setup_failed",
})
# AssistantMessage.error values; "unknown" and "invalid_request" defer to other signals.
_ASSISTANT_ERRORS = {
    "authentication_failed": "authentication_failed",
    "oauth_org_not_allowed": "permission_denied",
    "verification_required": "authentication_failed",
    "cloud_credential_error": "authentication_failed",
    "account_on_hold": "permission_denied",
    "billing_error": "billing_error",
    "rate_limit": "transient_api_error",
    "overloaded": "transient_api_error",
    "server_error": "transient_api_error",
    "model_not_found": "model_not_found",
    "max_output_tokens": "execution_error",
}
# Signal exits use 128 + signum in the SDK protocol, or -signum in asyncio.
_SIGNALS_BY_EXIT_CODE: dict[int, int] = {137: 9, 143: 15, 130: 2, -9: 9, -15: 15, -2: 2}


def _raw(obj: Any) -> dict[str, Any] | None:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        try:
            return dataclasses.asdict(obj)
        except Exception:
            return None
    return None


def _bundled_cli_path() -> Path:
    cli_name = "claude.exe" if platform.system() == "Windows" else "claude"
    return Path(claude_agent_sdk.__file__).parent / "_bundled" / cli_name


class AnthropicProvider(ProviderAdapter):
    name = "anthropic"

    def __init__(self, *, cli_path: str | None = None) -> None:
        self._cli_path = cli_path

    def ensure_available(self) -> None:
        if self._cli_path is not None:
            return
        if _bundled_cli_path().exists() or shutil.which("claude"):
            return
        raise ProviderNotAvailableError(
            "Claude Code runtime was not found. The Claude Agent SDK bundles it "
            "on supported wheels, or install Claude Code on PATH / pass "
            "provider_options={'cli_path': ...}."
        )

    def check_credentials(self, req: RunRequest) -> str | None:
        env = {**os.environ, **req.env}
        if any(env.get(name, "").strip() for name in _API_KEY_ENV):
            return None
        if any(env.get(name, "").strip().lower() in _TRUE_VALUES for name in _PROVIDER_FLAG_ENV):
            return None
        return (
            "no Claude API credentials: set ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN or a "
            "cloud-provider flag (CLAUDE_CODE_USE_BEDROCK/VERTEX/FOUNDRY). "
            "cli_login='deny' never uses a stored claude.ai login"
        )

    def validate_request(self, req: RunRequest) -> None:
        effort = normalize_effort_for_provider("anthropic", req.effort)
        validate_tool_names(req.tools)
        for fn in req.tools:
            json_schema_for(fn)
        if req.cli_login == "require":
            raise ConfigError(
                "cli_login='require' is not supported for Claude; use an API key, "
                "auth token or cloud-provider credentials"
            )
        login_tokens = [name for name in _LOGIN_TOKEN_ENV if req.env.get(name)]
        if login_tokens:
            raise ConfigError(
                f"env {login_tokens} carry claude.ai login tokens; "
                "Claude runs use API-key or cloud-provider credentials"
            )
        unknown = sorted(set(req.extra_options) - _native_option_names())
        if unknown:
            raise ConfigError(f"extra_options {unknown} are not Claude Agent SDK options")
        active_mcp_servers = [
            server for server in req.mcp_servers if server.enabled is not False
        ]
        _validate_anthropic_mcp_servers(active_mcp_servers)
        if req.max_turns is not None and req.max_turns < 1:
            raise ConfigError("max_turns must be at least 1")
        if req.setting_sources is not None:
            invalid = [s for s in req.setting_sources if s not in _SETTING_SOURCES]
            if invalid:
                raise ConfigError(
                    f"unknown setting_sources {invalid}; expected any of "
                    f"{sorted(_SETTING_SOURCES)}"
                )
        owned = sorted(
            key
            for key in req.extra_options
            if key in _WRAPPER_OWNED_OPTIONS and _WRAPPER_OWNED_OPTIONS[key](req)
        )
        if owned:
            raise ConfigError(
                f"extra_options {owned} conflict with first-class Agent options that set them"
            )
        if req.builtin_tools is not None and "tools" in req.extra_options:
            raise ConfigError(
                "builtin_tools cannot be combined with extra_options['tools']"
            )
        if req.extra_options.get("include_partial_messages"):
            raise ConfigError(
                "extra_options['include_partial_messages']=True is not supported "
                "by agent-sdk-wrapper. Claude Agent SDK partial StreamEvent "
                "frames duplicate later complete AssistantMessage blocks; the "
                "wrapper exposes the complete messages instead."
            )
        inherited_effort = req.env.get(_EFFORT_ENV)
        if effort and inherited_effort is not None and inherited_effort != effort:
            raise ConfigError(
                f"effort={effort!r} conflicts with env[{_EFFORT_ENV!r}]={inherited_effort!r}"
            )
        if req.web_tools is True:
            if req.builtin_tools == "none":
                raise ConfigError("web_tools=True conflicts with builtin_tools='none'")
            blocked = [name for name in _WEB_TOOL_NAMES if name in req.disallowed_tools]
            if blocked:
                raise ConfigError(f"web_tools=True conflicts with disallowed_tools {blocked}")

    def _build_options(
        self, req: RunRequest, stderr_tail: collections.deque[str] | None = None
    ) -> ClaudeAgentOptions:
        self.validate_request(req)

        allowed = list(req.allowed_tools)
        disallowed = list(req.disallowed_tools)
        builtin: list[str] | None = None
        if req.builtin_tools is not None:
            builtin = [] if req.builtin_tools == "none" else list(req.builtin_tools)

        if req.web_tools is False:
            disallowed.extend(name for name in _WEB_TOOL_NAMES if name not in disallowed)
        elif req.web_tools is True and builtin is not None:
            builtin.extend(name for name in _WEB_TOOL_NAMES if name not in builtin)

        active_mcp_servers = [
            server for server in req.mcp_servers if server.enabled is not False
        ]
        mcp_servers: dict[str, Any] = {}
        server, tool_names = to_anthropic_tools(req.tools)
        if server is not None:
            mcp_servers["agent_sdk_wrapper_tools"] = server
            allowed.extend(tool_names)
        mcp_servers.update(_anthropic_mcp_servers(active_mcp_servers))
        allowed.extend(_anthropic_tool_names(active_mcp_servers, enabled=True))
        disallowed.extend(_anthropic_tool_names(active_mcp_servers, enabled=False))

        agents = None
        if req.subagents:
            agents = {
                name: AgentDefinition(
                    description=sub.description,
                    prompt=sub.prompt,
                    tools=sub.tools,
                    model=sub.model,
                    maxTurns=sub.max_turns,
                )
                for name, sub in req.subagents.items()
            }
            # The delegation tool must be both available and approved.
            if "Agent" not in allowed:
                allowed.append("Agent")
            if builtin is not None and "Agent" not in builtin:
                builtin.append("Agent")

        effort = normalize_effort_for_provider("anthropic", req.effort)
        env = dict(req.env)
        # An empty value keeps an inherited effort from overriding the CLI default.
        env.setdefault(_EFFORT_ENV, effort or "")
        env.setdefault(_BACKGROUND_TASKS_ENV, "1")
        for name in _LOGIN_TOKEN_ENV:
            env[name] = ""

        extra = dict(req.extra_options)
        user_stderr = extra.pop("stderr", None)

        kwargs: dict[str, Any] = {
            "model": req.model,
            "system_prompt": req.system_prompt,
            "max_turns": req.max_turns,
            "effort": effort,
            "cwd": req.cwd,
            "env": env,
            "allowed_tools": allowed,
            "disallowed_tools": disallowed,
            "permission_mode": req.permission_mode,
            "mcp_servers": mcp_servers,
            "setting_sources": [] if req.setting_sources is None else list(req.setting_sources),
            "max_buffer_size": _DEFAULT_MAX_BUFFER_SIZE,
        }
        if stderr_tail is not None or user_stderr is not None:
            kwargs["stderr"] = _stderr_callback(stderr_tail, user_stderr)
        if req.session_id:
            kwargs["resume"] = req.session_id
        if self._cli_path is not None:
            kwargs["cli_path"] = self._cli_path
        if agents:
            kwargs["agents"] = agents
        if req.output_schema is not None:
            kwargs["output_format"] = {
                "type": "json_schema",
                "schema": json_schema_of_type(req.output_schema),
            }
        if builtin is not None:
            kwargs["tools"] = builtin
        if "thinking" not in extra:
            kwargs["thinking"] = dict(_DEFAULT_THINKING)
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        kwargs.update(extra)
        return ClaudeAgentOptions(**kwargs)

    async def stream(self, req: RunRequest) -> AsyncIterator[AgentEvent]:
        self.validate_request(req)
        self.ensure_available()
        problem = self.check_credentials(req)
        if problem:
            yield Error(message=problem, error_type="authentication_failed")
            return

        stderr_tail: collections.deque[str] = collections.deque(maxlen=_STDERR_TAIL_LINES)
        options = self._build_options(req, stderr_tail)
        seen_session = False
        seen_text = False
        seen_thinking = False
        # After a retraction only the result's usage is still meaningful.
        retracted = False
        session_model: str | None = None
        seen_uuids: set[str] = set()
        pending = _PendingText()
        # The latest error-bearing assistant message: (AssistantMessage.error, its text).
        assistant_error: tuple[str | None, str] | None = None
        # Map tool_use_id to the tool name for result events.
        tool_names: dict[str, str] = {}
        subagent_tasks: set[str] = set()
        provider_log = ProviderEventLogger(
            "anthropic",
            req.artifacts_dir,
            req.on_provider_event,
            run_id=req.run_id,
        )

        try:
            try:
                async with contextlib.aclosing(
                    claude_agent_sdk.query(prompt=req.prompt, options=options)
                ) as messages:
                    async for message in messages:
                        provider_log.write(message)
                        if retracted and not isinstance(message, ResultMessage):
                            continue
                        if not pending.continues(message):
                            text = pending.flush()
                            if text is not None:
                                seen_text = True
                                yield text
                        if isinstance(message, AssistantMessage):
                            if message.uuid is not None:
                                if message.uuid in seen_uuids:
                                    continue
                                seen_uuids.add(message.uuid)
                            if message.parent_tool_use_id:
                                yield WarningEvent(
                                    message="Subagent message omitted from portable output; "
                                    "inspect on_provider_event"
                                )
                                continue
                            if message.error is not None or message.model == _SYNTHETIC_MODEL:
                                text = _message_text(message)
                                assistant_error = (message.error, text)
                                if text:
                                    yield WarningEvent(message=text)
                                continue
                            # A later normal message means the earlier API error was recovered.
                            assistant_error = None
                            for event in _assistant_events(
                                message, tool_names, req.include_raw, pending
                            ):
                                seen_text = seen_text or isinstance(event, Text)
                                seen_thinking = seen_thinking or isinstance(event, Thinking)
                                yield event
                        elif isinstance(message, UserMessage):
                            content = message.content
                            if message.parent_tool_use_id or not isinstance(content, list):
                                continue
                            for block in content:
                                if isinstance(block, ToolResultBlock):
                                    yield ToolResult(
                                        id=block.tool_use_id,
                                        name=tool_names.get(block.tool_use_id),
                                        output=_stringify(block.content),
                                        is_error=bool(block.is_error),
                                        raw=_raw(block) if req.include_raw else None,
                                    )
                        elif isinstance(message, TaskStartedMessage):
                            subagent_type = message.data.get("subagent_type")
                            if message.task_type in _SUBAGENT_TASK_TYPES or subagent_type:
                                subagent_tasks.add(message.task_id)
                                yield SubagentStarted(
                                    task_id=message.task_id,
                                    name=subagent_type or message.task_type or "",
                                    description=message.description,
                                )
                        elif isinstance(message, TaskNotificationMessage):
                            if message.task_id in subagent_tasks:
                                subagent_tasks.discard(message.task_id)
                                yield SubagentEnded(
                                    task_id=message.task_id,
                                    status=message.status,
                                    summary=message.summary,
                                )
                        elif isinstance(message, TaskUpdatedMessage):
                            if (
                                message.status in TERMINAL_TASK_STATUSES
                                and message.task_id in subagent_tasks
                            ):
                                subagent_tasks.discard(message.task_id)
                                yield SubagentEnded(task_id=message.task_id, status=message.status)
                        elif isinstance(message, SystemMessage):
                            data = message.data
                            if message.subtype == "model_refusal_fallback" and data.get(
                                "retracted_message_uuids"
                            ):
                                yield Error(
                                    message="Claude retracted earlier messages after a refusal; "
                                    "the v1 event contract cannot retract emitted output",
                                    error_type="provider_protocol_error",
                                )
                                retracted = True
                                pending.flush()
                                continue
                            if message.subtype == "api_retry":
                                yield _api_retry_warning(message, include_raw=req.include_raw)
                            compacted = _compaction_event(message)
                            if compacted is not None:
                                yield compacted
                            model = data.get("model") or None
                            if data.get("session_id") and (
                                not seen_session or (model and model != session_model)
                            ):
                                seen_session = True
                                session_model = model or session_model
                                yield SessionInfo(id=data["session_id"], model=session_model)
                        elif isinstance(message, ResultMessage):
                            if not seen_session and message.session_id:
                                seen_session = True
                                yield SessionInfo(id=message.session_id)
                            if message.model_usage or message.usage:
                                usage = _usage_event(
                                    message.usage or {},
                                    message.total_cost_usd,
                                    requests=message.num_turns,
                                    model_usage=message.model_usage,
                                    include_raw=req.include_raw,
                                )
                                # Reasoning occurred even when no thinking block was shown.
                                if usage.usage.reasoning_output_tokens and not seen_thinking:
                                    seen_thinking = True
                                    yield Thinking(text="")
                                yield usage
                            if retracted:
                                return
                            if (
                                req.output_schema is not None
                                and message.structured_output is not None
                            ):
                                try:
                                    value = validate_output(
                                        req.output_schema, message.structured_output
                                    )
                                except AgentSdkWrapperError as exc:
                                    yield Error(
                                        message=str(exc), error_type="structured_output_failed"
                                    )
                                    return
                                yield StructuredOutput(value=value)
                            error = _result_error(message, assistant_error)
                            if error is not None:
                                yield error
                            elif not seen_text and message.result:
                                seen_text = True
                                yield Text(text=message.result)
                            # The first result ends the run; later frames are not part of it.
                            return
                        elif isinstance(message, RateLimitEvent):
                            yield _rate_limit_warning(message, include_raw=req.include_raw)
                        elif isinstance(message, StreamEvent):
                            raise AgentSdkWrapperError(
                                "Claude Agent SDK emitted a partial StreamEvent, but "
                                "agent-sdk-wrapper does not support Claude partial messages. "
                                "Do not enable extra_options['include_partial_messages']."
                            )
            except Exception:
                # Keep a completed answer even when the runtime then fails.
                text = pending.flush()
                if text is not None:
                    yield text
                raise
            text = pending.flush()
            if text is not None:
                yield text
            yield Error(
                message="Claude stream ended without a result",
                error_type="provider_protocol_error",
            )
        except CLINotFoundError as exc:
            raise ProviderNotAvailableError(str(exc), cause=exc) from exc
        except CLIConnectionError as exc:
            raise TransientError(
                f"connection to Claude Code runtime failed: {exc}", cause=exc
            ) from exc
        except ProcessError as exc:
            stderr = "\n".join(stderr_tail)
            msg = f"{exc}\n{stderr}" if stderr else str(exc)
            signum = _SIGNALS_BY_EXIT_CODE.get(exc.exit_code) if exc.exit_code else None
            if signum is not None:
                raise ProcessTerminatedError(signum, message=msg, cause=exc) from exc
            if classify(stderr) == TRANSIENT:
                raise TransientError(msg, cause=exc) from exc
            raise AgentSdkWrapperError(msg, cause=exc) from exc
        except CLIJSONDecodeError as exc:
            raise AgentSdkWrapperError(f"failed to decode CLI output: {exc}", cause=exc) from exc
        except ClaudeSDKError as exc:
            raise AgentSdkWrapperError(str(exc), cause=exc) from exc


def _stderr_callback(
    tail: collections.deque[str] | None, user: Callable[[str], None] | None
) -> Callable[[str], None]:
    """Keep a tail for error reports; without a user callback, still show CLI stderr."""

    def callback(line: str) -> None:
        if tail is not None:
            tail.append(line)
        if user is not None:
            user(line)
        else:
            print(line, file=sys.stderr)

    return callback


@functools.cache
def _native_option_names() -> frozenset[str]:
    return frozenset(field.name for field in dataclasses.fields(ClaudeAgentOptions))


def _message_text(message: AssistantMessage) -> str:
    return "".join(block.text for block in message.content if isinstance(block, TextBlock))


@dataclasses.dataclass
class _PendingText:
    """Text blocks of one assistant message, which the CLI emits one frame per block."""

    message_id: str | None = None
    parts: list[str] = dataclasses.field(default_factory=list)

    def continues(self, message: Any) -> bool:
        # Status frames can arrive between the frames of one message.
        if isinstance(message, (SystemMessage, RateLimitEvent)):
            return True
        return (
            isinstance(message, AssistantMessage)
            and message.message_id is not None
            and message.message_id == self.message_id
            and not message.parent_tool_use_id
            and message.error is None
            and bool(message.content)
            and isinstance(message.content[0], TextBlock)
        )

    def flush(self) -> Text | None:
        if not self.parts:
            return None
        text = Text(text="".join(self.parts))
        self.parts.clear()
        return text


def _assistant_events(
    message: AssistantMessage, tool_names: dict[str, str], include_raw: bool, pending: _PendingText
) -> list[AgentEvent]:
    """Map one assistant frame; text accumulates in ``pending`` until a non-text block."""

    events: list[AgentEvent] = []
    for block in message.content:
        if isinstance(block, TextBlock):
            pending.message_id = message.message_id
            pending.parts.append(block.text)
            continue
        text = pending.flush()
        if text is not None:
            events.append(text)
        if isinstance(block, ThinkingBlock):
            events.append(_thinking_event(block))
        elif isinstance(block, (ToolUseBlock, ServerToolUseBlock)):
            if block.id:
                tool_names[block.id] = block.name
            events.append(
                ToolCall(
                    id=block.id,
                    name=block.name,
                    input=block.input,
                    raw=_raw(block) if include_raw else None,
                )
            )
        elif isinstance(block, ServerToolResultBlock):
            events.append(
                ToolResult(
                    id=block.tool_use_id,
                    name=tool_names.get(block.tool_use_id),
                    output=_stringify(block.content),
                    raw=_raw(block) if include_raw else None,
                )
            )
    return events


def _thinking_event(block: ThinkingBlock) -> Thinking:
    """Map thinking text, or report the encrypted signature length for redacted blocks."""

    if block.thinking.strip():
        return Thinking(text=block.thinking)
    return Thinking(text=block.thinking, redacted_bytes=len(block.signature) or None)


def _compaction_event(message: SystemMessage) -> ContextCompacted | None:
    """Map a ``compact_boundary`` system message to a normalized event."""

    if message.subtype != "compact_boundary":
        return None
    metadata = message.data.get("compact_metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    return ContextCompacted(
        trigger=str(metadata.get("trigger", "unknown")),
        pre_tokens=metadata.get("pre_tokens"),
    )


def _result_error(
    message: ResultMessage, assistant_error: tuple[str | None, str] | None = None
) -> Error | None:
    """Classify a result from structured signals first, then its error text."""

    reason = message.terminal_reason
    error_code, error_text = assistant_error or (None, "")
    detail = _error_detail(message, error_text)
    if reason in ("aborted_streaming", "aborted_tools"):
        return Error(message=detail or reason, error_type="cancelled")
    if message.subtype == "error_max_turns" or reason == "max_turns":
        return Error(message=detail or "reached the configured max turns", error_type="max_turns")
    if message.stop_reason == "refusal":
        return Error(message=detail or "the model refused the request", error_type="refused")
    if reason in _TERMINAL_REASON_ERRORS:
        return Error(message=detail or reason, error_type=_TERMINAL_REASON_ERRORS[reason])
    if message.subtype in _SUBTYPE_ERRORS:
        return Error(message=detail or message.subtype, error_type=_SUBTYPE_ERRORS[message.subtype])
    failed = (
        message.is_error
        or message.subtype != "success"
        or reason in _FAILURE_TERMINAL_REASONS
    )
    if not failed:
        return None
    status = message.api_error_status
    error_type = _classify(detail, status=status, assistant_error=error_code)
    if error_type is None:
        # Without an HTTP status, an API-stage failure means no response arrived.
        dropped = status is None and reason in (None, "api_error", "completed")
        error_type = "transient_api_error" if dropped else "execution_error"
    return Error(
        message=detail or "run reported an error",
        error_type=error_type,
    )


def _classify(text: str, *, status: int | None, assistant_error: str | None) -> str | None:
    if assistant_error == "authentication_failed" and status == 403:
        return "permission_denied"
    if assistant_error in _ASSISTANT_ERRORS:
        return _ASSISTANT_ERRORS[assistant_error]
    error_type = classify(text, status)
    # An invalid_request assistant error yields only to a more specific type.
    generic = error_type is None or error_type.startswith("api_error_")
    if assistant_error == "invalid_request" and generic:
        return "invalid_request"
    return error_type


def _error_detail(message: ResultMessage, assistant_text: str = "") -> str:
    """Pick the most specific failure text an errored result carries."""

    if assistant_text:
        return assistant_text
    if message.errors:
        return "; ".join(str(e) for e in message.errors)
    if message.result:
        return str(message.result)
    if message.api_error_status is not None:
        return f"API error {message.api_error_status}"
    return ""


def _rate_limit_warning(message: RateLimitEvent, *, include_raw: bool) -> WarningEvent:
    info = message.rate_limit_info
    details = [f"Claude rate limit status: {info.status}"]
    if info.rate_limit_type:
        details.append(f"type={info.rate_limit_type}")
    if info.utilization is not None:
        details.append(f"utilization={info.utilization}")
    if info.resets_at is not None:
        details.append(f"resets_at={info.resets_at}")
    return WarningEvent(
        message=", ".join(details),
        raw=_raw(message) if include_raw else None,
    )


def _api_retry_warning(message: SystemMessage, *, include_raw: bool) -> WarningEvent:
    """Report a retry the Claude runtime makes on its own, before the wrapper sees an error."""

    data = message.data
    status = data.get("error_status")
    delay = data.get("retry_delay_ms")
    text = f"Claude API error {status}" if status else "Claude API request failed"
    if data.get("error"):
        text += f" ({data['error']})"
    text += f"; runtime retry {data.get('attempt', '?')}/{data.get('max_retries', '?')}"
    if isinstance(delay, (int, float)):
        text += f" in {delay / 1000:.1f}s"
    return WarningEvent(message=text, raw=_raw(message) if include_raw else None)


def _stringify(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", "") or _compact_json(item))
            else:
                parts.append(str(item))
        return "".join(parts)
    if content is None or isinstance(content, dict):
        return _compact_json(content)
    return str(content)


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _anthropic_mcp_servers(servers: list[McpServer]) -> dict[str, Any]:
    configs: dict[str, Any] = {}
    for server in servers:
        if isinstance(server, McpStdioServer):
            config: dict[str, Any] = {
                "type": "stdio",
                "command": server.command,
            }
            if server.args:
                config["args"] = list(server.args)
            env = stdio_server_env(server)
            if env:
                config["env"] = env
            configs[server.name] = config
        elif isinstance(server, McpHttpServer):
            config = {
                "type": "http",
                "url": server.url,
            }
            if server.headers:
                config["headers"] = dict(server.headers)
            configs[server.name] = config
    return configs


def _validate_anthropic_mcp_servers(servers: list[McpServer]) -> None:
    for server in servers:
        unsupported: list[str] = []
        if server.default_tools_approval_mode is not None:
            unsupported.append("default_tools_approval_mode")
        if server.tool_approval_modes:
            unsupported.append("tool_approval_modes")
        if server.required is not None:
            unsupported.append("required")
        if server.startup_timeout_sec is not None:
            unsupported.append("startup_timeout_sec")
        if server.tool_timeout_sec is not None:
            unsupported.append("tool_timeout_sec")
        if isinstance(server, McpStdioServer) and server.cwd is not None:
            unsupported.append("cwd")
        if isinstance(server, McpHttpServer):
            if server.env_http_headers:
                unsupported.append("env_http_headers")
            if server.bearer_token_env_var is not None:
                unsupported.append("bearer_token_env_var")
        if unsupported:
            raise ConfigError(
                f"Anthropic MCP server {server.name!r} does not support: "
                f"{', '.join(unsupported)}"
            )


def _anthropic_tool_names(servers: list[McpServer], *, enabled: bool) -> list[str]:
    out: list[str] = []
    for server in servers:
        tools = server.enabled_tools if enabled else server.disabled_tools
        for tool in tools or []:
            out.append(f"mcp__{server.name}__{tool}")
    return out


def _usage_event(
    usage: dict[str, Any],
    cost: float | None,
    *,
    requests: int = 0,
    model_usage: dict[str, Any] | None = None,
    include_raw: bool = False,
) -> Usage:
    """Include cache in input totals. Prefer ``model_usage`` for subagent coverage.

    Fall back to ``usage``; never add both. Requests use the main-loop turn count.
    """

    raw = {"usage": usage, "model_usage": model_usage} if model_usage else usage
    if model_usage:
        usage = {
            normalized: sum(int(model.get(native, 0) or 0) for model in model_usage.values())
            for normalized, native in (
                ("input_tokens", "inputTokens"),
                ("output_tokens", "outputTokens"),
                ("cache_read_input_tokens", "cacheReadInputTokens"),
                ("cache_creation_input_tokens", "cacheCreationInputTokens"),
                ("thinking_tokens", "thinkingTokens"),
            )
        }
    else:
        details = usage.get("output_tokens_details")
        details = details if isinstance(details, dict) else {}
        usage = {**usage, "thinking_tokens": details.get("thinking_tokens")}
    cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
    cache_write = int(usage.get("cache_creation_input_tokens", 0) or 0)
    inp = int(usage.get("input_tokens", 0) or 0) + cache_read + cache_write
    out = int(usage.get("output_tokens", 0) or 0)
    return Usage(
        usage=TokenUsage(
            requests=requests,
            input_tokens=inp,
            output_tokens=out,
            total_tokens=inp + out,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            reasoning_output_tokens=int(usage.get("thinking_tokens", 0) or 0),
        ),
        cost_usd=cost,
        raw=raw if include_raw else None,
    )
