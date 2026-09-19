"""Claude Agent SDK adapter. The SDK manages the runtime and API retries."""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import json
import platform
import re
import shutil
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from ..artifacts import ProviderEventLogger
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
from ..tools import to_anthropic_tools, validate_tool_names
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
_SYNTHETIC_MODEL = "<synthetic>"
_SUBAGENT_TASK_TYPES = frozenset({"local_agent", "remote_agent"})
_TERMINAL_TASK_STATUSES = frozenset({"completed", "failed", "stopped", "killed"})
# First-class options compute these native keys; extra_options must not replace them.
_WRAPPER_OWNED_OPTIONS = frozenset({
    "agents",
    "allowed_tools",
    "cwd",
    "disallowed_tools",
    "effort",
    "env",
    "max_turns",
    "mcp_servers",
    "model",
    "output_format",
    "permission_mode",
    "resume",
    "setting_sources",
    "system_prompt",
})

# Retryable statuses reported in errored ResultMessage values.
_RETRYABLE_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504, 529})
_STATUS_ERRORS = {
    400: "invalid_request",
    401: "authentication_failed",
    402: "billing_error",
    403: "permission_denied",
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
    "blocking_limit": "usage_limit_exceeded",
    "rapid_refill_breaker": "usage_limit_exceeded",
}
# AssistantMessage.error values; "unknown" and "invalid_request" defer to other signals.
_ASSISTANT_ERRORS = {
    "authentication_failed": "authentication_failed",
    "oauth_org_not_allowed": "authentication_failed",
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
_TEXT_ERRORS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\b(?:not logged in|invalid (?:x-)?api[ _-]?key|authentication_error|unauthorized)\b",
            re.IGNORECASE,
        ),
        "authentication_failed",
    ),
    (re.compile(r"\bcredit balance\b|\bbilling\b", re.IGNORECASE), "billing_error"),
    (
        re.compile(r"\bprompt is too long\b|\bcontext window\b", re.IGNORECASE),
        "context_window_exceeded",
    ),
    (
        re.compile(r"\bmodel\b.{0,80}\b(?:not found|does not exist)\b", re.IGNORECASE),
        "model_not_found",
    ),
)
_TRANSIENT_TEXT = re.compile(
    r"\b(?:rate[ _-]?limit(?:ed)?|overloaded(?:_error)?|temporarily unavailable|server busy"
    r"|at capacity|connection (?:error|reset|refused)|timed out|ECONNRESET|ECONNREFUSED"
    r"|ETIMEDOUT|stream disconnected|API Error: (?:429|5\d\d))\b",
    re.IGNORECASE,
)

# Signal exits use 128 + signum in the SDK protocol, or -signum in asyncio.
_SIGNALS_BY_EXIT_CODE: dict[int, int] = {137: 9, 143: 15, 130: 2, -9: 9, -15: 15, -2: 2}


def _raw(obj: Any) -> dict[str, Any] | None:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        try:
            return dataclasses.asdict(obj)
        except Exception:
            return None
    return None


def _bundled_cli_path(claude_agent_sdk: Any) -> Path:
    cli_name = "claude.exe" if platform.system() == "Windows" else "claude"
    return Path(claude_agent_sdk.__file__).parent / "_bundled" / cli_name


class AnthropicProvider(ProviderAdapter):
    name = "anthropic"

    def __init__(self, *, cli_path: str | None = None) -> None:
        self._cli_path = cli_path

    def ensure_available(self) -> None:
        try:
            import claude_agent_sdk
        except ImportError as exc:
            raise ProviderNotAvailableError(
                "claude-agent-sdk is not installed. Install agent-sdk-wrapper with the "
                "Anthropic dependencies enabled."
            ) from exc
        if self._cli_path is not None:
            return
        if _bundled_cli_path(claude_agent_sdk).exists() or shutil.which("claude"):
            return
        raise ProviderNotAvailableError(
            "Claude Code runtime was not found. The Claude Agent SDK bundles it "
            "on supported wheels, or install Claude Code on PATH / pass "
            "provider_options={'cli_path': ...}."
        )

    def validate_request(self, req: RunRequest) -> None:
        effort = normalize_effort_for_provider("anthropic", req.effort)
        validate_tool_names(req.tools)
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
        owned = sorted(_WRAPPER_OWNED_OPTIONS & req.extra_options.keys())
        if owned:
            raise ConfigError(
                f"extra_options {owned} duplicate first-class Agent options; pass those instead"
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
    ):
        from claude_agent_sdk import AgentDefinition, ClaudeAgentOptions

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
        if effort:
            env.setdefault(_EFFORT_ENV, effort)
        env.setdefault(_BACKGROUND_TASKS_ENV, "1")

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
        try:
            return ClaudeAgentOptions(**kwargs)
        except TypeError as exc:
            msg = f"invalid Claude Agent SDK option: {exc}"
            raise AgentSdkWrapperError(msg, cause=exc) from exc

    async def stream(self, req: RunRequest) -> AsyncIterator[AgentEvent]:
        self.validate_request(req)
        self.ensure_available()

        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeSDKError,
            CLIConnectionError,
            CLIJSONDecodeError,
            CLINotFoundError,
            ProcessError,
            RateLimitEvent,
            ResultMessage,
            StreamEvent,
            SystemMessage,
            TaskNotificationMessage,
            TaskStartedMessage,
            TaskUpdatedMessage,
            ToolResultBlock,
            UserMessage,
            query,
        )

        stderr_tail: collections.deque[str] = collections.deque(maxlen=_STDERR_TAIL_LINES)
        options = self._build_options(req, stderr_tail)
        seen_session = False
        seen_text = False
        seen_thinking = False
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
            attempt=req.attempt,
        )

        try:
            async with contextlib.aclosing(
                query(prompt=req.prompt, options=options)
            ) as messages:
                async for message in messages:
                    provider_log.write(message)
                    if isinstance(message, AssistantMessage):
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
                        for event in _assistant_events(message, tool_names, req.include_raw):
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
                        data = message.data if isinstance(message.data, dict) else {}
                        subagent_type = data.get("subagent_type")
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
                        status = message.status or message.patch.get("status")
                        if (
                            status in _TERMINAL_TASK_STATUSES
                            and message.task_id in subagent_tasks
                        ):
                            subagent_tasks.discard(message.task_id)
                            yield SubagentEnded(task_id=message.task_id, status=status)
                    elif isinstance(message, SystemMessage):
                        data = message.data if isinstance(message.data, dict) else {}
                        if message.subtype == "model_refusal_fallback" and data.get(
                            "retracted_message_uuids"
                        ):
                            yield Error(
                                message="Claude retracted earlier messages after a refusal; "
                                "the v1 event contract cannot retract emitted output",
                                error_type="provider_protocol_error",
                            )
                            return
                        compacted = _compaction_event(message)
                        if compacted is not None:
                            yield compacted
                        if not seen_session and data.get("session_id"):
                            seen_session = True
                            yield SessionInfo(
                                id=data["session_id"], model=data.get("model") or None
                            )
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
                                continue
                            yield StructuredOutput(value=value)
                        error = _result_error(message, assistant_error)
                        if error is not None:
                            yield error
                        elif not seen_text and message.result:
                            seen_text = True
                            yield Text(text=message.result)
                    elif isinstance(message, RateLimitEvent):
                        yield _rate_limit_warning(message, include_raw=req.include_raw)
                    elif isinstance(message, StreamEvent):
                        raise AgentSdkWrapperError(
                            "Claude Agent SDK emitted a partial StreamEvent, but "
                            "agent-sdk-wrapper does not support Claude partial messages. "
                            "Do not enable extra_options['include_partial_messages']."
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
            if _looks_transient(stderr):
                raise TransientError(msg, cause=exc) from exc
            raise AgentSdkWrapperError(msg, cause=exc) from exc
        except CLIJSONDecodeError as exc:
            raise AgentSdkWrapperError(f"failed to decode CLI output: {exc}", cause=exc) from exc
        except ClaudeSDKError as exc:
            raise AgentSdkWrapperError(str(exc), cause=exc) from exc


def _stderr_callback(
    tail: collections.deque[str] | None, user: Callable[[str], None] | None
) -> Callable[[str], None]:
    def callback(line: str) -> None:
        if tail is not None:
            tail.append(line)
        if user is not None:
            user(line)

    return callback


def _message_text(message: Any) -> str:
    from claude_agent_sdk import TextBlock

    return "".join(block.text for block in message.content if isinstance(block, TextBlock))


def _assistant_events(
    message: Any, tool_names: dict[str, str], include_raw: bool
) -> list[AgentEvent]:
    """Map one assistant message; contiguous text blocks form one ``Text``."""

    from claude_agent_sdk import (
        ServerToolResultBlock,
        ServerToolUseBlock,
        TextBlock,
        ThinkingBlock,
        ToolUseBlock,
    )

    events: list[AgentEvent] = []
    text_parts: list[str] = []

    def flush_text() -> None:
        if text_parts:
            events.append(Text(text="".join(text_parts)))
            text_parts.clear()

    for block in message.content:
        if isinstance(block, TextBlock):
            text_parts.append(block.text)
            continue
        flush_text()
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
    flush_text()
    return events


def _thinking_event(block: Any) -> Thinking:
    """Map thinking text, or report the encrypted signature length for redacted blocks."""

    text = block.thinking or ""
    signature = getattr(block, "signature", None) or ""
    if text.strip():
        return Thinking(text=text)
    return Thinking(text=text, redacted_bytes=len(signature) or None)


def _compaction_event(message: Any) -> ContextCompacted | None:
    """Map a ``compact_boundary`` system message to a normalized event."""

    if getattr(message, "subtype", None) != "compact_boundary":
        return None
    data = message.data if isinstance(message.data, dict) else {}
    metadata = data.get("compact_metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    return ContextCompacted(
        trigger=str(metadata.get("trigger", "unknown")),
        pre_tokens=metadata.get("pre_tokens"),
    )


def _result_error(
    message: Any, assistant_error: tuple[str | None, str] | None = None
) -> Error | None:
    """Classify a result from structured signals first, then its error text."""

    reason = message.terminal_reason
    error_code, error_text = assistant_error or (None, "")
    detail = _error_detail(message, error_text)
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
        or reason not in (None, "completed")
    )
    if not failed:
        return None
    error_type = _classify(detail, status=message.api_error_status, assistant_error=error_code)
    return Error(
        message=detail or "run reported an error",
        error_type=error_type or "execution_error",
        retryable=error_type == "transient_api_error",
    )


def _classify(text: str, *, status: int | None, assistant_error: str | None) -> str | None:
    if assistant_error in _ASSISTANT_ERRORS:
        return _ASSISTANT_ERRORS[assistant_error]
    for pattern, error_type in _TEXT_ERRORS:
        if pattern.search(text):
            return error_type
    if status is not None and status in _RETRYABLE_STATUS_CODES:
        return "transient_api_error"
    if status is not None and status in _STATUS_ERRORS:
        return _STATUS_ERRORS[status]
    if _looks_transient(text):
        return "transient_api_error"
    if assistant_error == "invalid_request":
        return "invalid_request"
    if status is not None:
        return f"api_error_{status}"
    return None


def _error_detail(message: Any, assistant_text: str = "") -> str:
    """Pick the most specific failure text an errored result carries."""

    if assistant_text:
        return assistant_text
    errors = getattr(message, "errors", None)
    if errors:
        return "; ".join(str(e) for e in errors)
    if message.result:
        return str(message.result)
    if message.api_error_status is not None:
        return f"API error {message.api_error_status}"
    return ""


def _rate_limit_warning(message: Any, *, include_raw: bool) -> WarningEvent:
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


def _looks_transient(text: str) -> bool:
    return bool(_TRANSIENT_TEXT.search(text))


def _stringify(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", "") or json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "".join(parts)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


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
