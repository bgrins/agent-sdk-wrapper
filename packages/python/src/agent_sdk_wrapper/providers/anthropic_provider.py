"""Anthropic adapter, backed by the Claude Agent SDK (``claude_agent_sdk``).

The SDK launches a Claude Code runtime internally, preferring its bundled binary
when available and falling back to ``PATH``. It retries transient API errors
internally; we map its process-level exceptions onto the unified error
hierarchy.
"""

from __future__ import annotations

import dataclasses
import platform
import shutil
from collections.abc import AsyncIterator
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
from ..tools import to_anthropic_tools
from .base import ProviderAdapter

_DEFAULT_THINKING: dict[str, str] = {"type": "adaptive", "display": "summarized"}
_WEB_TOOL_NAMES: tuple[str, ...] = ("WebSearch", "WebFetch")

# HTTP statuses worth another attempt. A mid-run API call that returned one of
# these — or that got no response at all — is the common transient failure; the
# SDK reports it as an errored ``ResultMessage`` rather than raising.
_RETRYABLE_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504, 529})

# Exit codes the runtime reports when an external signal killed it. The control
# protocol uses the shell convention (128 + signum); the subprocess transport
# reports asyncio's negative return code.
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
        normalize_effort_for_provider("anthropic", req.effort)
        active_mcp_servers = [
            server for server in req.mcp_servers if server.enabled is not False
        ]
        _validate_anthropic_mcp_servers(active_mcp_servers)
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

    def _build_options(self, req: RunRequest):
        from claude_agent_sdk import AgentDefinition, ClaudeAgentOptions

        allowed = list(req.allowed_tools)
        disallowed = list(req.disallowed_tools)

        if req.web_tools is False:
            for name in _WEB_TOOL_NAMES:
                if name not in disallowed:
                    disallowed.append(name)

        active_mcp_servers = [
            server for server in req.mcp_servers if server.enabled is not False
        ]
        self.validate_request(req)

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
            if "Agent" not in allowed:
                allowed.append("Agent")  # the subagent-delegation tool

        kwargs: dict[str, Any] = {
            "model": req.model,
            "system_prompt": req.system_prompt,
            "max_turns": req.max_turns,
            "effort": normalize_effort_for_provider("anthropic", req.effort),
            "cwd": req.cwd,
            "env": req.env,
            "allowed_tools": allowed,
            "disallowed_tools": disallowed,
            "permission_mode": req.permission_mode,
            "mcp_servers": mcp_servers,
        }
        if req.session_id:
            kwargs["resume"] = req.session_id
        if req.setting_sources is not None:
            kwargs["setting_sources"] = req.setting_sources
        if self._cli_path is not None:
            kwargs["cli_path"] = self._cli_path
        if agents:
            kwargs["agents"] = agents
        if req.output_schema is not None:
            kwargs["output_format"] = {
                "type": "json_schema",
                "schema": json_schema_of_type(req.output_schema),
            }
        if req.builtin_tools is not None:
            kwargs["tools"] = [] if req.builtin_tools == "none" else list(req.builtin_tools)
        if "thinking" not in req.extra_options:
            kwargs["thinking"] = dict(_DEFAULT_THINKING)
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        kwargs.update(req.extra_options)
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
            ServerToolResultBlock,
            ServerToolUseBlock,
            StreamEvent,
            SystemMessage,
            TaskNotificationMessage,
            TaskStartedMessage,
            TextBlock,
            ThinkingBlock,
            ToolResultBlock,
            ToolUseBlock,
            UserMessage,
            query,
        )

        options = self._build_options(req)
        seen_session = False
        # tool_use_id -> tool name, so a tool result can name the tool it
        # answers without the caller having to correlate ids itself.
        tool_names: dict[str, str] = {}
        provider_log = ProviderEventLogger(
            "anthropic", req.artifacts_dir, req.on_provider_event
        )

        try:
            async for message in query(prompt=req.prompt, options=options):
                provider_log.write(message)
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            yield Text(text=block.text)
                        elif isinstance(block, ThinkingBlock):
                            yield _thinking_event(block)
                        elif isinstance(block, (ToolUseBlock, ServerToolUseBlock)):
                            if block.id:
                                tool_names[block.id] = block.name
                            yield ToolCall(
                                id=block.id,
                                name=block.name,
                                input=block.input,
                                raw=_raw(block) if req.include_raw else None,
                            )
                elif isinstance(message, UserMessage):
                    content = message.content
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, ToolResultBlock):
                                yield ToolResult(
                                    id=block.tool_use_id,
                                    name=tool_names.get(block.tool_use_id),
                                    output=_stringify(block.content),
                                    is_error=bool(block.is_error),
                                    raw=_raw(block) if req.include_raw else None,
                                )
                            elif isinstance(block, ServerToolResultBlock):
                                yield ToolResult(
                                    id=block.tool_use_id,
                                    name=tool_names.get(block.tool_use_id),
                                    output=_stringify(block.content),
                                    is_error=False,
                                    raw=_raw(block) if req.include_raw else None,
                                )
                elif isinstance(message, TaskStartedMessage):
                    yield SubagentStarted(
                        task_id=message.task_id,
                        name=message.task_type or "",
                        description=message.description,
                    )
                elif isinstance(message, TaskNotificationMessage):
                    yield SubagentEnded(
                        task_id=message.task_id,
                        status=message.status,
                        summary=message.summary,
                    )
                elif isinstance(message, SystemMessage):
                    compacted = _compaction_event(message)
                    if compacted is not None:
                        yield compacted
                    if not seen_session and isinstance(message.data, dict):
                        sid = message.data.get("session_id")
                        if sid:
                            seen_session = True
                            yield SessionInfo(id=sid)
                elif isinstance(message, ResultMessage):
                    if not seen_session and message.session_id:
                        seen_session = True
                        yield SessionInfo(id=message.session_id)
                    if message.model_usage or message.usage:
                        yield _usage_event(
                            message.usage or {},
                            message.total_cost_usd,
                            requests=message.num_turns,
                            model_usage=message.model_usage,
                        )
                    if req.output_schema is not None and message.structured_output is not None:
                        yield StructuredOutput(
                            value=validate_output(req.output_schema, message.structured_output)
                        )
                    error = _result_error(message)
                    if error is not None:
                        yield error
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
            msg = f"{exc}"
            signum = _SIGNALS_BY_EXIT_CODE.get(exc.exit_code) if exc.exit_code else None
            if signum is not None:
                raise ProcessTerminatedError(signum, message=msg, cause=exc) from exc
            if _looks_transient(getattr(exc, "stderr", "") or msg):
                raise TransientError(msg, cause=exc) from exc
            raise AgentSdkWrapperError(msg, cause=exc) from exc
        except CLIJSONDecodeError as exc:
            raise AgentSdkWrapperError(f"failed to decode CLI output: {exc}", cause=exc) from exc
        except ClaudeSDKError as exc:
            raise AgentSdkWrapperError(str(exc), cause=exc) from exc


def _thinking_event(block: Any) -> Thinking:
    """Map a thinking block, reporting redacted length when the text is hidden.

    Some thinking display modes return an encrypted signature the provider
    round-trips but never exposes. Reporting its size keeps a redacted item
    distinguishable from an empty one.
    """

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


def _result_error(message: Any) -> Error | None:
    """Classify a terminal ``ResultMessage`` into a normalized error, if any.

    The SDK reports most failures as an errored result rather than an
    exception, so the terminal reason is only recoverable from ``subtype``,
    ``stop_reason``, and ``api_error_status``. Mapping them here is what lets
    the run report ``max_turns`` or ``refused`` instead of a generic error, and
    lets callers tell a retryable upstream blip from a permanent failure.
    """

    if message.subtype == "error_max_turns":
        return Error(
            message=_error_detail(message) or "reached the configured max turns",
            error_type="max_turns",
        )
    if message.stop_reason == "refusal":
        return Error(
            message=_error_detail(message) or "the model refused the request",
            error_type="refused",
        )
    if not message.is_error:
        return None
    if _is_transient_result(message):
        status = message.api_error_status
        detail = f"API error {status}" if status is not None else "connection dropped"
        return Error(
            message=f"transient upstream failure: {detail}",
            error_type="transient_api_error",
            retryable=True,
        )
    # Not stop_reason: that describes how generation ended (``end_turn``,
    # ``stop_sequence``) and is unrelated to why the run failed, so using it
    # here labels an HTTP 400 as "stop_sequence". Prefer the failing status,
    # then a genuine error subtype.
    if message.api_error_status is not None:
        error_type = f"api_error_{message.api_error_status}"
    elif message.subtype and message.subtype.startswith("error"):
        error_type = message.subtype
    else:
        error_type = "result_error"
    return Error(
        message=_error_detail(message) or "run reported an error",
        error_type=error_type,
    )


def _error_detail(message: Any) -> str:
    """Pick the most specific failure text an errored result carries."""

    errors = getattr(message, "errors", None)
    if errors:
        return "; ".join(str(e) for e in errors)
    if message.api_error_status is not None:
        return f"API error {message.api_error_status}"
    if message.result:
        return str(message.result)
    return str(message.subtype or "")


def _is_transient_result(message: Any) -> bool:
    """Whether an errored result is a retryable upstream failure.

    A mid-run API call that failed leaves ``subtype`` at ``success`` — the run
    itself did not fail structurally. It is worth retrying when the call
    returned a retryable status, or got no response at all (a dropped
    connection). Client errors and structural run failures are not.
    """

    if message.subtype != "success":
        return False
    status = message.api_error_status
    return status is None or status in _RETRYABLE_STATUS_CODES


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
    low = text.lower()
    return any(s in low for s in ("rate limit", "overloaded", "429", "503", "timeout", "timed out"))


def _stringify(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", "") or str(item))
            else:
                parts.append(str(item))
        return "".join(parts)
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
) -> Usage:
    """Normalize an SDK usage mapping onto :class:`TokenUsage`.

    Anthropic reports ``input_tokens`` net of cache, with the cached portion in
    its own counters, so the cache counts are folded back in to match the
    convention the wrapper publishes (and what the Codex adapter already
    emits). Prefer model_usage, which includes subagents and auxiliary calls;
    the legacy usage field covers only the main loop. Never add both together.
    The result's main-loop turn count remains a proxy for requests, not a count
    of all requests made by subagents.
    """

    raw = usage
    if model_usage:
        raw = {"usage": usage, "model_usage": model_usage}
        usage = {
            normalized: sum(int(model.get(native, 0) or 0) for model in model_usage.values())
            for normalized, native in (
                ("input_tokens", "inputTokens"),
                ("output_tokens", "outputTokens"),
                ("cache_read_input_tokens", "cacheReadInputTokens"),
                ("cache_creation_input_tokens", "cacheCreationInputTokens"),
            )
        }
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
        ),
        cost_usd=cost,
        raw=raw,
    )
