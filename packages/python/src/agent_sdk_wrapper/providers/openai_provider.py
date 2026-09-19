"""Codex SDK adapter. Each wrapper run maps to one Codex turn."""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import json
import os
import re
import shutil
import signal
import sys
import tempfile
import textwrap
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

from ..artifacts import ProviderEventLogger, sdk_dir_for
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
from ..tools import CODEX_TOOL_SERVER, tool_description, tool_name, validate_tool_names
from .base import ProviderAdapter

_THREAD_RESUME_OPTION_KEYS = {
    "approval_mode",
    "base_instructions",
    "config",
    "cwd",
    "developer_instructions",
    "model",
    "model_provider",
    "personality",
    "sandbox",
    "service_tier",
}
_CODEX_NATIVE_TOOL_FILTER_NAMES = {
    "agent",
    "command",
    "file_change",
    "image_generation",
    "view_image",
    "web_search",
}
_CONFIG_KEY_PART_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_DEFAULT_REASONING_SUMMARY = "auto"


class OpenAIProvider(ProviderAdapter):
    name = "openai"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        config: Any = None,
        codex: Any = None,
        thread_id: str | None = None,
        approval_mode: Any = None,
        sandbox: Any = None,
        model_provider: str | None = None,
        effort: Any = None,
        summary: Any = None,
        personality: Any = None,
        service_tier: str | None = None,
        ephemeral: bool | None = None,
        debug: bool = False,
        thread_options: dict[str, Any] | None = None,
        turn_options: dict[str, Any] | None = None,
    ) -> None:
        self._api_key = api_key
        self._config = config
        self._codex = codex
        self._thread_id = thread_id
        self._approval_mode = approval_mode
        self._sandbox = sandbox
        self._model_provider = model_provider
        self._effort = (
            normalize_effort_for_provider("openai", effort)
            if isinstance(effort, str) or effort is None
            else effort
        )
        self._summary = _DEFAULT_REASONING_SUMMARY if summary is None else summary
        self._personality = personality
        self._service_tier = service_tier
        self._ephemeral = ephemeral
        self._debug = debug
        self._thread_options = dict(thread_options or {})
        self._turn_options = dict(turn_options or {})

    def ensure_available(self) -> None:
        try:
            import openai_codex  # noqa: F401
        except ImportError as exc:
            raise ProviderNotAvailableError(
                "the 'openai_codex' package is not installed. Install the "
                "OpenAI Codex Python SDK with 'pip install openai-codex'."
            ) from exc
        if _config_has_codex_bin(self._config):
            return
        if _codex_cli_bin_available() or shutil.which("codex"):
            return
        raise ProviderNotAvailableError(
            "Codex runtime was not found. Install a compatible openai-codex-cli-bin "
            "wheel, install the Codex CLI on PATH, or pass "
            "provider_options={'config': {'codex_bin': '...'}}."
        )

    def validate_request(self, req: RunRequest) -> None:
        _validate_supported(req)
        if self._api_key and not self._launches_codex():
            raise ConfigError(
                "api_key requires the provider to launch Codex; authenticate the "
                "pre-built codex client or custom launch command instead"
            )

    def _launches_codex(self) -> bool:
        return self._codex is None and _config_value(self._config, "launch_args_override") is None

    def _login_api_key(self) -> str | None:
        # A pre-built client or custom launch command cannot take the ephemeral
        # credential store override, so logging in would overwrite auth.json.
        if not self._launches_codex():
            return None
        return self._api_key or os.environ.get("OPENAI_API_KEY") or None

    async def stream(self, req: RunRequest) -> AsyncIterator[AgentEvent]:
        self.validate_request(req)
        self.ensure_available()

        from openai_codex import is_retryable_error

        codex: Any = None
        try:
            api_key = self._login_api_key()
            with _runtime_config(req) as runtime_config:
                if self._codex is not None and runtime_config.config_overrides:
                    raise ConfigError(
                        "Codex tools, subagents, and web_tools require the provider "
                        "to launch Codex; a pre-built codex client cannot be reconfigured"
                    )
                config_overrides = runtime_config.config_overrides
                if api_key:
                    # Keep the API key in memory instead of replacing auth.json.
                    config_overrides += (
                        _config_override("cli_auth_credentials_store", value="ephemeral"),
                    )
                async with self._codex_client(req, config_overrides) as codex:
                    process = _codex_process(codex)
                    try:
                        async for event in self._run(codex, req, runtime_config, api_key):
                            yield event
                    except Exception as exc:
                        await _raise_if_signaled(process, exc)
                        raise
        except (ProviderNotAvailableError, ConfigError, AgentSdkWrapperError):
            raise
        except FileNotFoundError as exc:
            raise ProviderNotAvailableError(str(exc), cause=exc) from exc
        except Exception as exc:
            if is_retryable_error(exc) or _looks_transient(exc):
                raise TransientError(str(exc), cause=exc) from exc
            raise AgentSdkWrapperError(f"{type(exc).__name__}: {exc}", cause=exc) from exc
        finally:
            _write_sdk_debug_log(codex, req.artifacts_dir, debug=self._debug)

    async def _run(
        self,
        codex: Any,
        req: RunRequest,
        runtime_config: _RuntimeConfig,
        api_key: str | None,
    ) -> AsyncIterator[AgentEvent]:
        from openai_codex import ApprovalMode, Sandbox

        if api_key:
            await codex.login_api_key(api_key)

        approval_mode = _enum_value(ApprovalMode, self._approval_mode)
        sandbox = _enum_value(Sandbox, self._sandbox)
        thread_kwargs, turn_kwargs = self._build_options(req, approval_mode, sandbox)

        thread_id = req.session_id or self._thread_id
        if thread_id:
            _validate_thread_resume_options(thread_kwargs)
            thread = await codex.thread_resume(thread_id, **thread_kwargs)
        else:
            thread = await codex.thread_start(**thread_kwargs)
        yield SessionInfo(id=thread.id)

        for warning in runtime_config.warnings:
            yield WarningEvent(message=warning)

        turn = await thread.turn(req.prompt, **turn_kwargs)
        async for event in _stream_turn(turn, req, thread.id):
            yield event

    @asynccontextmanager
    async def _codex_client(self, req: RunRequest, config_overrides: tuple[str, ...] = ()):
        if self._codex is not None:
            yield self._codex
            return

        from openai_codex import AsyncCodex

        config = _codex_config(
            self._config,
            req.env,
            req.artifacts_dir,
            debug=self._debug,
            config_overrides=config_overrides,
        )
        async with AsyncCodex(config=config) as codex:
            yield codex

    def _build_options(
        self, req: RunRequest, approval_mode: Any, sandbox: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        req_effort = normalize_effort_for_provider("openai", req.effort)
        extra = dict(req.extra_options)
        thread_options = dict(self._thread_options)
        turn_options = dict(self._turn_options)
        thread_options.update(extra.pop("thread_options", {}))
        turn_options.update(extra.pop("turn_options", {}))
        if extra:
            keys = ", ".join(sorted(extra))
            raise ConfigError(
                "unsupported Codex SDK extra_options keys: "
                f"{keys}. Use 'thread_options' or 'turn_options'."
            )

        thread_options.setdefault("model", req.model)
        thread_options.setdefault("model_provider", self._model_provider)
        thread_options.setdefault("cwd", _as_str(req.cwd))
        thread_options.setdefault("developer_instructions", req.system_prompt)
        thread_options.setdefault("approval_mode", approval_mode)
        thread_options.setdefault("sandbox", sandbox)
        thread_options.setdefault("personality", self._personality)
        thread_options.setdefault("service_tier", self._service_tier)
        thread_options.setdefault("ephemeral", self._ephemeral)

        turn_options.setdefault("model", req.model)
        turn_options.setdefault("cwd", _as_str(req.cwd))
        turn_options.setdefault("approval_mode", approval_mode)
        turn_options.setdefault("sandbox", sandbox)
        turn_options.setdefault("effort", req_effort or self._effort)
        turn_options.setdefault("summary", self._summary)
        turn_options.setdefault("personality", self._personality)
        turn_options.setdefault("service_tier", self._service_tier)
        if req.output_schema is not None:
            turn_options.setdefault("output_schema", _codex_output_schema(req.output_schema))

        return _drop_none(thread_options), _drop_none(turn_options)


async def _stream_turn(
    turn: Any,
    req: RunRequest,
    thread_id: str | None = None,
) -> AsyncIterator[AgentEvent]:
    """Normalize one Codex turn, ending with its terminal state."""

    provider_log = ProviderEventLogger(
        "openai",
        req.artifacts_dir,
        req.on_provider_event,
        run_id=req.run_id,
        attempt=req.attempt,
    )
    text_delta_parts: dict[str | None, list[str]] = {}
    thinking_delta_parts: dict[str | None, list[str]] = {}
    texts: list[str] = []
    usage = _TurnUsage()
    completed_action_items = 0
    interrupted_for_max_turns = False
    # A non-retried error notification precedes the failed turn/completed; emit one Error.
    reported_error: Error | None = None

    async for event in turn.stream():
        provider_log.write(event)
        method = getattr(event, "method", "")
        payload = getattr(event, "payload", None)
        if method == "item/agentMessage/delta":
            delta = getattr(payload, "delta", "") or ""
            if delta:
                item_id = _codex_item_id(payload)
                text_delta_parts.setdefault(item_id, []).append(delta)
            continue

        if method in {
            "item/reasoning/textDelta",
            "item/reasoning/summaryTextDelta",
        }:
            delta = getattr(payload, "delta", "") or ""
            if delta:
                item_id = _codex_item_id(payload)
                thinking_delta_parts.setdefault(item_id, []).append(delta)
            continue

        if method == "item/completed":
            item = getattr(payload, "item", None)
            root = getattr(item, "root", item)
            root_type = getattr(root, "type", "")
            if root_type == "agentMessage":
                item_id = _codex_item_id(root)
                buffered_text = _pop_delta_buffer(text_delta_parts, item_id)
                text = getattr(root, "text", "") or buffered_text
                if text:
                    texts.append(text)
                    yield Text(text=text, raw=_raw(event) if req.include_raw else None)
                continue
            if root_type == "reasoning":
                item_id = _codex_item_id(root)
                buffered_text = _pop_delta_buffer(thinking_delta_parts, item_id)
                text = _reasoning_text(root) or buffered_text
                # Preserve empty reasoning items: they can carry billed tokens.
                yield Thinking(text=text, raw=_raw(event) if req.include_raw else None)
                continue
            if root_type == "plan":
                text = getattr(root, "text", "") or ""
                if text:
                    yield Thinking(text=text, raw=_raw(event) if req.include_raw else None)
                continue
            if root_type == "contextCompaction":
                yield ContextCompacted(
                    trigger="codex",
                    raw=_raw(event) if req.include_raw else None,
                )
                continue
            tool_events = _tool_events(root, event, req.include_raw)
            for tool_event in tool_events:
                yield tool_event
            if _counts_toward_max_turns(root_type):
                completed_action_items += 1
                if (
                    req.max_turns is not None
                    and completed_action_items >= req.max_turns
                    and not interrupted_for_max_turns
                ):
                    interrupted_for_max_turns = True
                    # Keep draining so buffered text and usage still arrive.
                    await _interrupt_for_max_turns(turn, req.max_turns)
            continue

        if method == "thread/tokenUsage/updated":
            usage.add(getattr(payload, "token_usage", None) or getattr(payload, "tokenUsage", None))
            continue

        if method == "error":
            error = getattr(payload, "error", None)
            # Keep SDK retries as warnings; the terminal state determines success.
            if getattr(payload, "will_retry", False):
                yield WarningEvent(message=_error_message(error) or "Codex reported an error")
            else:
                reported_error = _error_event(error, _raw(event) if req.include_raw else None)
            continue

        if method == "turn/completed":
            for text in _drain_delta_buffers(text_delta_parts):
                texts.append(text)
                yield Text(text=text)
            for text in _drain_delta_buffers(thinking_delta_parts):
                yield Thinking(text=text)
            usage_event = usage.event(req.include_raw)
            if usage_event is not None:
                yield usage_event
            turn_info = getattr(payload, "turn", None)
            status = _status_value(getattr(turn_info, "status", None))
            if status == "failed":
                error = getattr(turn_info, "error", None)
                if error is not None:
                    yield _error_event(error, _raw(event) if req.include_raw else None)
                else:
                    yield reported_error or Error(
                        message="Codex turn failed", error_type="provider_exception"
                    )
                return
            if status == "interrupted":
                if interrupted_for_max_turns:
                    yield Error(
                        message=(
                            f"Codex max_turns={req.max_turns} reached after "
                            f"{completed_action_items} completed action item(s); "
                            "interrupted turn"
                        ),
                        error_type="max_turns",
                    )
                else:
                    yield Error(message="Codex turn was interrupted", error_type="cancelled")
                return
            if reported_error is not None:
                yield WarningEvent(message=reported_error.message)
            if req.output_schema is not None:
                yield _structured_output_event(req.output_schema, texts[-1] if texts else "")
            return

    yield reported_error or Error(
        message="Codex turn stream ended before turn/completed",
        error_type="provider_protocol_error",
    )


def _structured_output_event(output_schema: type, text: str) -> AgentEvent:
    if not text:
        return Error(
            message="Codex returned no structured output", error_type="structured_output_failed"
        )
    try:
        parsed = _structured_value(output_schema, _parse_json(text))
        return StructuredOutput(value=validate_output(output_schema, parsed))
    except AgentSdkWrapperError as exc:
        return Error(message=str(exc), error_type="structured_output_failed")


def _validate_supported(req: RunRequest) -> None:
    normalize_effort_for_provider("openai", req.effort)
    validate_tool_names(req.tools)
    if req.output_schema is not None:
        _codex_output_schema(req.output_schema)
    unsupported: list[str] = []
    if req.max_turns is not None and req.max_turns < 1:
        unsupported.append("max_turns < 1")
    if req.builtin_tools is not None:
        unsupported.append(
            "builtin_tools. Codex built-in tools cannot be disabled or allowlisted "
            "through agent-sdk-wrapper yet"
        )
    if req.permission_mode is not None:
        unsupported.append("permission_mode")
    if req.setting_sources is not None:
        unsupported.append("setting_sources")
    if (req.allowed_tools or req.disallowed_tools) and not (req.tools or req.mcp_servers):
        unsupported.append("allowed_tools/disallowed_tools without callable tools or MCP servers")
    if req.allowed_tools or req.disallowed_tools:
        unsupported_filters = _unsupported_tool_filters(req)
        if unsupported_filters.non_wrapper:
            unsupported.append(
                "Codex tool filters for non-wrapper tools: "
                + ", ".join(sorted(unsupported_filters.non_wrapper))
            )
        if unsupported_filters.native:
            unsupported.append(
                "Codex native tool filters: "
                + ", ".join(sorted(unsupported_filters.native))
                + ". Codex built-in tools are not controlled by agent-sdk-wrapper "
                "allowed_tools/disallowed_tools"
            )
    unsupported_subagents = _unsupported_subagent_controls(req.subagents)
    if unsupported_subagents:
        unsupported.extend(unsupported_subagents)
    if unsupported:
        raise ConfigError(
            "the OpenAI Codex SDK provider does not support: "
            f"{', '.join(unsupported)}"
        )


_CODEX_ACTION_ITEM_TYPES = {
    "collabAgentToolCall",
    "commandExecution",
    "dynamicToolCall",
    "fileChange",
    "imageGeneration",
    "imageView",
    "mcpToolCall",
    "webSearch",
}


_USAGE_FIELDS = {
    "input_tokens": ("inputTokens", "input_tokens"),
    "cache_read_tokens": ("cachedInputTokens", "cached_input_tokens"),
    "cache_write_tokens": ("cacheWriteInputTokens", "cache_write_input_tokens"),
    "output_tokens": ("outputTokens", "output_tokens"),
    "reasoning_output_tokens": ("reasoningOutputTokens", "reasoning_output_tokens"),
    "total_tokens": ("totalTokens", "total_tokens"),
}


@dataclasses.dataclass
class _TurnUsage:
    """Per-turn usage from the thread's cumulative ``total`` and per-request ``last``.

    The first update's ``total - last`` is the thread's usage before this turn,
    which covers resumed history without state kept across runs. Native output
    already includes reasoning and ``totalTokens`` is input plus output.
    """

    before: dict[str, int] | None = None
    total: dict[str, int] | None = None
    updates: int = 0
    raw: dict[str, Any] | None = None

    def add(self, token_usage: Any) -> None:
        data = _to_plain(token_usage)
        if not isinstance(data, dict):
            return
        total = _usage_breakdown(data.get("total", data))
        last = _usage_breakdown(data["last"]) if "last" in data else total
        if self.before is None:
            self.before = {key: total[key] - last[key] for key in total}
        self.total = total
        self.updates += 1
        self.raw = data

    def event(self, include_raw: bool) -> Usage | None:
        if self.total is None or self.before is None:
            return None
        delta = {key: max(0, self.total[key] - self.before[key]) for key in self.total}
        # Count usage updates as a proxy for model requests.
        return Usage(
            usage=TokenUsage(**delta, requests=self.updates),
            raw=self.raw if include_raw else None,
        )


def _usage_breakdown(data: Any) -> dict[str, int]:
    values = {key: _int_field(data, *aliases) for key, aliases in _USAGE_FIELDS.items()}
    if not values["total_tokens"]:
        values["total_tokens"] = values["input_tokens"] + values["output_tokens"]
    return values


def _counts_toward_max_turns(root_type: str) -> bool:
    return root_type in _CODEX_ACTION_ITEM_TYPES


def _codex_item_id(value: Any) -> str | None:
    item_id = getattr(value, "item_id", None)
    if item_id is None:
        item_id = getattr(value, "id", None)
    if item_id is None:
        item_id = getattr(value, "itemId", None)
    return item_id if isinstance(item_id, str) else None


def _pop_delta_buffer(
    buffers: dict[str | None, list[str]],
    item_id: str | None,
) -> str:
    if item_id in buffers:
        return "".join(buffers.pop(item_id))
    if item_id is not None and None in buffers:
        return "".join(buffers.pop(None))
    return ""


def _drain_delta_buffers(buffers: dict[str | None, list[str]]) -> list[str]:
    drained = ["".join(parts) for _, parts in sorted(buffers.items(), key=_buffer_sort_key)]
    buffers.clear()
    return [text for text in drained if text]


def _buffer_sort_key(item: tuple[str | None, list[str]]) -> str:
    key, _ = item
    return "" if key is None else key


async def _interrupt_for_max_turns(turn: Any, max_turns: int) -> None:
    from openai_codex.errors import InvalidRequestError

    interrupt = getattr(turn, "interrupt", None)
    if not callable(interrupt):
        raise AgentSdkWrapperError(
            f"Codex max_turns={max_turns} reached, but the SDK turn cannot be interrupted"
        )
    try:
        await interrupt()
    except InvalidRequestError:
        # The turn finished before the interrupt landed; its turn/completed still follows.
        pass


@dataclasses.dataclass(frozen=True)
class _UnsupportedToolFilters:
    non_wrapper: tuple[str, ...] = ()
    native: tuple[str, ...] = ()


def _unsupported_tool_filters(req: RunRequest) -> _UnsupportedToolFilters:
    if not (req.tools or req.mcp_servers):
        return _UnsupportedToolFilters()

    filters = (*req.allowed_tools, *req.disallowed_tools)
    managed_servers = {server.name for server in req.mcp_servers}
    callable_tools = {tool_name(fn) for fn in req.tools}
    known_tools_by_server = {
        server.name: set(server.enabled_tools or ())
        for server in req.mcp_servers
        if server.enabled_tools is not None
    }
    external_tools_fully_known = all(
        server.enabled_tools is not None for server in req.mcp_servers
    )
    if req.tools:
        managed_servers.add(CODEX_TOOL_SERVER)
        known_tools_by_server[CODEX_TOOL_SERVER] = callable_tools
    known_unqualified_tools = {
        name for tools in known_tools_by_server.values() for name in tools
    }

    non_wrapper: list[str] = []
    native: list[str] = []
    for spec in filters:
        server, tool = _split_tool_filter(spec)
        if server is not None and server not in managed_servers:
            non_wrapper.append(spec)
            continue
        if server is not None and tool not in known_tools_by_server.get(server, {tool}):
            non_wrapper.append(spec)
            continue
        if server is None and tool in _CODEX_NATIVE_TOOL_FILTER_NAMES:
            native.append(spec)
            continue
        if (
            server is None
            and external_tools_fully_known
            and tool not in known_unqualified_tools
        ):
            non_wrapper.append(spec)
    return _UnsupportedToolFilters(tuple(set(non_wrapper)), tuple(set(native)))


def _unsupported_subagent_controls(subagents: dict[str, Any]) -> list[str]:
    unsupported: list[str] = []
    for name, subagent in subagents.items():
        # Codex treats None and [] as defaults. Reject non-empty tool allowlists.
        if subagent.tools:
            unsupported.append(
                f"SubagentDef.tools for Codex subagent {name!r}. Codex subagent "
                "tool filters are not controlled by agent-sdk-wrapper"
            )
        if subagent.max_turns is not None:
            unsupported.append(
                f"SubagentDef.max_turns for Codex subagent {name!r}. Codex has no "
                "direct per-subagent max-turns option"
            )
    return unsupported


def _validate_thread_resume_options(options: dict[str, Any]) -> None:
    unsupported = sorted(set(options) - _THREAD_RESUME_OPTION_KEYS)
    if unsupported:
        raise ConfigError(
            "unsupported Codex thread_resume options with thread_id: "
            f"{', '.join(unsupported)}"
        )


@dataclasses.dataclass
class _RuntimeConfig:
    config_overrides: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@contextmanager
def _runtime_config(req: RunRequest):
    web_tools_override: tuple[str, ...] = ()
    if req.web_tools is not None:
        # Codex ignores the legacy tools.web_search flag; the top-level mode controls the tool.
        web_tools_override = (
            _config_override("web_search", value="live" if req.web_tools else "disabled"),
        )

    if not req.tools and not req.subagents and not req.mcp_servers:
        yield _RuntimeConfig(config_overrides=web_tools_override)
        return

    with tempfile.TemporaryDirectory(prefix="agent-sdk-wrapper-codex-") as tmp:
        root = Path(tmp)
        overrides: list[str] = list(web_tools_override)
        warnings: list[str] = []
        if req.tools:
            overrides.extend(
                _tool_config_overrides(
                    req.tools,
                    root,
                    req.cwd,
                    allowed_tools=req.allowed_tools,
                    disallowed_tools=req.disallowed_tools,
                )
            )
        if req.mcp_servers:
            overrides.extend(
                _mcp_config_overrides(
                    req.mcp_servers,
                    allowed_tools=req.allowed_tools,
                    disallowed_tools=req.disallowed_tools,
                )
            )
        if req.subagents:
            overrides.extend(_subagent_config_overrides(req.subagents, root, warnings))
        yield _RuntimeConfig(tuple(overrides), tuple(warnings))


def _tool_config_overrides(
    callables: list[Any],
    root: Path,
    cwd: str | Path | None,
    *,
    allowed_tools: list[str],
    disallowed_tools: list[str],
) -> list[str]:
    tool_dir = root / "tools"
    tool_dir.mkdir()
    manifest = tool_dir / "tools.json"
    script = tool_dir / "server.py"
    entries = [_tool_entry(fn) for fn in callables]
    manifest.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    script.write_text(_tool_server_script(), encoding="utf-8")

    overrides = [
        _config_override("mcp_servers", CODEX_TOOL_SERVER, "command", value=sys.executable),
        _config_override(
            "mcp_servers",
            CODEX_TOOL_SERVER,
            "args",
            value=[str(script)],
        ),
        _config_override("mcp_servers", CODEX_TOOL_SERVER, "required", value=True),
        _config_override(
            "mcp_servers",
            CODEX_TOOL_SERVER,
            "default_tools_approval_mode",
            value="approve",
        ),
    ]
    if cwd is not None:
        overrides.append(
            _config_override("mcp_servers", CODEX_TOOL_SERVER, "cwd", value=_as_str(cwd))
        )
    tool_names = [entry["name"] for entry in entries]
    enabled_tools = _server_enabled_tools(CODEX_TOOL_SERVER, None, allowed_tools, tool_names)
    disabled_tools = _server_disabled_tools(CODEX_TOOL_SERVER, [], disallowed_tools, tool_names)
    if enabled_tools is not None:
        overrides.append(
            _config_override(
                "mcp_servers",
                CODEX_TOOL_SERVER,
                "enabled_tools",
                value=enabled_tools,
            )
        )
    if disabled_tools:
        overrides.append(
            _config_override(
                "mcp_servers",
                CODEX_TOOL_SERVER,
                "disabled_tools",
                value=disabled_tools,
            )
        )
    return overrides


def _tool_entry(fn: Any) -> dict[str, Any]:
    importable = _is_importable(fn)
    try:
        source = _source_for_tool(fn)
    except ConfigError:
        if not importable:
            raise
        source = None
    return {
        "name": tool_name(fn),
        "description": tool_description(fn),
        "module": getattr(fn, "__module__", None),
        "qualname": getattr(fn, "__qualname__", None),
        "source": source,
        "source_name": getattr(fn, "__name__", None),
    }


def _is_importable(fn: Any) -> bool:
    module_name = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", "")
    if not module_name or module_name == "__main__" or "<locals>" in qualname:
        return False
    try:
        module = __import__(module_name, fromlist=["*"])
        obj: Any = module
        for part in qualname.split("."):
            obj = getattr(obj, part)
    except Exception:
        return False
    return obj is fn


def _source_for_tool(fn: Any) -> str:
    try:
        return textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError) as exc:
        source = _source_for_tool_from_repo_path(fn)
        if source is not None:
            return source
        raise ConfigError(
            "Codex Python callable tools must be importable or have inspectable source"
        ) from exc


def _source_for_tool_from_repo_path(fn: Any) -> str | None:
    code = getattr(fn, "__code__", None)
    filename = getattr(code, "co_filename", None)
    first_line = getattr(code, "co_firstlineno", None)
    if not filename or first_line is None:
        return None
    parts = Path(filename).parts
    for marker in ("src", "tests", "examples"):
        if marker not in parts:
            continue
        candidate = Path.cwd().joinpath(*parts[parts.index(marker) :])
        if not candidate.exists():
            continue
        lines = candidate.read_text(encoding="utf-8").splitlines(keepends=True)
        return textwrap.dedent("".join(inspect.getblock(lines[first_line - 1 :])))
    return None


def _tool_server_script() -> str:
    return textwrap.dedent(
        """
        from __future__ import annotations

        import importlib
        import json
        from pathlib import Path

        try:
            # mcp >= 2 renamed FastMCP to MCPServer; add_tool/run are unchanged.
            from mcp.server.mcpserver import MCPServer as _Server
        except ImportError:  # mcp < 2
            from mcp.server.fastmcp import FastMCP as _Server

        server = _Server("agent_sdk_wrapper_tools")


        def _resolve(entry):
            try:
                obj = importlib.import_module(entry["module"])
                for part in entry["qualname"].split("."):
                    obj = getattr(obj, part)
                return obj
            except Exception:
                if not entry.get("source"):
                    raise
                namespace = {}
                exec("from __future__ import annotations\\n" + entry["source"], namespace)
                return namespace[entry["source_name"]]


        for entry in json.loads(Path(__file__).with_name("tools.json").read_text()):
            server.add_tool(
                _resolve(entry),
                name=entry["name"],
                description=entry["description"],
                structured_output=False,
            )

        server.run("stdio")
        """
    ).lstrip()


def _mcp_config_overrides(
    servers: list[McpServer],
    *,
    allowed_tools: list[str],
    disallowed_tools: list[str],
) -> list[str]:
    overrides: list[str] = []
    for server in servers:
        _validate_config_key_part(server.name)
        if isinstance(server, McpStdioServer):
            overrides.append(
                _config_override("mcp_servers", server.name, "command", value=server.command)
            )
            if server.args:
                overrides.append(
                    _config_override("mcp_servers", server.name, "args", value=server.args)
                )
            if server.cwd is not None:
                overrides.append(
                    _config_override(
                        "mcp_servers", server.name, "cwd", value=_as_str(server.cwd)
                    )
                )
            env = stdio_server_env(server)
            if env:
                overrides.append(
                    _config_override("mcp_servers", server.name, "env", value=env)
                )
        elif isinstance(server, McpHttpServer):
            overrides.append(_config_override("mcp_servers", server.name, "url", value=server.url))
            if server.headers:
                overrides.append(
                    _config_override(
                        "mcp_servers", server.name, "http_headers", value=server.headers
                    )
                )
            if server.env_http_headers:
                overrides.append(
                    _config_override(
                        "mcp_servers",
                        server.name,
                        "env_http_headers",
                        value=server.env_http_headers,
                    )
                )
            if server.bearer_token_env_var:
                overrides.append(
                    _config_override(
                        "mcp_servers",
                        server.name,
                        "bearer_token_env_var",
                        value=server.bearer_token_env_var,
                    )
                )

        for key, value in _common_mcp_config(server, allowed_tools, disallowed_tools).items():
            overrides.append(_config_override("mcp_servers", server.name, key, value=value))
        for tool, mode in server.tool_approval_modes.items():
            overrides.append(
                _config_override(
                    "mcp_servers",
                    server.name,
                    "tools",
                    tool,
                    "approval_mode",
                    value=mode,
                )
            )
    return overrides


def _common_mcp_config(
    server: McpServer, allowed_tools: list[str], disallowed_tools: list[str]
) -> dict[str, Any]:
    config: dict[str, Any] = {}
    enabled_tools = _server_enabled_tools(
        server.name, server.enabled_tools, allowed_tools, server.enabled_tools
    )
    disabled_tools = _server_disabled_tools(
        server.name, server.disabled_tools, disallowed_tools, None
    )
    if enabled_tools is not None:
        config["enabled_tools"] = enabled_tools
    if disabled_tools:
        config["disabled_tools"] = disabled_tools
    for key in (
        "default_tools_approval_mode",
        "required",
        "enabled",
        "startup_timeout_sec",
        "tool_timeout_sec",
    ):
        value = getattr(server, key)
        if value is not None:
            config[key] = value
    return config


def _server_enabled_tools(
    server_name: str,
    server_enabled_tools: list[str] | None,
    allowed_tools: list[str],
    known_tools: list[str] | None,
) -> list[str] | None:
    if not allowed_tools:
        return list(server_enabled_tools) if server_enabled_tools is not None else None
    applicable = _tool_filter_names(server_name, allowed_tools)
    if known_tools is not None:
        applicable = [name for name in applicable if name in known_tools]
    if server_enabled_tools is None:
        return applicable
    allowed = set(applicable)
    return [name for name in server_enabled_tools if name in allowed]


def _server_disabled_tools(
    server_name: str,
    server_disabled_tools: list[str],
    disallowed_tools: list[str],
    known_tools: list[str] | None,
) -> list[str]:
    out = list(server_disabled_tools)
    for name in _tool_filter_names(server_name, disallowed_tools):
        if known_tools is not None and name not in known_tools:
            continue
        if name not in out:
            out.append(name)
    return out


def _tool_filter_names(server_name: str, specs: list[str]) -> list[str]:
    out: list[str] = []
    for spec in specs:
        server, tool = _split_tool_filter(spec)
        if server is None or server == server_name:
            out.append(tool)
    return out


def _split_tool_filter(spec: str) -> tuple[str | None, str]:
    if spec.startswith("mcp__"):
        parts = spec.split("__", 2)
        if len(parts) == 3 and parts[1] and parts[2]:
            return parts[1], parts[2]
    if "." in spec:
        server, tool = spec.split(".", 1)
        if server and tool:
            return server, tool
    return None, spec


def _subagent_config_overrides(
    subagents: dict[str, Any], root: Path, warnings: list[str]
) -> list[str]:
    unsupported = _unsupported_subagent_controls(subagents)
    if unsupported:
        raise ConfigError(
            "the OpenAI Codex SDK provider does not support: "
            f"{', '.join(unsupported)}"
        )
    agent_dir = root / "agents"
    agent_dir.mkdir()
    overrides = [_config_override("features", "multi_agent", value=True)]
    for name, subagent in subagents.items():
        _validate_config_key_part(name)
        config_file = agent_dir / f"{name}.config.toml"
        lines = []
        if subagent.prompt:
            lines.append(f"developer_instructions = {_toml_literal(subagent.prompt)}")
        if subagent.model:
            lines.append(f"model = {_toml_literal(subagent.model)}")
        config_file.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        overrides.extend(
            [
                _config_override("agents", name, "description", value=subagent.description),
                _config_override("agents", name, "config_file", value=str(config_file)),
            ]
        )
    return overrides


def _config_override(*parts: str, value: Any) -> str:
    return f"{_config_key(*parts)}={_toml_literal(value)}"


def _config_key(*parts: str) -> str:
    for part in parts:
        _validate_config_key_part(part)
    return ".".join(parts)


def _validate_config_key_part(part: str) -> None:
    if not _CONFIG_KEY_PART_RE.fullmatch(part):
        raise ConfigError(f"unsupported Codex config key part {part!r}")


def _toml_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_literal(item) for item in value) + "]"
    if isinstance(value, dict):
        items = [
            f"{json.dumps(str(key))} = {_toml_literal(item)}"
            for key, item in value.items()
        ]
        return "{ " + ", ".join(items) + " }"
    if value is None:
        raise ConfigError("None is not a valid Codex config override value")
    return json.dumps(str(value))


def _codex_config(
    config: Any,
    env: dict[str, str],
    _artifacts_dir: str | Path | None,
    *,
    debug: bool = False,
    config_overrides: tuple[str, ...] = (),
):
    env = _codex_env(env, debug=debug)
    codex_bin = _path_codex_bin_when_sdk_bin_missing()
    if config is None and not env and codex_bin is None and not config_overrides:
        return None

    from openai_codex import CodexConfig

    if config is None:
        return CodexConfig(
            codex_bin=codex_bin,
            config_overrides=config_overrides,
            env=env or None,
        )
    if isinstance(config, dict):
        kwargs = dict(config)
        if config_overrides and kwargs.get("launch_args_override") is not None:
            raise ConfigError(
                "Codex tools and subagents cannot be combined with launch_args_override"
            )
        if codex_bin is not None and "codex_bin" not in kwargs:
            kwargs["codex_bin"] = codex_bin
        if config_overrides:
            kwargs["config_overrides"] = tuple(kwargs.get("config_overrides", ())) + tuple(
                config_overrides
            )
        if env:
            kwargs["env"] = {**kwargs.get("env", {}), **env}
        return CodexConfig(**kwargs)
    updates: dict[str, Any] = {}
    if config_overrides and getattr(config, "launch_args_override", None) is not None:
        raise ConfigError(
            "Codex tools and subagents cannot be combined with launch_args_override"
        )
    if codex_bin is not None and getattr(config, "codex_bin", None) is None:
        updates["codex_bin"] = codex_bin
    if config_overrides:
        updates["config_overrides"] = tuple(getattr(config, "config_overrides", ())) + tuple(
            config_overrides
        )
    if env:
        current = getattr(config, "env", None) or {}
        updates["env"] = {**current, **env}
    if updates and dataclasses.is_dataclass(config):
        return dataclasses.replace(config, **updates)
    return config


def _codex_output_schema(tp: type) -> dict[str, Any]:
    """Return the schema Codex sends with ``strict: true``."""

    schema = _strict_schema(tp)
    _walk_schema(schema, lambda node: node.pop(_OPTIONAL_MARKER, None))
    return schema


def _structured_value(tp: type, value: Any) -> Any:
    """Drop nulls strict mode forced onto optional fields so their defaults apply."""

    schema = _strict_schema(tp)
    return _drop_optional_nulls(value, schema, schema)


# Marks properties that strict mode made nullable; never sent to Codex.
_OPTIONAL_MARKER = "x-agent-sdk-wrapper-optional"


def _strict_schema(tp: type) -> dict[str, Any]:
    """Rewrite a JSON schema into the Structured Outputs strict subset.

    Every property becomes required; optional ones become nullable; defaults are
    removed; ``$ref`` never has siblings; objects disallow additional properties.
    """

    root = deepcopy(json_schema_of_type(tp))
    if "$ref" in root:
        defs = root.get("$defs", {})
        root = {**deepcopy(_resolve_ref(root, root["$ref"])), "$defs": defs}
    if root.get("type") != "object":
        raise ConfigError(
            "Codex structured output requires an object schema (a Pydantic model, "
            f"dataclass, or TypedDict); {tp!r} is not an object"
        )
    _make_strict(root, root, "output")
    for name, definition in root.get("$defs", {}).items():
        _make_strict(definition, root, name)
    return root


def _make_strict(node: Any, root: dict[str, Any], where: str) -> None:
    if not isinstance(node, dict):
        return
    if "$ref" in node:
        if len(node) == 1:
            return
        siblings = {key: value for key, value in node.items() if key != "$ref"}
        resolved = deepcopy(_resolve_ref(root, node["$ref"]))
        node.clear()
        node.update({**resolved, **siblings})
    node.pop("default", None)
    if "oneOf" in node:
        node["anyOf"] = [*node.get("anyOf", []), *node.pop("oneOf")]
        node.pop("discriminator", None)
    if isinstance(node.get("allOf"), list) and len(node["allOf"]) == 1:
        node.update({**node.pop("allOf")[0], **node})
        _make_strict(node, root, where)
        return
    if not any(key in node for key in ("type", "enum", "const", "anyOf", "allOf", "$ref")):
        raise ConfigError(
            f"Codex structured output cannot express {where}: it accepts any value; "
            "strict mode needs a concrete type"
        )
    for key in ("anyOf", "allOf", "prefixItems"):
        for index, branch in enumerate(node.get(key, [])):
            _make_strict(branch, root, f"{where}[{index}]")
    if isinstance(node.get("items"), dict):
        _make_strict(node["items"], root, f"{where}[]")
    types = node.get("type")
    if types == "object" or (isinstance(types, list) and "object" in types):
        _make_object_strict(node, root, where)


def _make_object_strict(node: dict[str, Any], root: dict[str, Any], where: str) -> None:
    if (
        "properties" not in node
        or "patternProperties" in node
        or node.get("additionalProperties", False) is not False
    ):
        raise ConfigError(
            f"Codex structured output cannot express {where}: strict mode has no "
            "free-form objects; use a model with named fields instead of a dict"
        )
    properties = node["properties"]
    required = set(node.get("required", []))
    optional: list[str] = []
    for name, prop in properties.items():
        _make_strict(prop, root, f"{where}.{name}")
        if name not in required and not _schema_nullable(prop, root):
            properties[name] = {"anyOf": [prop, {"type": "null"}]}
            optional.append(name)
    node["required"] = list(properties)
    node["additionalProperties"] = False
    if optional:
        node[_OPTIONAL_MARKER] = optional


def _schema_nullable(node: Any, root: dict[str, Any]) -> bool:
    if not isinstance(node, dict):
        return False
    if "$ref" in node:
        return _schema_nullable(_resolve_ref(root, node["$ref"]), root)
    types = node.get("type")
    if types == "null" or (isinstance(types, list) and "null" in types):
        return True
    if "const" in node and node["const"] is None:
        return True
    if None in node.get("enum", ()):
        return True
    return any(_schema_nullable(branch, root) for branch in node.get("anyOf", ()))


def _resolve_ref(root: dict[str, Any], ref: str) -> dict[str, Any]:
    if not ref.startswith("#/"):
        raise ConfigError(f"Codex structured output cannot resolve schema reference {ref!r}")
    node: Any = root
    for part in ref[2:].split("/"):
        node = node.get(part) if isinstance(node, dict) else None
    if not isinstance(node, dict):
        raise ConfigError(f"Codex structured output cannot resolve schema reference {ref!r}")
    return node


def _walk_schema(node: Any, visit: Any) -> None:
    if not isinstance(node, dict):
        return
    visit(node)
    for key in ("properties", "$defs"):
        for child in node.get(key, {}).values():
            _walk_schema(child, visit)
    for key in ("anyOf", "allOf", "prefixItems"):
        for child in node.get(key, []):
            _walk_schema(child, visit)
    _walk_schema(node.get("items"), visit)


def _drop_optional_nulls(value: Any, node: Any, root: dict[str, Any]) -> Any:
    if not isinstance(node, dict):
        return value
    if "$ref" in node:
        node = _resolve_ref(root, node["$ref"])
    if isinstance(value, dict) and "properties" in node:
        optional = set(node.get(_OPTIONAL_MARKER, ()))
        properties = node["properties"]
        return {
            key: _drop_optional_nulls(item, properties.get(key), root)
            for key, item in value.items()
            if not (item is None and key in optional)
        }
    if isinstance(value, list) and isinstance(node.get("items"), dict):
        return [_drop_optional_nulls(item, node["items"], root) for item in value]
    for branch in node.get("anyOf", ()):
        resolved = _resolve_ref(root, branch["$ref"]) if "$ref" in branch else branch
        if (isinstance(value, dict) and "properties" in resolved) or (
            isinstance(value, list) and "items" in resolved
        ):
            return _drop_optional_nulls(value, resolved, root)
    return value


def _config_has_codex_bin(config: Any) -> bool:
    return bool(_config_value(config, "codex_bin"))


def _config_value(config: Any, key: str) -> Any:
    if isinstance(config, dict):
        return config.get(key)
    return getattr(config, key, None)


def _codex_cli_bin_available() -> bool:
    try:
        import codex_cli_bin  # noqa: F401
    except ImportError:
        return False
    return True


def _path_codex_bin_when_sdk_bin_missing() -> str | None:
    if _codex_cli_bin_available():
        return None
    return shutil.which("codex")


def _codex_env(env: dict[str, str], *, debug: bool = False) -> dict[str, str]:
    merged = dict(env)
    if debug:
        merged.setdefault("RUST_LOG", "debug")
        merged.setdefault("RUST_BACKTRACE", "1")
    return merged


def _write_sdk_debug_log(
    codex: Any,
    artifacts_dir: str | Path | None,
    *,
    debug: bool = False,
) -> Path | None:
    if artifacts_dir is None or not debug:
        return None
    path = sdk_dir_for(artifacts_dir) / "openai-codex.debug.log"
    lines = [
        "# OpenAI Codex SDK debug log",
        "# Captured from the SDK-managed Codex runtime stderr buffer.",
        "",
    ]
    tail = _codex_stderr_tail(codex)
    if tail:
        lines.append(tail)
    else:
        lines.append("No SDK/runtime stderr output was captured.")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def _codex_process(codex: Any) -> Any:
    return getattr(getattr(getattr(codex, "_client", None), "_sync", None), "_proc", None)


async def _raise_if_signaled(process: Any, exc: BaseException) -> None:
    """Raise ProcessTerminatedError when the app-server died from a signal."""

    from openai_codex.errors import TransportClosedError

    poll = getattr(process, "poll", None)
    if not callable(poll):
        return
    returncode = poll()
    if returncode is None and isinstance(exc, TransportClosedError):
        # stdout can close a moment before the exit status is reapable.
        try:
            returncode = await asyncio.to_thread(process.wait, 2)
        except Exception:
            return
    if not isinstance(returncode, int) or returncode >= 0:
        return
    number = -returncode
    try:
        name = signal.Signals(number).name
    except ValueError:
        name = str(number)
    raise ProcessTerminatedError(
        number, message=f"Codex app-server was killed by signal {name}: {exc}", cause=exc
    ) from exc


def _codex_stderr_tail(codex: Any) -> str | None:
    client = getattr(codex, "_client", None)
    sync_client = getattr(client, "_sync", None)
    stderr_tail = getattr(sync_client, "_stderr_tail", None)
    if callable(stderr_tail):
        try:
            return stderr_tail(limit=400)
        except TypeError:
            return stderr_tail()
    return None


def _enum_value(enum_type: Any, value: Any) -> Any:
    if value is None or isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except ValueError:
        key = str(value).replace("-", "_")
        try:
            return enum_type[key]
        except KeyError as exc:
            supported = ", ".join(member.value for member in enum_type)
            raise ConfigError(
                f"invalid {enum_type.__name__} value {value!r}; expected {supported}"
            ) from exc


def _tool_events(root: Any, event: Any, include_raw: bool) -> list[AgentEvent]:
    """Map a completed action to a tool call and result, both carrying the tool name."""

    events = _build_tool_events(root, event, include_raw)
    names = {e.id: e.name for e in events if isinstance(e, ToolCall) and e.id}
    for tool_event in events:
        if isinstance(tool_event, ToolResult) and tool_event.name is None:
            tool_event.name = names.get(tool_event.id)
    return events


def _build_tool_events(root: Any, event: Any, include_raw: bool) -> list[AgentEvent]:
    root_type = getattr(root, "type", "")
    if not root_type:
        return []
    if root_type == "commandExecution":
        item_id = _item_id(event, root)
        command = getattr(root, "command", None)
        status = getattr(root, "status", None)
        output = _command_output(root)
        return [
            ToolCall(
                id=item_id,
                name="command",
                input={"command": command} if command is not None else None,
                raw=_raw(event) if include_raw else None,
            ),
            ToolResult(
                id=item_id,
                output=output,
                is_error=_status_value(status) in {"failed", "declined"},
                raw=_raw(root) if include_raw else None,
            ),
        ]
    if root_type == "fileChange":
        item_id = _item_id(event, root)
        status = _status_value(getattr(root, "status", None))
        return [
            ToolCall(
                id=item_id,
                name="file_change",
                input={"changes": _to_plain(getattr(root, "changes", []))},
                raw=_raw(event) if include_raw else None,
            ),
            ToolResult(
                id=item_id,
                output=_stringify_output(
                    {
                        "status": status,
                        "changes": _to_plain(getattr(root, "changes", [])),
                    }
                ),
                is_error=status in {"failed", "declined"},
                raw=_raw(root) if include_raw else None,
            ),
        ]
    if root_type == "mcpToolCall":
        item_id = _item_id(event, root)
        error = getattr(root, "error", None)
        return [
            ToolCall(
                id=item_id,
                name=_tool_name(getattr(root, "server", None), getattr(root, "tool", None)),
                input=_tool_input(getattr(root, "arguments", None)),
                raw=_raw(event) if include_raw else None,
            ),
            ToolResult(
                id=item_id,
                output=_error_message(error) or _stringify_output(getattr(root, "result", None)),
                is_error=_status_value(getattr(root, "status", None)) == "failed"
                or error is not None,
                raw=_raw(root) if include_raw else None,
            ),
        ]
    if root_type == "dynamicToolCall":
        item_id = _item_id(event, root)
        return [
            ToolCall(
                id=item_id,
                name=_tool_name(getattr(root, "namespace", None), getattr(root, "tool", None)),
                input=_tool_input(getattr(root, "arguments", None)),
                raw=_raw(event) if include_raw else None,
            ),
            ToolResult(
                id=item_id,
                output=_stringify_output(getattr(root, "content_items", None)),
                is_error=_status_value(getattr(root, "status", None)) == "failed"
                or getattr(root, "success", True) is False,
                raw=_raw(root) if include_raw else None,
            ),
        ]
    if root_type == "collabAgentToolCall":
        item_id = _item_id(event, root)
        status = _status_value(getattr(root, "status", None))
        return [
            ToolCall(
                id=item_id,
                name=_tool_name("agent", getattr(root, "tool", None)),
                input={
                    key: value
                    for key, value in {
                        "prompt": getattr(root, "prompt", None),
                        "model": getattr(root, "model", None),
                        "receiver_thread_ids": getattr(root, "receiver_thread_ids", None),
                    }.items()
                    if value is not None
                }
                or None,
                raw=_raw(event) if include_raw else None,
            ),
            ToolResult(
                id=item_id,
                output=_stringify_output(
                    {
                        "status": status,
                        "agents_states": _to_plain(getattr(root, "agents_states", None)),
                    }
                ),
                is_error=status in {"failed", "cancelled", "canceled"},
                raw=_raw(root) if include_raw else None,
            ),
        ]
    if root_type == "webSearch":
        item_id = _item_id(event, root)
        query = getattr(root, "query", None)
        action = _to_plain(getattr(root, "action", None))
        return [
            ToolCall(
                id=item_id,
                name="web_search",
                input={"query": query, "action": action},
                raw=_raw(event) if include_raw else None,
            ),
            ToolResult(
                id=item_id,
                output=_stringify_output(action or {"query": query}),
                raw=_raw(root) if include_raw else None,
            ),
        ]
    if root_type == "imageView":
        item_id = _item_id(event, root)
        path = getattr(root, "path", None)
        return [
            ToolCall(
                id=item_id,
                name="view_image",
                input={"path": str(path)} if path is not None else None,
                raw=_raw(event) if include_raw else None,
            ),
            ToolResult(
                id=item_id,
                output=str(path) if path is not None else None,
                raw=_raw(root) if include_raw else None,
            ),
        ]
    if root_type == "imageGeneration":
        item_id = _item_id(event, root)
        status = _status_value(getattr(root, "status", None))
        output = {
            "status": status,
            "result": getattr(root, "result", None),
            "saved_path": _as_str(getattr(root, "saved_path", None)),
            "revised_prompt": getattr(root, "revised_prompt", None),
        }
        return [
            ToolCall(
                id=item_id,
                name="image_generation",
                input={"revised_prompt": getattr(root, "revised_prompt", None)},
                raw=_raw(event) if include_raw else None,
            ),
            ToolResult(
                id=item_id,
                output=_stringify_output(output),
                is_error=status not in {"completed", "succeeded", "success"},
                raw=_raw(root) if include_raw else None,
            ),
        ]
    return []


def _reasoning_text(root: Any) -> str:
    parts: list[str] = []
    for value in getattr(root, "summary", None) or []:
        if value:
            parts.append(str(value))
    for value in getattr(root, "content", None) or []:
        if value:
            parts.append(str(value))
    return "\n".join(parts)


def _command_output(root: Any) -> str:
    aggregated = getattr(root, "aggregated_output", None) or getattr(root, "aggregatedOutput", None)
    if aggregated is not None:
        return str(aggregated)
    pieces: list[str] = []
    stdout = getattr(root, "stdout", None)
    stderr = getattr(root, "stderr", None)
    if stdout:
        pieces.append(str(stdout))
    if stderr:
        pieces.append(str(stderr))
    status = _status_value(getattr(root, "status", None))
    if not pieces and status:
        pieces.append(status)
    return "\n".join(pieces)


def _tool_name(namespace: Any, name: Any) -> str | None:
    normalized_name = _status_value(name)
    if namespace and normalized_name:
        return f"{namespace}.{normalized_name}"
    return None if not normalized_name else normalized_name


def _tool_input(value: Any) -> dict[str, Any] | None:
    plain = _to_plain(value)
    if plain is None:
        return None
    if isinstance(plain, dict):
        return plain
    return {"value": plain}


def _error_message(error: Any) -> str | None:
    if error is None:
        return None
    return str(getattr(error, "message", None) or error)


def _stringify_output(value: Any) -> str | None:
    plain = _to_plain(value)
    if plain is None:
        return None
    if isinstance(plain, str):
        return plain
    return json.dumps(plain)


def _int_field(data: Any, *keys: str) -> int:
    if not isinstance(data, dict):
        return 0
    for key in keys:
        value = data.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0
    return 0


def _parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


_TRANSIENT = "transient_api_error"
_CODEX_ERROR_TYPES = {
    "contextWindowExceeded": "context_window_exceeded",
    "sessionBudgetExceeded": "max_budget",
    "usageLimitExceeded": "usage_limit_exceeded",
    "rateLimitExceeded": _TRANSIENT,
    "serverOverloaded": _TRANSIENT,
    "internalServerError": _TRANSIENT,
    "unauthorized": "authentication_failed",
    "badRequest": "invalid_request",
    "cyberPolicy": "refused",
    "misalignmentPolicyViolation": "refused",
    "sandboxError": "execution_error",
    "threadRollbackFailed": "execution_error",
    "activeTurnNotSteerable": "invalid_request",
}
# codexErrorInfo variants that carry an optional upstream HTTP status.
_CODEX_HTTP_ERRORS = {
    "httpConnectionFailed",
    "responseStreamConnectionFailed",
    "responseStreamDisconnected",
    "responseTooManyFailedAttempts",
}
_HTTP_STATUS_RE = re.compile(
    r"\b(?:status(?: code)?|http)[:\s]+(\d{3})\b"
    r"|\b(\d{3}) (?:bad request|unauthorized|payment required|forbidden|not found"
    r"|too many requests|internal server error|bad gateway|service unavailable"
    r"|gateway timeout)\b",
    re.IGNORECASE,
)
_ERROR_PATTERNS = tuple(
    (re.compile(pattern, re.IGNORECASE), error_type)
    for pattern, error_type in (
        (
            r"\bcontext[ _-]?window\b|\bcontext_length_exceeded\b"
            r"|\bmaximum context length\b|\bprompt is too long\b",
            "context_window_exceeded",
        ),
        (
            r"\binsufficient_quota\b|\bexceeded your current quota\b|\bquota exceeded\b"
            r"|\busage limit\b",
            "usage_limit_exceeded",
        ),
        (r"\bbilling\b|\bcredit balance\b", "billing_error"),
        (
            r"\bunauthorized\b|\bnot logged in\b|\binvalid_api_key\b"
            r"|\b(?:invalid|incorrect|missing) api key\b",
            "authentication_failed",
        ),
        (r"\bforbidden\b|\bpermission denied\b", "permission_denied"),
        (
            r"\bmodel_not_found\b|\bunknown model\b"
            r"|\bmodel\b.{0,80}?\b(?:does not exist|not found|is not supported)\b",
            "model_not_found",
        ),
        (
            r"\brate[ _-]?limit|\boverloaded\b|\bserver busy\b|\bat capacity\b"
            r"|\bstream disconnected\b|\bconnection (?:reset|refused|closed|timed out)\b"
            r"|\btimed out\b|\btemporarily unavailable\b",
            _TRANSIENT,
        ),
        (r"\binvalid_request_error\b|\bbad request\b|\binvalid prompt\b", "invalid_request"),
    )
)


def _error_event(error: Any, raw: dict[str, Any] | None = None) -> Error:
    message = _error_message(error) or "Codex reported an error"
    details = getattr(error, "additional_details", None)
    if details and details not in message:
        message = f"{message}: {details}"
    info = getattr(error, "codex_error_info", None)
    error_type = _classify_codex_error(info, message)
    return Error(
        message=message, error_type=error_type, retryable=error_type == _TRANSIENT, raw=raw
    )


def _classify_codex_error(info: Any, message: str) -> str:
    """Prefer the structured ``codexErrorInfo``; fall back to the message for ``other``."""

    plain = _to_plain(info)
    plain = getattr(plain, "value", plain)
    if isinstance(plain, str) and plain in _CODEX_ERROR_TYPES:
        return _CODEX_ERROR_TYPES[plain]
    if isinstance(plain, dict) and plain:
        kind, detail = next(iter(plain.items()))
        if kind in _CODEX_ERROR_TYPES:
            return _CODEX_ERROR_TYPES[kind]
        if kind in _CODEX_HTTP_ERRORS:
            status = detail.get("httpStatusCode") if isinstance(detail, dict) else None
            return _http_error_type(status, message) if status else _TRANSIENT
    return _message_error_type(message) or "provider_exception"


def _message_error_type(message: str) -> str | None:
    match = _HTTP_STATUS_RE.search(message)
    if match:
        return _http_error_type(int(match.group(1) or match.group(2)), message)
    return _pattern_error_type(message)


def _pattern_error_type(message: str) -> str | None:
    for pattern, error_type in _ERROR_PATTERNS:
        if pattern.search(message):
            return error_type
    return None


def _http_error_type(status: int, message: str) -> str:
    if status in (408, 429) or status >= 500:
        return _TRANSIENT
    if status == 401:
        return "authentication_failed"
    if status == 402:
        return "billing_error"
    if status == 403:
        return "permission_denied"
    specific = _pattern_error_type(message)
    if specific is not None and specific != _TRANSIENT:
        return specific
    if status in (400, 422):
        return "invalid_request"
    return f"api_error_{status}"


def _status_value(status: Any) -> str:
    return str(getattr(status, "value", status) or "")


def _item_id(event: Any, root: Any) -> str | None:
    payload = getattr(event, "payload", None)
    return (
        getattr(root, "id", None)
        or getattr(root, "item_id", None)
        or getattr(root, "itemId", None)
        or getattr(payload, "item_id", None)
        or getattr(payload, "itemId", None)
    )


def _drop_none(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _raw(value: Any) -> dict[str, Any] | None:
    plain = _to_plain(value)
    return plain if isinstance(plain, dict) else None


def _to_plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        try:
            return _to_plain(value.model_dump(mode="json", by_alias=True))
        except Exception:
            return _to_plain(value.model_dump())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        try:
            return _to_plain(dataclasses.asdict(value))
        except Exception:
            return None
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_plain(item) for item in value]
    return value


def _as_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _looks_transient(exc: BaseException) -> bool:
    return _message_error_type(str(exc)) == _TRANSIENT
