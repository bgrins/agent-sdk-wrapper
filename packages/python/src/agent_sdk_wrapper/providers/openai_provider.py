"""Codex SDK adapter. Each wrapper run maps to one Codex turn."""

from __future__ import annotations

import ast
import asyncio
import builtins
import dataclasses
import dis
import functools
import inspect
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, contextmanager, suppress
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

from pydantic import TypeAdapter, ValidationError

from ..artifacts import ProviderEventLogger
from ..classify import TRANSIENT, classify
from ..errors import (
    AgentSdkWrapperError,
    ConfigError,
    ProcessTerminatedError,
    ProviderNotAvailableError,
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
from ..mcp import McpHttpServer, McpServer, McpStdioServer
from ..request import RunRequest, normalize_effort_for_provider
from ..structured import json_schema_of_type, validate_output
from ..tools import (
    CODEX_TOOL_SERVER,
    json_schema_for,
    tool_description,
    tool_name,
    validate_tool_names,
)
from .base import ProviderAdapter

_ACCESS_TOKEN_ENV = "CODEX_ACCESS_TOKEN"
_API_KEY_ENV = "OPENAI_API_KEY"
_API_KEY_ENVS = (_API_KEY_ENV, "CODEX_API_KEY")
_CREDENTIAL_OVERRIDE_KEYS = frozenset({"cli_auth_credentials_store", "forced_login_method"})
_WEB_SEARCH_OVERRIDE_KEYS = frozenset({"web_search", "tools.web_search"})
# Keys that would replace the blank API keys the wrapper sets for commands: the
# entries themselves, or a table at either parent key.
_COMMAND_KEY_POLICY_KEYS = frozenset(
    {
        "shell_environment_policy",
        "shell_environment_policy.set",
        *(f"shell_environment_policy.set.{name}" for name in _API_KEY_ENVS),
    }
)
_SNAPSHOT_OVERRIDE_KEYS = frozenset({"features", "features.shell_snapshot"})
_CONFIG_KEY_PART_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DEFAULT_REASONING_SUMMARY = "auto"
_WRAPPER_TOOL_TIMEOUT_SEC = 600
_SHUTDOWN_GRACE_S = 2


class OpenAIProvider(ProviderAdapter):
    name = "openai"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        config: Any = None,
        codex: Any = None,
        approval_mode: Any = None,
        sandbox: Any = None,
        model_provider: str | None = None,
        thread_options: dict[str, Any] | None = None,
        turn_options: dict[str, Any] | None = None,
    ) -> None:
        self._api_key = api_key
        self._config = config
        self._codex = codex
        self._approval_mode = approval_mode
        self._sandbox = sandbox
        self._model_provider = model_provider
        self._thread_options = dict(thread_options or {})
        self._turn_options = dict(turn_options or {})

    def ensure_available(self) -> None:
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
        from openai_codex import ApprovalMode, Sandbox

        _validate_supported(req)
        self._validate_native_options(req)
        _enum_value(ApprovalMode, self._approval_mode)
        _enum_value(Sandbox, self._sandbox)
        caller_keys = _override_keys(self._config)
        thread_config_keys = self._native_config_keys(req)
        snapshot_keys = sorted(
            (caller_keys & _SNAPSHOT_OVERRIDE_KEYS)
            | (thread_config_keys & {"features.shell_snapshot"})
        )
        if snapshot_keys or any(
            "features" in config and not isinstance(config["features"], dict)
            for config in self._native_configs(req)
        ):
            raise ConfigError("Codex config cannot override features.shell_snapshot=False")
        controlled = sorted(caller_keys & _CREDENTIAL_OVERRIDE_KEYS)
        if controlled:
            raise ConfigError(
                f"config_overrides {controlled} conflict with cli_login, which controls "
                "how Codex stores and selects credentials"
            )
        if req.web_tools is not None and caller_keys & _WEB_SEARCH_OVERRIDE_KEYS:
            raise ConfigError("config_overrides for web search conflict with web_tools")
        exposed = sorted((caller_keys | thread_config_keys) & _COMMAND_KEY_POLICY_KEYS)
        if exposed:
            raise ConfigError(
                f"Codex config {exposed} would undo the shell_environment_policy.set entries "
                f"that hide {', '.join(_API_KEY_ENVS)} from model commands; set other "
                "shell_environment_policy keys one dotted key at a time"
            )
        if thread_config_keys & _CREDENTIAL_OVERRIDE_KEYS:
            raise ConfigError("thread or turn config cannot change how Codex stores credentials")
        if req.web_tools is not None and thread_config_keys & _WEB_SEARCH_OVERRIDE_KEYS:
            raise ConfigError("thread or turn config for web search conflicts with web_tools")
        if req.cli_login == "require" and (self._api_key or self._model_provider):
            raise ConfigError(
                "cli_login='require' uses the stored ChatGPT login; remove api_key and "
                "model_provider"
            )
        if self._api_key and not self._launches_codex():
            raise ConfigError(
                "api_key requires the provider to launch Codex; authenticate the "
                "pre-built codex client or custom launch command instead"
            )
        if not self._launches_codex() and (
            req.tools or req.subagents or req.mcp_servers or req.web_tools is not None
        ):
            raise ConfigError(
                "Codex tools, subagents, MCP servers and web_tools require the provider "
                "to launch Codex; a pre-built codex client or launch_args_override "
                "cannot be reconfigured"
            )

    def check_credentials(self, req: RunRequest) -> str | None:
        # require is checked against the runtime's account once it starts.
        if (
            req.cli_login == "require"
            or not self._launches_codex()
            or self._uses_custom_model_provider(req)
        ):
            return None
        if self._login_api_key(req):
            return None
        return (
            "no OpenAI API key: set OPENAI_API_KEY or provider_options={'api_key': ...}; "
            "cli_login='deny' never uses a stored Codex login"
        )

    def _launches_codex(self) -> bool:
        return self._codex is None and _config_value(self._config, "launch_args_override") is None

    def _login_api_key(self, req: RunRequest) -> str | None:
        # A pre-built client or custom launch command cannot take the ephemeral
        # credential store override, so logging in would overwrite auth.json.
        if not self._launches_codex():
            return None
        if self._api_key:
            return self._api_key
        # The run's env is what the child sees; an explicit empty value means no key.
        if _API_KEY_ENV in req.env:
            return req.env[_API_KEY_ENV] or None
        return os.environ.get(_API_KEY_ENV) or None

    def _native_configs(self, req: RunRequest) -> list[dict[str, Any]]:
        """Any per-thread or per-turn ``config`` the caller passes."""

        configs = []
        for options in (
            self._thread_options,
            self._turn_options,
            req.extra_options.get("thread_options", {}),
            req.extra_options.get("turn_options", {}),
        ):
            config = options.get("config") if isinstance(options, dict) else None
            if isinstance(config, dict):
                configs.append(config)
        return configs

    def _native_config_keys(self, req: RunRequest) -> set[str]:
        keys: set[str] = set()
        for config in self._native_configs(req):
            keys |= _dotted_keys(config)
        return keys

    def _uses_custom_model_provider(self, req: RunRequest) -> bool:
        thread_options = {
            **self._thread_options,
            **req.extra_options.get("thread_options", {}),
        }
        return bool(
            self._model_provider
            or thread_options.get("model_provider")
            or "model_provider" in _override_keys(self._config)
        )

    async def stream(self, req: RunRequest) -> AsyncIterator[AgentEvent]:
        self.validate_request(req)
        self.ensure_available()
        problem = self.check_credentials(req)
        if problem:
            yield Error(message=problem, error_type="authentication_failed")
            return

        codex: Any = None
        try:
            api_key = None if req.cli_login == "require" else self._login_api_key(req)
            with _runtime_config(req) as config_overrides:
                if self._launches_codex():
                    # Shell snapshots copy the runtime's env, credentials included, into
                    # CODEX_HOME. Commands inherit that env and their output is recorded,
                    # so they get blank API keys; MCP servers and model providers keep them.
                    config_overrides += (
                        _config_override("features", "shell_snapshot", value=False),
                        *(
                            _config_override("shell_environment_policy", "set", name, value="")
                            for name in _API_KEY_ENVS
                        ),
                    )
                if req.cli_login != "require" and self._launches_codex():
                    # Never read or write auth.json; an API key stays in memory.
                    config_overrides += (
                        _config_override("cli_auth_credentials_store", value="ephemeral"),
                    )
                async with self._codex_client(req, config_overrides) as codex:
                    process = _codex_process(codex)
                    try:
                        async for event in self._run(codex, req, api_key):
                            yield event
                    except Exception as exc:
                        await _raise_if_signaled(process, exc)
                        raise
        except AgentSdkWrapperError:
            raise
        except FileNotFoundError as exc:
            raise ProviderNotAvailableError(str(exc), cause=exc) from exc
        except Exception as exc:
            raise AgentSdkWrapperError(f"{type(exc).__name__}: {exc}", cause=exc) from exc

    async def _run(
        self, codex: Any, req: RunRequest, api_key: str | None
    ) -> AsyncIterator[AgentEvent]:
        from openai_codex import ApprovalMode, Sandbox

        if api_key:
            await codex.login_api_key(api_key)
        # Under deny, a custom provider may not use OpenAI auth, and the ephemeral store
        # still keeps auth.json out. require always needs the stored ChatGPT login.
        skip_account = req.cli_login != "require" and self._uses_custom_model_provider(req)
        problem = None if skip_account else _account_problem(await codex.account(), req.cli_login)
        if problem:
            yield Error(message=problem, error_type="authentication_failed")
            return

        approval_mode = _enum_value(ApprovalMode, self._approval_mode)
        sandbox = _enum_value(Sandbox, self._sandbox)
        thread_kwargs, turn_kwargs = self._build_options(req, approval_mode, sandbox)

        if req.session_id:
            thread = await codex.thread_resume(req.session_id, **thread_kwargs)
        else:
            thread = await codex.thread_start(**thread_kwargs)
        yield SessionInfo(id=thread.id, model=await _thread_model(thread))

        # A caller-owned client may consume its own global notifications.
        runtime_warnings = (
            None if self._codex is not None else _RuntimeWarnings(codex, thread.id, req.include_raw)
        )
        if runtime_warnings is not None:
            for warning in runtime_warnings.drain():
                yield warning

        turn = await thread.turn(req.prompt, **turn_kwargs)
        async for event in _stream_turn(turn, req, thread.id, runtime_warnings):
            yield event

    @asynccontextmanager
    async def _codex_client(self, req: RunRequest, config_overrides: tuple[str, ...] = ()):
        if self._codex is not None:
            yield self._codex
            return

        from openai_codex import AsyncCodex
        from openai_codex.errors import TransportClosedError

        # An access token is a ChatGPT login that replaces the stored one and that
        # commands would inherit; Codex treats an empty value as unset.
        env = {**req.env, _ACCESS_TOKEN_ENV: ""}
        config = _codex_config(self._config, env, config_overrides=config_overrides)
        codex = AsyncCodex(config=config)
        # A failed handshake closes the process, so keep its handle to read how it died.
        await codex._client.start()
        process = _codex_process(codex)
        try:
            await codex.__aenter__()
        except TransportClosedError as exc:
            await _raise_if_signaled(process, exc)
            raise
        try:
            yield codex
        finally:
            try:
                await asyncio.to_thread(_let_exit, process)
            finally:
                await codex.close()

    def _native_options(self, req: RunRequest) -> tuple[dict[str, Any], dict[str, Any]]:
        from openai_codex import ApprovalMode, Sandbox

        extra = dict(req.extra_options)
        thread_options = {**self._thread_options, **extra.pop("thread_options", {})}
        turn_options = {**self._turn_options, **extra.pop("turn_options", {})}
        if extra:
            keys = ", ".join(sorted(extra))
            raise ConfigError(
                "unsupported Codex SDK extra_options keys: "
                f"{keys}. Use 'thread_options' or 'turn_options'."
            )
        # The SDK maps these enums itself and rejects their string values after launch.
        for options in (thread_options, turn_options):
            for key, enum_type in (("approval_mode", ApprovalMode), ("sandbox", Sandbox)):
                if key in options:
                    options[key] = _enum_value(enum_type, options[key])
        return thread_options, turn_options

    def _validate_native_options(self, req: RunRequest) -> None:
        thread_options, turn_options = self._native_options(req)
        resuming = bool(req.session_id)
        if thread_options.get("ephemeral") and (resuming or req.continue_session):
            raise ConfigError(
                "ephemeral Codex threads cannot be resumed: each run starts a new "
                "app-server, so session_id and continue_session would not find the thread"
            )
        names = _sdk_option_names()
        method = "thread_resume" if resuming else "thread_start"
        # Resuming drops start-only options, so a continued session keeps its first run's.
        allowed = names[method] | (_start_only_options() if resuming else set())
        unknown = sorted(set(thread_options) - allowed)
        if unknown:
            raise ConfigError(f"unsupported Codex {method} options: {', '.join(unknown)}")
        unknown = sorted(set(turn_options) - names["turn"])
        if unknown:
            raise ConfigError(f"unsupported Codex turn options: {', '.join(unknown)}")
        for kind, options in (("thread", thread_options), ("turn", turn_options)):
            validators = _sdk_option_validators()[kind]
            for key, value in options.items():
                if key not in validators:
                    continue
                try:
                    validators[key].validate_python(value)
                except ValidationError as exc:
                    raise ConfigError(
                        f"invalid Codex {kind} option {key}={value!r}: {exc.errors()[0]['msg']}"
                    ) from exc

    def _build_options(
        self, req: RunRequest, approval_mode: Any, sandbox: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        req_effort = normalize_effort_for_provider("openai", req.effort)
        thread_options, turn_options = self._native_options(req)

        thread_options.setdefault("model", req.model)
        thread_options.setdefault("model_provider", self._model_provider)
        thread_options.setdefault("cwd", _as_str(req.cwd))
        thread_options.setdefault("developer_instructions", req.system_prompt)
        thread_options.setdefault("approval_mode", approval_mode)
        thread_options.setdefault("sandbox", sandbox)
        if req.session_id:
            for key in _start_only_options():
                thread_options.pop(key, None)

        turn_options.setdefault("model", req.model)
        turn_options.setdefault("cwd", _as_str(req.cwd))
        turn_options.setdefault("approval_mode", approval_mode)
        # No turn sandbox: the SDK sends it as a full policy with default writable roots
        # and network access, overriding sandbox_workspace_write from config.
        turn_options.setdefault("effort", req_effort)
        # Codex omits reasoning text without a summary mode.
        turn_options.setdefault("summary", _DEFAULT_REASONING_SUMMARY)
        if req.output_schema is not None:
            turn_options.setdefault("output_schema", _codex_output_schema(req.output_schema))

        return _drop_none(thread_options), _drop_none(turn_options)


async def _stream_turn(
    turn: Any,
    req: RunRequest,
    thread_id: str | None = None,
    runtime_warnings: _RuntimeWarnings | None = None,
) -> AsyncIterator[AgentEvent]:
    """Normalize one Codex turn, ending with its terminal state."""

    provider_log = ProviderEventLogger(
        "openai",
        req.artifacts_dir,
        req.on_provider_event,
        run_id=req.run_id,
    )
    text_delta_parts: dict[str | None, list[str]] = {}
    thinking_delta_parts: dict[str | None, list[str]] = {}
    texts: list[str] = []
    usage = _TurnUsage(resumed=bool(req.session_id))
    reasoned = False
    started_calls: set[str] = set()
    latest_plan: list[Any] | None = None
    # A non-retried error notification precedes the failed turn/completed; emit one Error.
    reported_error: Error | None = None

    async for event in turn.stream():
        provider_log.write(event)
        if runtime_warnings is not None:
            for warning in runtime_warnings.drain():
                yield warning
        method = getattr(event, "method", "")
        payload = getattr(event, "payload", None)
        usage.observe(method)
        if method == "item/agentMessage/delta":
            delta = getattr(payload, "delta", "") or ""
            if delta:
                item_id = getattr(payload, "item_id", None)
                text_delta_parts.setdefault(item_id, []).append(delta)
            continue

        if method in {
            "item/reasoning/textDelta",
            "item/reasoning/summaryTextDelta",
        }:
            delta = getattr(payload, "delta", "") or ""
            if delta:
                item_id = getattr(payload, "item_id", None)
                thinking_delta_parts.setdefault(item_id, []).append(delta)
            continue

        if method == "item/started":
            item = getattr(payload, "item", None)
            root = getattr(item, "root", item)
            # A started web search has no query yet; its call is emitted on completion.
            if getattr(root, "type", "") == "webSearch":
                continue
            tool_events = _tool_events(root, event, req.include_raw)
            if tool_events is not None:
                call = tool_events[0]
                if call.id is not None:
                    started_calls.add(call.id)
                yield call
            continue

        if method == "item/completed":
            item = getattr(payload, "item", None)
            root = getattr(item, "root", item)
            root_type = getattr(root, "type", "")
            if root_type == "agentMessage":
                item_id = getattr(root, "id", None)
                buffered_text = _pop_delta_buffer(text_delta_parts, item_id)
                text = getattr(root, "text", "") or buffered_text
                if text:
                    texts.append(text)
                    yield Text(text=text, raw=_raw(event) if req.include_raw else None)
                continue
            if root_type == "reasoning":
                item_id = getattr(root, "id", None)
                buffered_text = _pop_delta_buffer(thinking_delta_parts, item_id)
                text = _reasoning_text(root) or buffered_text
                # Preserve empty reasoning items: they can carry billed tokens.
                reasoned = True
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
            if tool_events is not None:
                call, result = tool_events
                if call.id is None or call.id not in started_calls:
                    yield call
                started_calls.discard(call.id)
                yield result
            continue

        if method == "turn/plan/updated":
            latest_plan = _field(payload, "plan", "plan") or []
            continue

        if method == "thread/tokenUsage/updated":
            usage.add(payload.token_usage)
            continue

        if method == "model/rerouted":
            from_model = getattr(payload, "from_model", None)
            to_model = getattr(payload, "to_model", None)
            reason = _to_plain(getattr(payload, "reason", None))
            yield WarningEvent(
                message=f"Codex rerouted the turn from {from_model} to {to_model} ({reason})",
                raw=_raw(event) if req.include_raw else None,
            )
            yield SessionInfo(id=thread_id or getattr(payload, "thread_id", ""), model=to_model)
            continue

        if method == "error":
            error = _field(payload, "error", "error")
            # Keep SDK retries as warnings; the terminal state determines success.
            if _field(payload, "will_retry", "willRetry"):
                yield WarningEvent(message=_turn_error_text(error))
            else:
                reported_error = _error_event(error, _raw(event) if req.include_raw else None)
            continue

        if method == "turn/completed":
            if latest_plan:
                yield Thinking(text=_plan_text(latest_plan))
                latest_plan = None
            for text in _drain_delta_buffers(text_delta_parts):
                texts.append(text)
                yield Text(text=text)
            for text in _drain_delta_buffers(thinking_delta_parts):
                reasoned = True
                yield Thinking(text=text)
            usage_event = usage.event(req.include_raw)
            if usage_event is not None:
                if usage_event.usage.reasoning_output_tokens and not reasoned:
                    yield Thinking(text="")
                yield usage_event
            turn_info = _field(payload, "turn", "turn")
            status = _status_value(_field(turn_info, "status", "status"))
            if status == "failed":
                error = _field(turn_info, "error", "error")
                if error is not None:
                    yield _error_event(error, _raw(event) if req.include_raw else None)
                else:
                    yield reported_error or Error(
                        message="Codex turn failed", error_type="provider_exception"
                    )
                return
            if status == "interrupted":
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


async def _thread_model(thread: Any) -> str | None:
    """Return the model the runtime resolved for a started or resumed thread."""

    from openai_codex.errors import CodexError

    read = getattr(thread, "read", None)
    if not callable(read):
        return None
    try:
        response = await read()
    except CodexError:
        return None
    model = getattr(getattr(response, "thread", None), "model", None)
    return model if isinstance(model, str) and model else None


class _RuntimeWarnings:
    """Surface thread-level runtime notifications, which arrive outside the turn stream.

    MCP startup failures and config warnings reach only the SDK's global queue, which
    nothing else reads when the provider owns the client.
    """

    def __init__(self, codex: Any, thread_id: str, include_raw: bool) -> None:
        router = getattr(getattr(getattr(codex, "_client", None), "_sync", None), "_router", None)
        self._queue = getattr(router, "_global_notifications", None)
        self._thread_id = thread_id
        self._include_raw = include_raw

    def drain(self) -> list[WarningEvent]:
        warnings: list[WarningEvent] = []
        while self._queue is not None:
            try:
                notification = self._queue.get_nowait()
            except queue.Empty:
                break
            warning = self._warning(notification)
            if warning is not None:
                warnings.append(warning)
        return warnings

    def _warning(self, notification: Any) -> WarningEvent | None:
        method = getattr(notification, "method", None)
        payload = getattr(notification, "payload", None)
        thread_id = getattr(payload, "thread_id", None)
        if thread_id is not None and thread_id != self._thread_id:
            return None
        if method == "mcpServer/startupStatus/updated":
            if _status_value(getattr(payload, "status", None)) != "failed":
                return None
            name = getattr(payload, "name", "")
            message = getattr(payload, "error", None) or f"MCP server {name!r} failed to start"
        elif method == "configWarning":
            details = getattr(payload, "details", None)
            summary = getattr(payload, "summary", "")
            message = f"{summary}: {details}" if details else summary
        elif method == "warning":
            message = getattr(payload, "message", "")
        else:
            return None
        return WarningEvent(
            message=message, raw=_raw(notification) if self._include_raw else None
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
    for fn in req.tools:
        _tool_entry(fn)
    if req.output_schema is not None:
        _codex_output_schema(req.output_schema)
    _mcp_config_overrides(req.mcp_servers)
    for name, subagent in req.subagents.items():
        _validate_config_key_part(name)
        _toml_literal([subagent.description, subagent.prompt, subagent.model or ""])
    unsupported: list[str] = []
    if req.max_turns is not None:
        unsupported.append("max_turns. Codex has no turn limit")
    if req.builtin_tools is not None:
        unsupported.append(
            "builtin_tools. Codex built-in tools cannot be disabled or allowlisted "
            "through agent-sdk-wrapper yet"
        )
    if req.permission_mode is not None:
        unsupported.append("permission_mode")
    if req.setting_sources is not None:
        unsupported.append("setting_sources")
    if req.allowed_tools or req.disallowed_tools:
        unsupported.append(
            "allowed_tools/disallowed_tools. Pass only the callable tools you want, and "
            "filter MCP server tools with enabled_tools/disabled_tools"
        )
    unsupported_subagents = _unsupported_subagent_controls(req.subagents)
    if unsupported_subagents:
        unsupported.extend(unsupported_subagents)
    if unsupported:
        raise ConfigError(
            "the OpenAI Codex SDK provider does not support: "
            f"{', '.join(unsupported)}"
        )


@dataclasses.dataclass
class _TurnUsage:
    """Per-turn usage from the thread's cumulative ``total`` and per-request ``last``.

    An update that changes ``total`` reports one request, whose usage is ``last``,
    so resumed history is excluded without state kept across runs. Codex repeats the
    unchanged usage before the ``error`` notification of a failed request, and reports
    an exhausted context window with an empty ``last``; neither is a request. A resumed
    thread's first update has no earlier total to compare, so it waits for the next
    notification: an error shows it was a repeat.
    """

    resumed: bool
    usage: dict[str, int] = dataclasses.field(default_factory=dict)
    requests: int = 0
    total: dict[str, int] | None = None
    first: dict[str, int] | None = None
    raw: dict[str, Any] | None = None

    def add(self, token_usage: Any) -> None:
        total = _usage_breakdown(token_usage.total)
        last = _usage_breakdown(token_usage.last)
        if not last["total_tokens"]:
            return
        self._settle(counts=True)
        if self.total is None and self.resumed:
            self.first = last
        elif total != self.total:
            self._count(last)
        self.total = total
        self.raw = _raw(token_usage)

    def observe(self, method: str) -> None:
        """Settle a waiting first update by the method of a notification after it."""

        if method != "thread/tokenUsage/updated":
            self._settle(counts=method != "error")

    def _settle(self, *, counts: bool) -> None:
        if self.first is not None and counts:
            self._count(self.first)
        self.first = None

    def _count(self, last: dict[str, int]) -> None:
        for key, value in last.items():
            self.usage[key] = self.usage.get(key, 0) + value
        self.requests += 1

    def event(self, include_raw: bool) -> Usage | None:
        if not self.requests:
            return None
        return Usage(
            usage=TokenUsage(**self.usage, requests=self.requests),
            raw=self.raw if include_raw else None,
        )


def _usage_breakdown(breakdown: Any) -> dict[str, int]:
    """Normalize a ``TokenUsageBreakdown``; output already includes reasoning."""

    # totalTokens is not always input plus output: a full context window reports its size.
    return {
        "input_tokens": breakdown.input_tokens,
        "cache_read_tokens": breakdown.cached_input_tokens,
        "cache_write_tokens": breakdown.cache_write_input_tokens or 0,
        "output_tokens": breakdown.output_tokens,
        "reasoning_output_tokens": breakdown.reasoning_output_tokens,
        "total_tokens": breakdown.input_tokens + breakdown.output_tokens,
    }


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


@functools.cache
def _sdk_option_names() -> dict[str, frozenset[str]]:
    """Keyword options the SDK's thread and turn methods accept."""

    from openai_codex import AsyncCodex, AsyncThread

    def keywords(method: Any) -> frozenset[str]:
        return frozenset(
            name
            for name, param in inspect.signature(method).parameters.items()
            if param.kind is inspect.Parameter.KEYWORD_ONLY
        )

    return {
        "thread_start": keywords(AsyncCodex.thread_start),
        "thread_resume": keywords(AsyncCodex.thread_resume),
        "turn": keywords(AsyncThread.turn),
    }


def _start_only_options() -> frozenset[str]:
    names = _sdk_option_names()
    return names["thread_start"] - names["thread_resume"]


# SDK keywords whose wire params field has another name.
_SDK_PARAM_FIELDS = {
    "include_turns": "exclude_turns",
    "source": "turn_trigger",
    "turn_service_tier": "service_tier_for_turn",
}


@functools.cache
def _sdk_option_validators() -> dict[str, dict[str, TypeAdapter[Any]]]:
    """Validators for option values from the SDK's wire params, which it checks after launch.

    ``approval_mode`` and ``sandbox`` are SDK enums mapped to other wire types.
    """

    from openai_codex.generated.v2_all import (
        ThreadResumeParams,
        ThreadStartParams,
        TurnStartParams,
    )

    def validators(keywords: frozenset[str], fields: dict[str, Any]) -> dict[str, TypeAdapter[Any]]:
        out = {}
        for key in keywords - {"approval_mode", "sandbox"}:
            field = fields.get(_SDK_PARAM_FIELDS.get(key, key))
            if field is not None:
                out[key] = TypeAdapter(field.annotation)
        return out

    names = _sdk_option_names()
    thread_fields = {**ThreadResumeParams.model_fields, **ThreadStartParams.model_fields}
    return {
        "thread": validators(names["thread_start"] | names["thread_resume"], thread_fields),
        "turn": validators(names["turn"], TurnStartParams.model_fields),
    }


@contextmanager
def _runtime_config(req: RunRequest):
    """Yield the config overrides for the request's tools, MCP servers and subagents."""

    web_tools_override: tuple[str, ...] = ()
    if req.web_tools is not None:
        # Codex ignores the legacy tools.web_search flag; the top-level mode controls the tool.
        web_tools_override = (
            _config_override("web_search", value="live" if req.web_tools else "disabled"),
        )

    if not req.tools and not req.subagents and not req.mcp_servers:
        yield web_tools_override
        return

    with tempfile.TemporaryDirectory(prefix="agent-sdk-wrapper-codex-") as tmp:
        root = Path(tmp)
        overrides: list[str] = list(web_tools_override)
        if req.tools:
            overrides.extend(_tool_config_overrides(req.tools, root, req.cwd, req.env))
        if req.mcp_servers:
            overrides.extend(_mcp_config_overrides(req.mcp_servers))
        if req.subagents:
            overrides.extend(_subagent_config_overrides(req.subagents, root))
        yield tuple(overrides)


def _tool_config_overrides(
    callables: list[Any], root: Path, cwd: str | Path | None, env: dict[str, str]
) -> list[str]:
    tool_dir = root / "tools"
    tool_dir.mkdir()
    manifest = _tool_manifest(callables)
    script = tool_dir / "server.py"
    (tool_dir / "tools.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
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
        # Codex starts stdio servers with a minimal env; forward the parent's by name
        # so values never appear on the command line.
        _config_override(
            "mcp_servers",
            CODEX_TOOL_SERVER,
            "env_vars",
            value=sorted(name for name in {**os.environ, **env} if _ENV_NAME_RE.fullmatch(name)),
        ),
        # Codex's default MCP tool timeout is too short for arbitrary Python callables.
        _config_override(
            "mcp_servers",
            CODEX_TOOL_SERVER,
            "tool_timeout_sec",
            value=_WRAPPER_TOOL_TIMEOUT_SEC,
        ),
    ]
    if cwd is not None:
        overrides.append(
            _config_override("mcp_servers", CODEX_TOOL_SERVER, "cwd", value=_as_str(cwd))
        )
    return overrides


def _tool_manifest(callables: list[Any]) -> dict[str, Any]:
    """Describe the tool server: the parent's import path and one entry per tool."""

    return {
        # JSON cannot carry lone surrogates, which undecodable path bytes become.
        "sys_path": [
            os.path.abspath(path or os.curdir) for path in sys.path if _utf8_encodable(path)
        ],
        "tools": [_tool_entry(fn) for fn in callables],
    }


def _tool_entry(fn: Any) -> dict[str, Any]:
    """Describe how the tool server loads ``fn``: by import, or from its source."""

    json_schema_for(fn)
    source = None
    if not _is_importable(fn):
        source = _source_for_tool(fn)
        _check_source_fallback(fn, source)
    return {
        "name": tool_name(fn),
        "description": tool_description(fn),
        "module": getattr(fn, "__module__", None),
        "qualname": getattr(fn, "__qualname__", None),
        "source": source,
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
        raise ConfigError(
            "Codex Python callable tools must be importable or have inspectable source"
        ) from exc


def _check_source_fallback(fn: Any, source: str) -> None:
    """Reject callables whose source cannot run alone in the tool server.

    The server execs a non-importable callable's source in an empty namespace, so
    module globals, closure variables and decorators are unavailable there.
    """

    name = getattr(fn, "__name__", "")
    problems: list[str] = []
    if inspect.ismethod(fn):
        problems.append("it is a bound method")
    code = getattr(fn, "__code__", None)
    if code is not None and code.co_freevars:
        problems.append("it closes over " + ", ".join(code.co_freevars))
    try:
        definition = ast.parse(source).body[0]
    except (SyntaxError, IndexError):
        definition = None
    if not isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef)) or (
        definition.name != name
    ):
        problems.append("its source is not a plain function definition")
    else:
        if definition.decorator_list:
            problems.append("it is decorated")
        used = _global_names(code) if code is not None else set()
        used |= {
            node.id
            for part in _signature_nodes(definition)
            for node in ast.walk(part)
            if isinstance(node, ast.Name)
        }
        free = sorted(used - {name} - set(vars(builtins)))
        if free:
            problems.append("it uses module-level names " + ", ".join(free))
    if problems:
        raise ConfigError(
            f"Codex tool {tool_name(fn)!r} cannot be imported by the tool server and its "
            f"source cannot run alone: {'; '.join(problems)}. Define it at module level "
            "in an importable module"
        )


def _global_names(code: Any) -> set[str]:
    names = {
        instruction.argval
        for instruction in dis.get_instructions(code)
        if instruction.opname in {"LOAD_GLOBAL", "LOAD_NAME", "STORE_GLOBAL", "DELETE_GLOBAL"}
    }
    for const in code.co_consts:
        if inspect.iscode(const):
            names |= _global_names(const)
    return names


def _signature_nodes(definition: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.AST]:
    args = definition.args
    parameters = [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]
    nodes: list[ast.AST] = [
        param.annotation for param in parameters if param is not None and param.annotation
    ]
    nodes.extend(args.defaults)
    nodes.extend(default for default in args.kw_defaults if default is not None)
    if definition.returns is not None:
        nodes.append(definition.returns)
    return nodes


def _tool_server_script() -> str:
    """Serve the manifest's tools with the same schemas and calls as the Claude path."""

    return textwrap.dedent(
        """
        from __future__ import annotations

        import importlib
        import json
        import sys
        from pathlib import Path

        manifest = json.loads(Path(__file__).with_name("tools.json").read_text(encoding="utf-8"))
        # Import tools from the same paths the parent process used.
        sys.path[:0] = [path for path in manifest["sys_path"] if path not in sys.path]


        def _resolve(entry):
            if entry["source"] is None:
                obj = importlib.import_module(entry["module"])
                for part in entry["qualname"].split("."):
                    obj = getattr(obj, part)
                return obj
            namespace = {}
            exec("from __future__ import annotations\\n" + entry["source"], namespace)
            return namespace[entry["name"]]


        def _load():
            from agent_sdk_wrapper.tools import json_schema_for, tool_caller
            from mcp import types

            tools = {}
            for entry in manifest["tools"]:
                try:
                    fn = _resolve(entry)
                    schema = json_schema_for(fn)
                    call = tool_caller(fn)
                except Exception as exc:
                    raise RuntimeError(
                        f"cannot load tool {entry['name']!r}: {type(exc).__name__}: {exc}"
                    ) from exc
                spec = types.Tool(
                    name=entry["name"], description=entry["description"], input_schema=schema
                )
                tools[entry["name"]] = (spec, call)
            return tools


        def _refuse_initialize(message):
            # Codex reports an initialize error in its startup failure; an exit only
            # reports a closed connection.
            for line in sys.stdin:
                request = json.loads(line)
                if request.get("method") == "initialize":
                    error = {"code": -32603, "message": message}
                    response = {"jsonrpc": "2.0", "id": request["id"], "error": error}
                    print(json.dumps(response), flush=True)
                    return


        def _serve(tools):
            import anyio
            from mcp import types
            from mcp.server.lowlevel import Server
            from mcp.server.stdio import stdio_server

            async def list_tools(ctx, params):
                return types.ListToolsResult(tools=[spec for spec, _ in tools.values()])

            async def call_tool(ctx, params):
                if params.name not in tools:
                    text, is_error = f"Error: unknown tool {params.name!r}", True
                else:
                    text, is_error = await tools[params.name][1](params.arguments or {})
                content = [types.TextContent(type="text", text=text)]
                return types.CallToolResult(content=content, is_error=is_error)

            server = Server(
                "agent_sdk_wrapper_tools", on_list_tools=list_tools, on_call_tool=call_tool
            )

            async def main():
                async with stdio_server() as (read_stream, write_stream):
                    options = server.create_initialization_options()
                    await server.run(read_stream, write_stream, options)

            anyio.run(main)


        try:
            loaded = _load()
        except Exception as exc:
            _refuse_initialize(str(exc))
            raise
        _serve(loaded)
        """
    ).lstrip()


def _mcp_config_overrides(servers: list[McpServer]) -> list[str]:
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
            if server.env_passthrough:
                # Codex copies these from its own env; values stay off the command line.
                overrides.append(
                    _config_override(
                        "mcp_servers", server.name, "env_vars", value=server.env_passthrough
                    )
                )
            if server.env:
                overrides.append(
                    _config_override("mcp_servers", server.name, "env", value=server.env)
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

        for key, value in _common_mcp_config(server).items():
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


def _common_mcp_config(server: McpServer) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if server.enabled_tools is not None:
        config["enabled_tools"] = list(server.enabled_tools)
    if server.disabled_tools:
        config["disabled_tools"] = list(server.disabled_tools)
    # Configured servers are trusted, as on Claude, where their tools are pre-approved.
    config["default_tools_approval_mode"] = server.default_tools_approval_mode or "approve"
    for key in (
        "required",
        "enabled",
        "startup_timeout_sec",
        "tool_timeout_sec",
    ):
        value = getattr(server, key)
        if value is not None:
            config[key] = value
    return config


def _subagent_config_overrides(subagents: dict[str, Any], root: Path) -> list[str]:
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
        return _toml_string(value)
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_literal(item) for item in value) + "]"
    if isinstance(value, dict):
        items = [
            f"{_toml_string(str(key))} = {_toml_literal(item)}"
            for key, item in value.items()
        ]
        return "{ " + ", ".join(items) + " }"
    if value is None:
        raise ConfigError("None is not a valid Codex config override value")
    return _toml_string(str(value))


_TOML_ESCAPES = {
    '"': '\\"',
    "\\": "\\\\",
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
}


def _toml_string(value: str) -> str:
    """Encode a TOML basic string.

    JSON escapes split non-BMP characters into surrogate pairs, which TOML rejects.
    """

    out = ['"']
    for char in value:
        code = ord(char)
        if char in _TOML_ESCAPES:
            out.append(_TOML_ESCAPES[char])
        elif code < 0x20 or code == 0x7F:
            out.append(f"\\u{code:04X}")
        elif 0xD800 <= code <= 0xDFFF:
            raise ConfigError(f"Codex config strings cannot contain lone surrogates: {value!r}")
        else:
            out.append(char)
    out.append('"')
    return "".join(out)


def _codex_config(
    config: Any,
    env: dict[str, str],
    *,
    config_overrides: tuple[str, ...] = (),
):
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
            # Later overrides win, so the caller's own config_overrides come last.
            kwargs["config_overrides"] = tuple(config_overrides) + tuple(
                kwargs.get("config_overrides", ())
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
        updates["config_overrides"] = tuple(config_overrides) + tuple(
            getattr(config, "config_overrides", ())
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


def _make_strict(
    node: Any, root: dict[str, Any], where: str, inlining: frozenset[str] = frozenset()
) -> None:
    if not isinstance(node, dict):
        return
    if "$ref" in node:
        if len(node) == 1:
            return
        ref = node["$ref"]
        if ref in inlining:
            # Inlining a recursive reference never ends; its annotations are optional.
            node.clear()
            node["$ref"] = ref
            return
        siblings = {key: value for key, value in node.items() if key != "$ref"}
        resolved = deepcopy(_resolve_ref(root, ref))
        node.clear()
        node.update({**resolved, **siblings})
        inlining = inlining | {ref}
    node.pop("default", None)
    if "oneOf" in node:
        node["anyOf"] = [*node.get("anyOf", []), *node.pop("oneOf")]
        node.pop("discriminator", None)
    if isinstance(node.get("allOf"), list) and len(node["allOf"]) == 1:
        node.update({**node.pop("allOf")[0], **node})
        _make_strict(node, root, where, inlining)
        return
    if not any(key in node for key in ("type", "enum", "const", "anyOf", "allOf", "$ref")):
        raise ConfigError(
            f"Codex structured output cannot express {where}: it accepts any value; "
            "strict mode needs a concrete type"
        )
    for key in ("anyOf", "allOf", "prefixItems"):
        for index, branch in enumerate(node.get(key, [])):
            _make_strict(branch, root, f"{where}[{index}]", inlining)
    if isinstance(node.get("items"), dict):
        _make_strict(node["items"], root, f"{where}[]", inlining)
    types = node.get("type")
    if types == "object" or (isinstance(types, list) and "object" in types):
        _make_object_strict(node, root, where, inlining)


def _make_object_strict(
    node: dict[str, Any], root: dict[str, Any], where: str, inlining: frozenset[str]
) -> None:
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
        _make_strict(prop, root, f"{where}.{name}", inlining)
        if name not in required and not _schema_nullable(prop, root):
            properties[name] = {"anyOf": [prop, {"type": "null"}]}
            optional.append(name)
    node["required"] = list(properties)
    node["additionalProperties"] = False
    if optional:
        node[_OPTIONAL_MARKER] = optional


def _schema_nullable(node: Any, root: dict[str, Any], seen: frozenset[str] = frozenset()) -> bool:
    if not isinstance(node, dict):
        return False
    if "$ref" in node:
        ref = node["$ref"]
        return ref not in seen and _schema_nullable(_resolve_ref(root, ref), root, seen | {ref})
    types = node.get("type")
    if types == "null" or (isinstance(types, list) and "null" in types):
        return True
    if "const" in node and node["const"] is None:
        return True
    if None in node.get("enum", ()):
        return True
    return any(_schema_nullable(branch, root, seen) for branch in node.get("anyOf", ()))


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
    branches = [
        _resolve_ref(root, branch["$ref"]) if "$ref" in branch else branch
        for branch in node.get("anyOf", ())
    ]
    if isinstance(value, dict):
        objects = [branch for branch in branches if "properties" in branch]
        match = _matching_object_branch(objects, value, root)
        if match is not None:
            return _drop_optional_nulls(value, match, root)
    if isinstance(value, list):
        for branch in branches:
            if "items" in branch:
                return _drop_optional_nulls(value, branch, root)
    return value


def _matching_object_branch(
    branches: list[dict[str, Any]], value: dict[str, Any], root: dict[str, Any]
) -> dict[str, Any] | None:
    """Pick the union member whose keys and constants fit ``value``."""

    def fits(branch: dict[str, Any]) -> bool:
        properties = branch["properties"]
        if not set(value) <= set(properties):
            return False
        for key, item in value.items():
            field = properties[key]
            field = _resolve_ref(root, field["$ref"]) if "$ref" in field else field
            if "const" in field and item != field["const"]:
                return False
            if "enum" in field and item not in field["enum"]:
                return False
        return True

    exact = [b for b in branches if fits(b) and set(b["properties"]) == set(value)]
    return next(iter(exact or [b for b in branches if fits(b)] or branches), None)


def _utf8_encodable(text: str) -> bool:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _dotted_keys(config: dict[str, Any], prefix: str = "") -> set[str]:
    keys: set[str] = set()
    for key, value in config.items():
        name = f"{prefix}{key}"
        keys.add(name)
        if isinstance(value, dict):
            keys |= _dotted_keys(value, f"{name}.")
    return keys


def _override_keys(config: Any) -> set[str]:
    """Keys the caller sets through ``config_overrides``."""

    return {
        str(item).partition("=")[0].strip()
        for item in _config_value(config, "config_overrides") or ()
    }


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


def _account_problem(response: Any, cli_login: str) -> str | None:
    """Explain why the runtime's active credentials violate ``cli_login``."""

    account = getattr(getattr(response, "account", None), "root", None)
    kind = getattr(account, "type", None)
    if cli_login == "require":
        if kind == "chatgpt":
            return None
        return "cli_login='require' needs a stored ChatGPT login; run `codex login`"
    if kind in ("apiKey", "amazonBedrock") or not getattr(response, "requires_openai_auth", True):
        return None
    return (
        "no OpenAI credentials: set OPENAI_API_KEY; "
        "cli_login='deny' never uses a stored Codex login"
    )


def _codex_process(codex: Any) -> Any:
    return getattr(getattr(getattr(codex, "_client", None), "_sync", None), "_proc", None)


def _let_exit(process: Any) -> None:
    """Close the app-server's stdin and wait for it to exit, which stops its commands.

    The SDK's close sends SIGTERM right after closing stdin; the app-server dies at once
    and commands it started keep running.
    """

    if process is None or process.poll() is not None:
        return
    with suppress(OSError, ValueError):
        process.stdin.close()
    with suppress(subprocess.TimeoutExpired):
        process.wait(_SHUTDOWN_GRACE_S)


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
    if not isinstance(returncode, int):
        return
    # -signum from Popen, or 128 + signum from a shell or launcher that relays the death.
    if returncode < 0:
        number = -returncode
    elif 128 < returncode < 160:
        number = returncode - 128
    else:
        return
    try:
        name = signal.Signals(number).name
    except ValueError:
        name = str(number)
    raise ProcessTerminatedError(
        number, message=f"Codex app-server was killed by signal {name}: {exc}", cause=exc
    ) from exc


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


def _tool_events(root: Any, event: Any, include_raw: bool) -> tuple[ToolCall, ToolResult] | None:
    """Map an action item to a tool call and result, both carrying the tool name."""

    events = _build_tool_events(root, event, include_raw)
    if not events:
        return None
    call, result = cast(tuple[ToolCall, ToolResult], tuple(events))
    if result.name is None:
        result.name = call.name
    return call, result


def _build_tool_events(root: Any, event: Any, include_raw: bool) -> list[AgentEvent]:
    root_type = getattr(root, "type", "")
    if not root_type:
        return []
    if root_type == "commandExecution":
        item_id = getattr(root, "id", None)
        command = getattr(root, "command", None)
        status = getattr(root, "status", None)
        output = getattr(root, "aggregated_output", None) or ""
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
        item_id = getattr(root, "id", None)
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
                output=_stringify_output(_to_plain(getattr(root, "changes", []))),
                is_error=status in {"failed", "declined"},
                raw=_raw(root) if include_raw else None,
            ),
        ]
    if root_type == "mcpToolCall":
        item_id = getattr(root, "id", None)
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
        item_id = getattr(root, "id", None)
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
        item_id = getattr(root, "id", None)
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
        item_id = getattr(root, "id", None)
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
        item_id = getattr(root, "id", None)
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
        item_id = getattr(root, "id", None)
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
    return json.dumps(plain, ensure_ascii=False, separators=(",", ":"))


def _plan_text(plan: list[Any]) -> str:
    """Render a Codex plan as the checklist TypeScript emits for todo lists."""

    lines = []
    for step in plan:
        done = _status_value(_field(step, "status", "status")) == "completed"
        lines.append(f"- [{'x' if done else ' '}] {_field(step, 'step', 'step')}")
    return "\n".join(lines)


def _parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


_CODEX_ERROR_TYPES = {
    "contextWindowExceeded": "context_window_exceeded",
    "sessionBudgetExceeded": "max_budget",
    "usageLimitExceeded": "usage_limit_exceeded",
    "rateLimitExceeded": TRANSIENT,
    "serverOverloaded": TRANSIENT,
    "internalServerError": TRANSIENT,
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
def _error_event(error: Any, raw: dict[str, Any] | None = None) -> Error:
    message = _turn_error_text(error)
    info = _field(error, "codex_error_info", "codexErrorInfo")
    error_type = _classify_codex_error(info, message)
    return Error(message=message, error_type=error_type, raw=raw)


def _turn_error_text(error: Any) -> str:
    """Keep the runtime's own text; an empty message can leave it only in the details."""

    message = str(_field(error, "message", "message") or "")
    details = str(_field(error, "additional_details", "additionalDetails") or "")
    if message and details and details not in message:
        return f"{message}: {details}"
    return message or details or "Codex reported an error"


def _field(value: Any, attr: str, key: str) -> Any:
    """Read a field from an SDK model, or from the raw dict of an unparsed payload."""

    if isinstance(value, dict):
        return value.get(key, value.get(attr))
    params = getattr(value, "params", None)
    if isinstance(params, dict) and not hasattr(value, attr):
        return params.get(key, params.get(attr))
    return getattr(value, attr, None)


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
            return classify(message, status or None) or TRANSIENT
    return classify(message) or "provider_exception"


def _status_value(status: Any) -> str:
    return str(getattr(status, "value", status) or "")


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
