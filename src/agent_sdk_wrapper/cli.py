"""``agent-sdk-wrapper`` — a thin CLI over the unified :class:`Agent`."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import tomllib
from pathlib import Path
from typing import Any

from . import (
    Agent,
    ConfigError,
    Error,
    McpHttpServer,
    McpServer,
    McpStdioServer,
    RunFinished,
    RunStatus,
    Text,
    __version__,
    normalize_builtin_tools,
)
from .logging import LOGGER_NAME


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agent-sdk-wrapper")
    p.add_argument("--version", action="version", version=f"agent-sdk-wrapper {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run a prompt against a provider.")
    run.add_argument(
        "--config",
        default=None,
        type=Path,
        help="Load run defaults from a TOML or JSON config file.",
    )
    run.add_argument(
        "--provider",
        default=None,
        choices=["anthropic", "openai", "codex"],
        help="Provider to use. Optional when --model identifies the provider.",
    )
    run.add_argument("--model", default=None)
    run.add_argument("--prompt", default=None)
    run.add_argument("--prompt-file", default=None, type=Path)
    run.add_argument("--system-prompt", default=None)
    run.add_argument(
        "--output",
        default=None,
        choices=["jsonl", "text", "json"],
        help=(
            "jsonl: stream envelopes; text: final text only; json: RunResult as JSON. "
            "Defaults to jsonl, or text when --stream is set."
        ),
    )
    run.add_argument("--stream", action="store_true", help="Stream text deltas to stdout.")
    run.add_argument("--trace-file", default=None, type=Path)
    run.add_argument(
        "--artifacts-dir",
        default=None,
        type=Path,
        help="Write trace.jsonl, manifest.json, and provider artifacts to this directory.",
    )
    run.add_argument("--cwd", default=None, type=Path)
    run.add_argument("--max-turns", default=None, type=int)
    run.add_argument(
        "--effort",
        default=None,
        choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
        help="Reasoning effort tier. Unsupported provider/tier combinations are rejected.",
    )
    run.add_argument("--timeout", default=None, type=float)
    run.add_argument("--max-retries", default=None, type=int)
    run.add_argument("--include-raw", action="store_true", default=None)
    run.add_argument(
        "--builtin-tool",
        action="append",
        default=None,
        help=(
            "Provider built-in tool allowlist. Repeatable. Anthropic only; "
            "Codex rejects built-in tool controls."
        ),
    )
    run.add_argument(
        "--no-builtin-tools",
        action="store_true",
        default=None,
        help="Request no provider built-in tools where the provider can enforce it.",
    )
    web_tools = run.add_mutually_exclusive_group()
    web_tools.add_argument(
        "--web-tools",
        dest="web_tools",
        action="store_const",
        const=True,
        default=None,
        help="Enable WebSearch/WebFetch (Anthropic) or tools.web_search (Codex).",
    )
    web_tools.add_argument(
        "--no-web-tools",
        dest="web_tools",
        action="store_const",
        const=False,
        help="Disable WebSearch/WebFetch (Anthropic) or tools.web_search (Codex).",
    )
    run.add_argument(
        "--allowed-tool",
        action="append",
        default=None,
        help="Allow a builtin tool (anthropic). Repeatable.",
    )
    run.add_argument(
        "--disallowed-tool",
        action="append",
        default=None,
        help="Disallow a builtin, callable, or MCP tool where supported. Repeatable.",
    )
    run.add_argument(
        "--session-id",
        default=None,
        help="Resume an existing provider session/thread.",
    )
    run.add_argument(
        "--continue-session",
        action="store_true",
        default=None,
        help="Store emitted session ids and continue the same session for later calls.",
    )
    run.add_argument(
        "--env",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Add an environment variable for the provider runtime. Repeatable.",
    )
    run.add_argument(
        "--provider-option",
        action="append",
        default=None,
        metavar="KEY=JSON",
        help="Set Agent provider_options using JSON values. Dotted keys build nested objects.",
    )
    run.add_argument(
        "--extra-option",
        action="append",
        default=None,
        metavar="KEY=JSON",
        help="Set RunRequest extra_options using JSON values. Dotted keys build nested objects.",
    )
    run.add_argument(
        "--permission-mode",
        default=None,
        choices=["default", "acceptEdits", "plan", "bypassPermissions", "dontAsk"],
    )
    run.add_argument("--verbose", "-v", action="count", default=0)
    return p


def _read_prompt(args: argparse.Namespace, config: dict[str, Any], base_dir: Path | None) -> str:
    if args.prompt and args.prompt_file:
        sys.exit("error: pass only one of --prompt / --prompt-file")
    if args.prompt is not None:
        return args.prompt
    if args.prompt_file is not None:
        return args.prompt_file.read_text(encoding="utf-8")
    if "prompt" in config and "prompt_file" in config:
        raise ConfigError("config may contain only one of prompt or prompt_file")
    if "prompt" in config:
        prompt = config["prompt"]
        if not isinstance(prompt, str):
            raise ConfigError("config field prompt must be a string")
        return prompt
    if "prompt_file" in config:
        prompt_file = _path_from_config(config["prompt_file"], base_dir, field="prompt_file")
        return prompt_file.read_text(encoding="utf-8")
    if not sys.stdin.isatty():
        return sys.stdin.read()
    sys.exit("error: provide --prompt, --prompt-file, or pipe one on stdin")


def _setup_logging(verbose: int) -> None:
    level = logging.WARNING if verbose == 0 else logging.INFO if verbose == 1 else logging.DEBUG
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)


async def _run(args: argparse.Namespace) -> int:
    config, config_base_dir = _load_cli_config(args.config)
    prompt = _read_prompt(args, config, config_base_dir)
    stream = args.stream or _config_bool(config, "stream", False)
    output = args.output or _config_get(config, "output", None) or ("text" if stream else "jsonl")
    if output not in _OUTPUTS:
        raise ConfigError("config field output must be one of: jsonl, text, json")
    if stream and output == "jsonl":
        raise ConfigError("--stream cannot be combined with --output jsonl")
    env = {
        **_config_str_dict(config, "env"),
        **_parse_env_assignments(args.env or []),
    }
    provider_options = _deep_merge(
        _config_mapping(config, "provider_options"),
        _parse_json_assignments(args.provider_option or [], flag="--provider-option"),
    )
    extra_options = _deep_merge(
        _config_mapping(config, "extra_options"),
        _parse_json_assignments(args.extra_option or [], flag="--extra-option"),
    )
    agent_kwargs: dict[str, Any] = dict(
        provider=args.provider or _config_get(config, "provider", None),
        model=args.model or _config_get(config, "model", None),
        system_prompt=args.system_prompt or _config_get(config, "system_prompt", None),
        cwd=args.cwd or _optional_path(config, "cwd", config_base_dir),
        env=env,
        max_turns=(
            args.max_turns
            if args.max_turns is not None
            else _config_get(config, "max_turns", None)
        ),
        effort=args.effort or _config_get(config, "effort", None),
        timeout=args.timeout if args.timeout is not None else _config_get(config, "timeout", None),
        max_retries=args.max_retries
        if args.max_retries is not None
        else _config_int(config, "max_retries", 2),
        include_raw=bool(
            args.include_raw
            if args.include_raw is not None
            else _config_bool(config, "include_raw", False)
        ),
        builtin_tools=_merge_builtin_tools(
            config, args.builtin_tool, args.no_builtin_tools
        ),
        web_tools=(
            args.web_tools
            if args.web_tools is not None
            else _config_optional_bool(config, "web_tools")
        ),
        allowed_tools=_merge_string_lists(config, "allowed_tools", args.allowed_tool),
        disallowed_tools=_merge_string_lists(config, "disallowed_tools", args.disallowed_tool),
        mcp_servers=_parse_mcp_servers(config.get("mcp_servers"), config_base_dir),
        session_id=args.session_id or _config_get(config, "session_id", None),
        continue_session=bool(
            args.continue_session
            if args.continue_session is not None
            else _config_bool(config, "continue_session", False)
        ),
        permission_mode=args.permission_mode or _config_get(config, "permission_mode", None),
        extra_options=extra_options,
        provider_options=provider_options,
        trace_file=args.trace_file or _optional_path(config, "trace_file", config_base_dir),
        artifacts_dir=(
            args.artifacts_dir or _optional_path(config, "artifacts_dir", config_base_dir)
        ),
    )
    agent = Agent(**agent_kwargs)

    if output == "jsonl":
        rc = 0
        async for env in agent.stream(prompt):
            if _stream_event_failed(env.event):
                rc = 1
            sys.stdout.write(env.to_json())
            sys.stdout.write("\n")
            sys.stdout.flush()
        return rc

    if stream:
        final_text_parts: list[str] = []
        rc = 0
        async for env in agent.stream(prompt):
            if _stream_event_failed(env.event):
                rc = 1
            if isinstance(env.event, Text):
                sys.stdout.write(env.event.text)
                sys.stdout.flush()
                final_text_parts.append(env.event.text)
        if final_text_parts:
            sys.stdout.write("\n")
        return rc

    result = await agent.run(prompt)
    if output == "text":
        sys.stdout.write(result.final_text)
        if result.final_text and not result.final_text.endswith("\n"):
            sys.stdout.write("\n")
    else:  # json
        json.dump(result.to_dict(), sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    return 0 if result.ok else 1


def _stream_event_failed(event: object) -> bool:
    if isinstance(event, Error):
        return True
    return isinstance(event, RunFinished) and event.status != RunStatus.SUCCESS


_CONFIG_KEYS = {
    "allowed_tools",
    "artifacts_dir",
    "builtin_tools",
    "continue_session",
    "cwd",
    "disallowed_tools",
    "effort",
    "env",
    "extra_options",
    "include_raw",
    "max_retries",
    "max_turns",
    "mcp_servers",
    "model",
    "output",
    "permission_mode",
    "prompt",
    "prompt_file",
    "provider",
    "provider_options",
    "session_id",
    "stream",
    "system_prompt",
    "timeout",
    "trace_file",
    "web_tools",
}
_OUTPUTS = {"jsonl", "text", "json"}
_MCP_COMMON_KEYS = {
    "default_tools_approval_mode",
    "disabled_tools",
    "enabled",
    "enabled_tools",
    "name",
    "required",
    "startup_timeout_sec",
    "tool_approval_modes",
    "tool_timeout_sec",
    "type",
}
_MCP_STDIO_KEYS = _MCP_COMMON_KEYS | {
    "args",
    "command",
    "cwd",
    "env",
    "env_passthrough",
}
_MCP_HTTP_KEYS = _MCP_COMMON_KEYS | {
    "bearer_token_env_var",
    "env_http_headers",
    "headers",
    "url",
}
_APPROVAL_MODES = {"auto", "prompt", "approve"}


def _load_cli_config(path: Path | None) -> tuple[dict[str, Any], Path | None]:
    if path is None:
        return {}, None
    try:
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            config = json.loads(text)
        else:
            config = tomllib.loads(text)
    except OSError as exc:
        raise ConfigError(f"could not read config file {path}: {exc}") from exc
    except (json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"could not parse config file {path}: {exc}") from exc

    if not isinstance(config, dict):
        raise ConfigError("config file must contain an object/table")
    unknown = sorted(set(config) - _CONFIG_KEYS)
    if unknown:
        raise ConfigError(f"unknown config field(s): {', '.join(unknown)}")
    return config, path.resolve().parent


def _config_get(config: dict[str, Any], key: str, default: Any) -> Any:
    return config.get(key, default)


def _config_bool(config: dict[str, Any], key: str, default: bool) -> bool:
    if key not in config:
        return default
    value = config[key]
    if not isinstance(value, bool):
        raise ConfigError(f"config field {key} must be a boolean")
    return value


def _config_int(config: dict[str, Any], key: str, default: int) -> int:
    if key not in config:
        return default
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"config field {key} must be an integer")
    return value


def _config_optional_bool(config: dict[str, Any], key: str) -> bool | None:
    if key not in config:
        return None
    value = config[key]
    if not isinstance(value, bool):
        raise ConfigError(f"config field {key} must be a boolean")
    return value


def _config_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    if key not in config:
        return {}
    value = config[key]
    if not isinstance(value, dict):
        raise ConfigError(f"config field {key} must be an object/table")
    return dict(value)


def _config_str_dict(config: dict[str, Any], key: str) -> dict[str, str]:
    if key not in config:
        return {}
    return _string_dict(config[key], f"config field {key}")


def _optional_path(config: dict[str, Any], key: str, base_dir: Path | None) -> Path | None:
    if key not in config:
        return None
    return _path_from_config(config[key], base_dir, field=key)


def _path_from_config(value: Any, base_dir: Path | None, *, field: str) -> Path:
    if not isinstance(value, str):
        raise ConfigError(f"config field {field} must be a string path")
    path = Path(value)
    if path.is_absolute() or base_dir is None:
        return path
    return base_dir / path


def _merge_string_lists(
    config: dict[str, Any], key: str, cli_values: list[str] | None
) -> list[str]:
    config_values = _string_list(config[key], f"config field {key}") if key in config else []
    return [*config_values, *(cli_values or [])]


def _merge_builtin_tools(
    config: dict[str, Any],
    cli_values: list[str] | None,
    no_builtin_tools: bool | None,
) -> object:
    if no_builtin_tools and cli_values:
        raise ConfigError("--no-builtin-tools cannot be combined with --builtin-tool")
    if no_builtin_tools:
        return "none"
    if cli_values is not None:
        return cli_values
    if "builtin_tools" in config:
        return normalize_builtin_tools(config["builtin_tools"])
    return None


def _parse_mcp_servers(value: Any, base_dir: Path | None) -> list[McpServer]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ConfigError("config field mcp_servers must be an array")

    servers: list[McpServer] = []
    for index, raw in enumerate(value):
        field = f"mcp_servers[{index}]"
        if not isinstance(raw, dict):
            raise ConfigError(f"{field} must be an object/table")

        server_type = raw.get("type")
        if server_type is None:
            server_type = "http" if "url" in raw else "stdio"
        if server_type == "stdio":
            _reject_unknown_keys(raw, _MCP_STDIO_KEYS, field)
            servers.append(
                McpStdioServer(
                    **_mcp_common_kwargs(raw, field),
                    command=_required_str(raw, "command", field),
                    args=_string_list(raw.get("args", []), f"{field}.args"),
                    cwd=(
                        _path_from_config(raw["cwd"], base_dir, field=f"{field}.cwd")
                        if "cwd" in raw
                        else None
                    ),
                    env=_string_dict(raw.get("env", {}), f"{field}.env"),
                    env_passthrough=_string_list(
                        raw.get("env_passthrough", []),
                        f"{field}.env_passthrough",
                    ),
                )
            )
        elif server_type == "http":
            _reject_unknown_keys(raw, _MCP_HTTP_KEYS, field)
            servers.append(
                McpHttpServer(
                    **_mcp_common_kwargs(raw, field),
                    url=_required_str(raw, "url", field),
                    headers=_string_dict(raw.get("headers", {}), f"{field}.headers"),
                    env_http_headers=_string_dict(
                        raw.get("env_http_headers", {}), f"{field}.env_http_headers"
                    ),
                    bearer_token_env_var=_optional_str(
                        raw, "bearer_token_env_var", field
                    ),
                )
            )
        else:
            raise ConfigError(f"{field}.type must be 'stdio' or 'http'")
    return servers


def _mcp_common_kwargs(raw: dict[str, Any], field: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"name": _required_str(raw, "name", field)}
    if "enabled_tools" in raw:
        kwargs["enabled_tools"] = _string_list(raw["enabled_tools"], f"{field}.enabled_tools")
    if "disabled_tools" in raw:
        kwargs["disabled_tools"] = _string_list(raw["disabled_tools"], f"{field}.disabled_tools")
    if "default_tools_approval_mode" in raw:
        kwargs["default_tools_approval_mode"] = _approval_mode(
            raw["default_tools_approval_mode"], f"{field}.default_tools_approval_mode"
        )
    if "tool_approval_modes" in raw:
        modes = _string_dict(raw["tool_approval_modes"], f"{field}.tool_approval_modes")
        kwargs["tool_approval_modes"] = {
            key: _approval_mode(value, f"{field}.tool_approval_modes.{key}")
            for key, value in modes.items()
        }
    for key in ("required", "enabled"):
        if key in raw:
            kwargs[key] = _optional_bool(raw[key], f"{field}.{key}")
    for key in ("startup_timeout_sec", "tool_timeout_sec"):
        if key in raw:
            kwargs[key] = _optional_float(raw[key], f"{field}.{key}")
    return kwargs


def _reject_unknown_keys(raw: dict[str, Any], allowed: set[str], field: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(f"unknown {field} field(s): {', '.join(unknown)}")


def _required_str(raw: dict[str, Any], key: str, field: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or value == "":
        raise ConfigError(f"{field}.{key} must be a non-empty string")
    return value


def _optional_str(raw: dict[str, Any], key: str, field: str) -> str | None:
    if key not in raw or raw[key] is None:
        return None
    value = raw[key]
    if not isinstance(value, str):
        raise ConfigError(f"{field}.{key} must be a string")
    return value


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"{field} must be an array of strings")
    return list(value)


def _string_dict(value: Any, field: str) -> dict[str, str]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise ConfigError(f"{field} must be an object/table of strings")
    return dict(value)


def _optional_bool(value: Any, field: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ConfigError(f"{field} must be a boolean")
    return value


def _optional_float(value: Any, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{field} must be a number")
    return float(value)


def _approval_mode(value: str, field: str) -> str:
    if value not in _APPROVAL_MODES:
        raise ConfigError(f"{field} must be one of: auto, prompt, approve")
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _parse_env_assignments(values: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for raw in values:
        key, value = _split_assignment(raw, flag="--env")
        env[key] = value
    return env


def _parse_json_assignments(values: list[str], *, flag: str) -> dict[str, Any]:
    options: dict[str, Any] = {}
    for raw in values:
        key, value = _split_assignment(raw, flag=flag)
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ConfigError(
                f"{flag} expects KEY=JSON; could not parse JSON for {key!r}: {exc.msg}"
            ) from exc
        _set_nested_option(options, key, parsed, flag=flag)
    return options


def _split_assignment(raw: str, *, flag: str) -> tuple[str, str]:
    if "=" not in raw:
        raise ConfigError(f"{flag} expects KEY=VALUE")
    key, value = raw.split("=", 1)
    if not key:
        raise ConfigError(f"{flag} key cannot be empty")
    return key, value


def _set_nested_option(target: dict[str, Any], key: str, value: Any, *, flag: str) -> None:
    parts = key.split(".")
    if any(part == "" for part in parts):
        raise ConfigError(f"{flag} key {key!r} contains an empty path segment")

    current = target
    for part in parts[:-1]:
        existing = current.setdefault(part, {})
        if not isinstance(existing, dict):
            raise ConfigError(f"{flag} key {key!r} conflicts with existing value")
        current = existing
    current[parts[-1]] = value


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _setup_logging(args.verbose)
    if args.command == "run":
        try:
            return asyncio.run(_run(args))
        except ConfigError as exc:
            sys.stderr.write(f"error: {exc}\n")
            return 2
        except KeyboardInterrupt:
            return 130
    return 1


if __name__ == "__main__":
    sys.exit(main())
