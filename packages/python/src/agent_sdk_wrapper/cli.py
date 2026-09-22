"""Command-line interface for ``Agent``."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import re
import sys
import tomllib
from pathlib import Path
from typing import Any, Literal, get_type_hints

from pydantic import TypeAdapter, ValidationError

from . import (
    Agent,
    ConfigError,
    Error,
    McpHttpServer,
    McpServer,
    McpStdioServer,
    ProcessTerminatedError,
    RunFinished,
    RunResult,
    RunStatus,
    SubagentDef,
    Text,
    __version__,
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
    run.add_argument(
        "--stream",
        action="store_true",
        help="Print each completed assistant message as it arrives.",
    )
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
        choices=["none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"],
        help="Reasoning effort tier. Unsupported provider/tier combinations are rejected.",
    )
    run.add_argument("--timeout", default=None, type=float)
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
        help="Enable WebSearch/WebFetch (Anthropic) or live web_search (Codex).",
    )
    web_tools.add_argument(
        "--no-web-tools",
        dest="web_tools",
        action="store_const",
        const=False,
        help="Disable WebSearch/WebFetch (Anthropic) or web_search (Codex).",
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
        help="Disallow a tool (anthropic). Repeatable.",
    )
    run.add_argument(
        "--session-id",
        default=None,
        help="Resume an existing provider session/thread.",
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
    run.add_argument(
        "--setting-source",
        action="append",
        default=None,
        choices=["user", "project", "local"],
        help="Claude on-disk settings to load. Repeatable; default none.",
    )
    run.add_argument(
        "--cli-login",
        default=None,
        choices=["deny", "require"],
        help="Whether the runtime may use its stored login (require: Codex only).",
    )
    run.add_argument("--verbose", "-v", action="count", default=0)
    return p


def _read_prompt(args: argparse.Namespace, config: dict[str, Any]) -> str:
    if args.prompt is not None and args.prompt_file is not None:
        raise ConfigError("pass only one of --prompt / --prompt-file")
    if args.prompt is not None:
        return args.prompt
    if args.prompt_file is not None:
        return _read_prompt_file(args.prompt_file)
    if "prompt" in config and "prompt_file" in config:
        raise ConfigError("config may contain only one of prompt or prompt_file")
    if "prompt" in config:
        return config["prompt"]
    if "prompt_file" in config:
        return _read_prompt_file(config["prompt_file"])
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise ConfigError("provide --prompt, --prompt-file, or pipe one on stdin")


def _read_prompt_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"could not read prompt file {path}: {exc}") from exc


def _setup_logging(verbose: int) -> None:
    level = logging.WARNING if verbose == 0 else logging.INFO if verbose == 1 else logging.DEBUG
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)


async def _run(args: argparse.Namespace) -> int:
    config = _load_cli_config(args.config)
    prompt = _read_prompt(args, config)
    stream = args.stream or config.get("stream", False)
    output = args.output or config.get("output") or ("text" if stream else "jsonl")
    if stream and output != "text":
        raise ConfigError(f"--stream cannot be combined with --output {output}")
    if args.no_builtin_tools and args.builtin_tool:
        raise ConfigError("--no-builtin-tools cannot be combined with --builtin-tool")
    flags = {
        "provider": args.provider,
        "model": args.model,
        "system_prompt": args.system_prompt,
        "cwd": args.cwd,
        "max_turns": args.max_turns,
        "effort": args.effort,
        "timeout": args.timeout,
        "include_raw": args.include_raw,
        "builtin_tools": "none" if args.no_builtin_tools else args.builtin_tool,
        "web_tools": args.web_tools,
        "allowed_tools": args.allowed_tool,
        "disallowed_tools": args.disallowed_tool,
        "session_id": args.session_id,
        "permission_mode": args.permission_mode,
        "cli_login": args.cli_login,
        "setting_sources": args.setting_source,
        "trace_file": args.trace_file,
        "artifacts_dir": args.artifacts_dir,
    }
    agent_kwargs = {key: value for key, value in config.items() if key not in _CLI_HINTS}
    agent_kwargs.update({key: value for key, value in flags.items() if value is not None})
    try:
        agent_kwargs["env"] = {
            **(agent_kwargs.get("env") or {}),
            **_parse_env_assignments(args.env or []),
        }
        for key, flag, values in (
            ("provider_options", "--provider-option", args.provider_option),
            ("extra_options", "--extra-option", args.extra_option),
        ):
            agent_kwargs[key] = _deep_merge(
                agent_kwargs.get(key) or {}, _parse_json_assignments(values or [], flag=flag)
            )
        agent = Agent(**agent_kwargs)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ConfigError(f"invalid settings: {exc}") from exc

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
                # Each Text is a whole assistant message; keep them apart.
                if final_text_parts:
                    sys.stdout.write("\n")
                sys.stdout.write(env.event.text)
                sys.stdout.flush()
                final_text_parts.append(env.event.text)
        if final_text_parts:
            sys.stdout.write("\n")
        return rc

    try:
        result = await agent.run(prompt)
    except ProcessTerminatedError as exc:
        if exc.result is not None:
            _write_result(exc.result, output)
        raise
    _write_result(result, output)
    return 0 if result.ok else 1


def _write_result(result: RunResult, output: str) -> None:
    if output == "text":
        sys.stdout.write(result.final_text)
        if result.final_text and not result.final_text.endswith("\n"):
            sys.stdout.write("\n")
    else:  # json
        json.dump(result.to_dict(), sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")


def _stream_event_failed(event: object) -> bool:
    if isinstance(event, Error):
        return True
    return isinstance(event, RunFinished) and event.status != RunStatus.SUCCESS


_CLI_HINTS: dict[str, Any] = {
    "prompt": str,
    "prompt_file": str,
    "output": Literal["jsonl", "text", "json"],
    "stream": bool,
}
# A config file sets these Agent keywords. Tools, schemas and callbacks need Python
# objects, and one prompt per process leaves nothing to continue.
_AGENT_KEYS = frozenset(inspect.signature(Agent).parameters) - {
    "tools",
    "output_schema",
    "on_event",
    "on_provider_event",
    "continue_session",
}
_CONFIG_HINTS: dict[str, Any] = {
    **{key: hint for key, hint in get_type_hints(Agent.__init__).items() if key in _AGENT_KEYS},
    **_CLI_HINTS,
    # Entries are checked against their dataclass's fields.
    "mcp_servers": list[dict[str, Any]] | None,
    "subagents": dict[str, dict[str, Any]] | None,
}
_PATH_KEYS = ("cwd", "trace_file", "artifacts_dir", "prompt_file")


def _load_cli_config(path: Path | None) -> dict[str, Any]:
    """Read a TOML or JSON file whose keys are Agent keywords plus CLI options."""

    if path is None:
        return {}
    try:
        text = path.read_text(encoding="utf-8")
        config = json.loads(text) if path.suffix.lower() == ".json" else tomllib.loads(text)
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"could not read config file {path}: {exc}") from exc
    except (json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"could not parse config file {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ConfigError("config file must contain an object/table")
    unknown = sorted(set(config) - set(_CONFIG_HINTS))
    if unknown:
        raise ConfigError(f"unknown config field(s): {', '.join(unknown)}")
    _check_types(config, _CONFIG_HINTS)

    # Relative paths are relative to the config file.
    base = path.resolve().parent
    for key in _PATH_KEYS:
        if config.get(key) is not None:
            config[key] = base / config[key]
    try:
        if config.get("mcp_servers") is not None:
            config["mcp_servers"] = [
                _mcp_server(dict(raw), base, f"mcp_servers[{index}].")
                for index, raw in enumerate(config["mcp_servers"])
            ]
        if config.get("subagents") is not None:
            config["subagents"] = {
                name: _dataclass(SubagentDef, spec, f"subagents.{name}.")
                for name, spec in config["subagents"].items()
            }
    except (TypeError, ValueError, AttributeError) as exc:
        raise ConfigError(f"invalid config: {exc}") from exc
    return config


def _mcp_server(raw: dict[str, Any], base: Path, where: str) -> McpServer:
    server_type = raw.pop("type", "http" if "url" in raw else "stdio")
    if server_type not in ("stdio", "http"):
        raise ConfigError("mcp_servers type must be 'stdio' or 'http'")
    cls = McpStdioServer if server_type == "stdio" else McpHttpServer
    server = _dataclass(cls, raw, where)
    if isinstance(server, McpStdioServer) and server.cwd is not None:
        server.cwd = base / server.cwd
    return server


def _dataclass[T](cls: type[T], raw: dict[str, Any], where: str) -> T:
    hints = get_type_hints(cls)
    _check_types({key: value for key, value in raw.items() if key in hints}, hints, where)
    return cls(**raw)


def _check_types(values: dict[str, Any], hints: dict[str, Any], where: str = "") -> None:
    """Reject values that don't match their hint exactly; Agent would read "false" as true."""

    for key, value in values.items():
        try:
            TypeAdapter(hints[key]).validate_python(value, strict=True)
        except ValidationError as exc:
            errors = exc.errors()
            if len(errors) > 1:  # one per member of a union
                raise ConfigError(
                    f"config field {where}{key} must be {_type_name(hints[key])}"
                ) from None
            loc = "".join(f"[{part}]" if isinstance(part, int) else f".{part}"
                          for part in errors[0]["loc"])
            raise ConfigError(f"config field {where}{key}{loc}: {errors[0]['msg']}") from None


def _type_name(hint: Any) -> str:
    if isinstance(hint, type):
        return hint.__name__
    return re.sub(r"\b[a-z_][\w.]*\.(?=[A-Z])", "", str(hint))


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
    # JSON output must be UTF-8. Lone surrogates in provider text would otherwise
    # abort output; the escapes this writes decode back to the same string in JSON.
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="backslashreplace")
    if args.command == "run":
        try:
            return asyncio.run(_run(args))
        except ConfigError as exc:
            sys.stderr.write(f"error: {exc}\n")
            return 2
        except ProcessTerminatedError as exc:
            sys.stderr.write(f"error: {exc}\n")
            return 128 + exc.signal
        except KeyboardInterrupt:
            return 130
    return 1


if __name__ == "__main__":
    sys.exit(main())
