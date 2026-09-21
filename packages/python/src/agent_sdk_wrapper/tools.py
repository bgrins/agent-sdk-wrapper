"""Expose typed Python callables through MCP.

Claude uses an in-process server; Codex uses a temporary stdio server.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import typing
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, create_model
from pydantic.fields import FieldInfo

from .errors import ConfigError
from .events import _jsonable

ANTHROPIC_TOOL_SERVER = "agent_sdk_wrapper_tools"
CODEX_TOOL_SERVER = "agent_sdk_wrapper_tools"
_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def tool_name(fn: Callable[..., Any]) -> str:
    return getattr(fn, "__name__", "tool")


def validate_tool_names(callables: list[Callable[..., Any]]) -> None:
    """Reject names the provider APIs refuse and duplicates they would shadow."""

    seen: set[str] = set()
    for fn in callables:
        name = tool_name(fn)
        if not _TOOL_NAME_RE.fullmatch(name):
            raise ConfigError(
                f"tool name {name!r} must be 1-64 letters, digits, '_' or '-'"
            )
        if name in seen:
            raise ConfigError(f"duplicate tool name {name!r}")
        seen.add(name)


def tool_description(fn: Callable[..., Any]) -> str:
    doc = inspect.getdoc(fn) or ""
    first = doc.strip().split("\n\n", 1)[0].strip()
    return first or tool_name(fn)


def _parameters(fn: Callable[..., Any]) -> list[inspect.Parameter]:
    params = []
    for name, param in inspect.signature(fn).parameters.items():
        if name in ("self", "cls") or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param.kind is param.POSITIONAL_ONLY:
            # Tool arguments arrive by name, so every call would fail.
            raise ConfigError(
                f"tool {tool_name(fn)!r} has positional-only parameter {name!r}; "
                "tool arguments are passed by keyword"
            )
        params.append(param)
    return params


def _annotations(fn: Callable[..., Any], params: list[inspect.Parameter]) -> dict[str, Any]:
    try:
        return typing.get_type_hints(fn, include_extras=True)
    except Exception:
        pass
    # Resolve each hint on its own so one unresolvable name only loses that parameter.
    namespace = getattr(fn, "__globals__", {})
    hints: dict[str, Any] = {}
    for param in params:
        annotation = param.annotation
        if isinstance(annotation, str):
            try:
                annotation = eval(annotation, namespace)  # noqa: S307
            except Exception:
                annotation = Any
        hints[param.name] = annotation
    return hints


def _schema_type(annotation: Any) -> Any:
    """Keep types JSON Schema can describe; pass anything else through unvalidated."""

    if annotation is inspect.Parameter.empty:
        return Any
    try:
        TypeAdapter(annotation).json_schema()
    except Exception:
        return Any
    return annotation


def _arguments_model(fn: Callable[..., Any]) -> type[BaseModel]:
    params = _parameters(fn)
    hints = _annotations(fn, params)
    # Positional field names with aliases accept any parameter name, including "_x".
    fields: dict[str, Any] = {}
    for index, param in enumerate(params):
        annotation = _schema_type(hints.get(param.name, param.annotation))
        if isinstance(param.default, FieldInfo):
            field = FieldInfo.merge_field_infos(param.default, Field(alias=param.name))
        else:
            default = ... if param.default is inspect.Parameter.empty else param.default
            field = Field(default, alias=param.name)
        fields[f"p{index}"] = (annotation, field)
    # A **kwargs tool receives the keys it doesn't name, as calling it directly would.
    extra = "allow" if _takes_extra_keywords(fn) else "ignore"
    return create_model(
        f"{tool_name(fn)}_arguments",
        __config__=ConfigDict(arbitrary_types_allowed=True, extra=extra),
        **fields,
    )


def _takes_extra_keywords(fn: Callable[..., Any]) -> bool:
    return any(
        param.kind is param.VAR_KEYWORD for param in inspect.signature(fn).parameters.values()
    )


# Keys whose values are data, not schemas; their "title" entries are content.
_SCHEMA_DATA_KEYS = frozenset({"default", "const", "enum", "examples"})


def _strip_titles(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: item if key in _SCHEMA_DATA_KEYS else _strip_titles(item)
            for key, item in value.items()
            if not (key == "title" and isinstance(item, str))
        }
    if isinstance(value, list):
        return [_strip_titles(item) for item in value]
    return value


def json_schema_for(fn: Callable[..., Any]) -> dict[str, Any]:
    """Build a JSON Schema object for a callable's parameters."""

    try:
        schema = _arguments_model(fn).model_json_schema()
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(
            f"cannot derive an input schema for tool {tool_name(fn)!r}: {exc}", cause=exc
        ) from exc
    schema = _strip_titles(schema)
    schema.setdefault("properties", {})
    return schema


def tool_caller(fn: Callable[..., Any]) -> Callable[[dict[str, Any]], Awaitable[tuple[str, bool]]]:
    """Validate arguments against ``json_schema_for``'s model, then call ``fn``.

    The returned coroutine gives ``(text, is_error)``. Both providers' tool servers
    use it, so a tool accepts the same arguments and reports the same text on each.
    """

    model = _arguments_model(fn)
    names = [param.name for param in _parameters(fn)]

    async def call(args: dict[str, Any]) -> tuple[str, bool]:
        try:
            validated = model.model_validate(args)
        except ValidationError as exc:
            return f"Error: invalid arguments: {exc}", True
        kwargs = {name: getattr(validated, f"p{i}") for i, name in enumerate(names)}
        kwargs.update(validated.model_extra or {})
        try:
            if inspect.iscoroutinefunction(fn):
                result = await fn(**kwargs)
            else:
                result = await asyncio.to_thread(fn, **kwargs)
                if inspect.isawaitable(result):
                    result = await result
            text = result if isinstance(result, str) else json.dumps(_jsonable(result))
        except Exception as exc:  # surface as a tool error, keep the loop alive
            return f"Error: {exc}", True
        return text, False

    return call


def _make_anthropic_handler(fn: Callable[..., Any]):
    call = tool_caller(fn)

    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        text, is_error = await call(args)
        out: dict[str, Any] = {"content": [{"type": "text", "text": text}]}
        if is_error:
            out["is_error"] = True
        return out

    return handler


def to_anthropic_tools(callables: list[Callable[..., Any]]):
    """Return ``(mcp_server_config | None, allowed_tool_names)``."""
    if not callables:
        return None, []
    from claude_agent_sdk import create_sdk_mcp_server, tool

    sdk_tools = []
    allowed: list[str] = []
    for fn in callables:
        name = tool_name(fn)
        sdk_tool = tool(name, tool_description(fn), json_schema_for(fn))(
            _make_anthropic_handler(fn)
        )
        sdk_tools.append(sdk_tool)
        allowed.append(f"mcp__{ANTHROPIC_TOOL_SERVER}__{name}")
    server = create_sdk_mcp_server(name=ANTHROPIC_TOOL_SERVER, version="1.0.0", tools=sdk_tools)
    return server, allowed
