"""Normalize plain Python callables into backend tool representations.

Callers pass ordinary functions (sync or async) with type hints and a
docstring. Anthropic tools are exposed through an in-process MCP server via
``create_sdk_mcp_server``; Codex tools are exposed through a temporary stdio MCP
server configured for the Codex runtime.
"""

from __future__ import annotations

import inspect
import json
import types
import typing
from collections.abc import Callable
from typing import Any

from .events import _jsonable

ANTHROPIC_TOOL_SERVER = "agent_sdk_wrapper_tools"
CODEX_TOOL_SERVER = "agent_sdk_wrapper_tools"
TOOL_DESCRIPTION_ATTR = "__agent_sdk_wrapper_tool_description__"
TOOL_NAME_ATTR = "__agent_sdk_wrapper_tool_name__"


def tool_name(fn: Callable[..., Any]) -> str:
    return getattr(fn, TOOL_NAME_ATTR, None) or getattr(fn, "__name__", "tool")


def tool_description(fn: Callable[..., Any]) -> str:
    override = getattr(fn, TOOL_DESCRIPTION_ATTR, None)
    if override:
        return override
    doc = inspect.getdoc(fn) or ""
    first = doc.strip().split("\n\n", 1)[0].strip()
    return first or tool_name(fn)


def _py_type_to_schema(annotation: Any) -> dict[str, Any]:
    origin = typing.get_origin(annotation)
    if origin is typing.Union or origin is types.UnionType:  # Optional[X] / X | None
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return _py_type_to_schema(args[0])
        return {}
    if origin in (list, tuple, set):
        return {"type": "array"}
    if origin is dict:
        return {"type": "object"}
    mapping = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
    }
    return mapping.get(annotation, {"type": "string"})


def json_schema_for(fn: Callable[..., Any]) -> dict[str, Any]:
    """Build a JSON Schema object for a callable's parameters."""
    sig = inspect.signature(fn)
    try:
        hints = typing.get_type_hints(fn)
    except Exception:
        hints = {}
    props: dict[str, Any] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        props[name] = _py_type_to_schema(hints.get(name, str))
        if param.default is inspect.Parameter.empty:
            required.append(name)
    schema: dict[str, Any] = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return schema


def _make_anthropic_handler(fn: Callable[..., Any]):
    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        try:
            result = fn(**args)
            if inspect.isawaitable(result):
                result = await result
            text = result if isinstance(result, str) else json.dumps(_jsonable(result))
            return {"content": [{"type": "text", "text": text}]}
        except Exception as exc:  # surface as a tool error, keep the loop alive
            return {
                "content": [{"type": "text", "text": f"Error: {exc}"}],
                "is_error": True,
            }

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
