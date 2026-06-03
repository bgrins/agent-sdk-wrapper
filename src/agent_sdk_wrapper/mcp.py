"""Shared MCP server configuration for provider adapters."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

type McpToolApprovalMode = Literal["auto", "prompt", "approve"]


@dataclass(kw_only=True)
class McpServerBase:
    """Common options for an MCP server exposed to an agent run."""

    name: str
    enabled_tools: list[str] | None = None
    disabled_tools: list[str] = field(default_factory=list)
    default_tools_approval_mode: McpToolApprovalMode | None = None
    tool_approval_modes: dict[str, McpToolApprovalMode] = field(default_factory=dict)
    required: bool | None = None
    enabled: bool | None = None
    startup_timeout_sec: float | None = None
    tool_timeout_sec: float | None = None


@dataclass(kw_only=True)
class McpStdioServer(McpServerBase):
    """A stdio MCP server launched by the provider runtime."""

    command: str
    args: list[str] = field(default_factory=list)
    cwd: str | Path | None = None
    env: dict[str, str] = field(default_factory=dict)
    env_passthrough: list[str] = field(default_factory=list)


@dataclass(kw_only=True)
class McpHttpServer(McpServerBase):
    """A streamable HTTP MCP server."""

    url: str
    headers: dict[str, str] = field(default_factory=dict)
    env_http_headers: dict[str, str] = field(default_factory=dict)
    bearer_token_env_var: str | None = None


type McpServer = McpStdioServer | McpHttpServer


def stdio_server_env(server: McpStdioServer) -> dict[str, str]:
    """Return explicit stdio MCP env with selected parent variables inherited."""

    env = {
        name: os.environ[name]
        for name in server.env_passthrough
        if name in os.environ
    }
    env.update(server.env)
    return env
