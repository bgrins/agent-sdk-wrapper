"""Tiny external MCP server used by live integration tests."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

server = FastMCP("brief_tools")


@server.tool(structured_output=False)
async def read_brief() -> str:
    """Return a deterministic integration-test brief."""

    return "LIVE_MCP_OK: external MCP server returned the release-readiness brief."


if __name__ == "__main__":
    server.run("stdio")
