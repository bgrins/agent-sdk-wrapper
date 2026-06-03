"""Read-only MCP tools for the auditor-style example."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

server = FastMCP("auditor_demo")


@server.tool(structured_output=False)
async def read_project_brief() -> str:
    """Return project facts used by the auditor-style demo."""

    return (
        "agent-sdk-wrapper is a small provider wrapper with Docker Compose "
        "example workflows, timestamped artifacts under results/, trace replay "
        "fixtures, and a static trace viewer."
    )


@server.tool(structured_output=False)
async def read_artifact_policy() -> str:
    """Return artifact expectations for the auditor-style demo."""

    return (
        "Committed traces are normalized JSONL fixtures. Raw provider events "
        "belong in local results/ artifacts and live fixture promotion redacts "
        "secret-looking API keys, bearer tokens, and auth headers."
    )


if __name__ == "__main__":
    server.run("stdio")
