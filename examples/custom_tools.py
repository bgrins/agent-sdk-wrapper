"""Expose Python functions as tools to the agent.

The callables are wrapped as an in-process MCP server for Anthropic and as a
temporary stdio MCP server for Codex.

    uv run python examples/custom_tools.py
"""

from __future__ import annotations

import asyncio

from _shared import ExampleOutput, example_artifacts_dir, example_config

from agent_sdk_wrapper import Agent

DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODEL = "claude-haiku-4-5"


def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def square(n: int) -> int:
    """Square an integer."""
    return n * n


async def main() -> None:
    provider, model = example_config(DEFAULT_PROVIDER, DEFAULT_MODEL)
    artifacts_dir = example_artifacts_dir("custom_tools", provider)
    out = ExampleOutput(artifacts_dir)
    try:
        agent = Agent(
            provider=provider,
            model=model,
            tools=[add, square],
            system_prompt="Use the provided tools to compute. Show your work briefly.",
            artifacts_dir=artifacts_dir,
        )
        result = await agent.run("What is (3 + 4) squared? Use the tools.")
        out.print(result.final_text)
        out.print(f"artifacts: {result.artifacts_dir}")
        out.print("---")
        for tc in result.tool_calls():
            out.print(f"called: {tc.name}({tc.input})")
    finally:
        out.save()


if __name__ == "__main__":
    asyncio.run(main())
