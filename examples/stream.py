"""Stream events from an agent run.

    uv run python examples/stream.py
"""

from __future__ import annotations

import asyncio

from _shared import ExampleOutput, example_artifacts_dir, example_config

from agent_sdk_wrapper import Agent, Text, ToolCall, ToolResult

DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODEL = "claude-haiku-4-5"


async def main() -> None:
    provider, model = example_config(DEFAULT_PROVIDER, DEFAULT_MODEL)
    artifacts_dir = example_artifacts_dir("stream", provider)
    out = ExampleOutput(artifacts_dir)
    try:
        agent = Agent(provider=provider, model=model, artifacts_dir=artifacts_dir)
        async for env in agent.stream("Count from 1 to 5, one per line."):
            ev = env.event
            if isinstance(ev, Text):
                out.write(ev.text)
            elif isinstance(ev, ToolCall):
                out.print(f"\n[tool_call] {ev.name}({ev.input})")
            elif isinstance(ev, ToolResult):
                out.print(f"\n[tool_result] {ev.output!r}")
        out.print()
        out.print(f"artifacts: {artifacts_dir}")
    finally:
        out.save()


if __name__ == "__main__":
    asyncio.run(main())
