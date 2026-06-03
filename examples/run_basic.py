"""Run a single prompt and print the final text + usage.

    uv run python examples/run_basic.py
"""

from __future__ import annotations

import asyncio

from _shared import ExampleOutput, example_artifacts_dir, example_config

from agent_sdk_wrapper import Agent

DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODEL = "claude-haiku-4-5"


async def main() -> None:
    provider, model = example_config(DEFAULT_PROVIDER, DEFAULT_MODEL)
    artifacts_dir = example_artifacts_dir("run_basic", provider)
    out = ExampleOutput(artifacts_dir)
    try:
        agent = Agent(
            provider=provider,
            model=model,
            system_prompt="Be concise.",
            artifacts_dir=artifacts_dir,
        )
        result = await agent.run("Say hello in 5 words.")
        out.print(f"status      : {result.status.value}")
        out.print(f"duration_ms : {result.duration_ms}")
        out.print(f"usage       : {result.usage}")
        out.print(f"cost_usd    : {result.cost_usd}")
        out.print(f"session_id  : {result.session_id}")
        out.print(f"artifacts   : {result.artifacts_dir}")
        out.print("---")
        out.print(result.final_text)
    finally:
        out.save()


if __name__ == "__main__":
    asyncio.run(main())
