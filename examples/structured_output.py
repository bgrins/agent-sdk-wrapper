"""Force a typed, validated output via a Pydantic schema.

    uv run python examples/structured_output.py
"""

from __future__ import annotations

import asyncio

from _shared import ExampleOutput, example_artifacts_dir, example_config
from pydantic import BaseModel

from agent_sdk_wrapper import Agent

DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODEL = "claude-haiku-4-5"


class Weather(BaseModel):
    city: str
    temperature_c: float
    conditions: str


async def main() -> None:
    provider, model = example_config(DEFAULT_PROVIDER, DEFAULT_MODEL)
    artifacts_dir = example_artifacts_dir("structured_output", provider)
    out = ExampleOutput(artifacts_dir)
    try:
        agent = Agent(
            provider=provider,
            model=model,
            output_schema=Weather,
            system_prompt="Make something up — this is a demo.",
            artifacts_dir=artifacts_dir,
        )
        result = await agent.run("Give me weather for Portland, OR.")
        out.print("status:", result.status.value)
        if result.error:
            out.print("error:", result.error)
        out.print("final_text:", result.final_text[:120])
        out.print("structured:", result.structured_output)
        out.print("artifacts:", result.artifacts_dir)
        if isinstance(result.structured_output, Weather):
            out.print(
                f"  -> {result.structured_output.city}: "
                f"{result.structured_output.temperature_c}C"
            )
    finally:
        out.save()


if __name__ == "__main__":
    asyncio.run(main())
