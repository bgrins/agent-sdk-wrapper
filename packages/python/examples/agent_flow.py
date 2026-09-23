"""Plan, draft, and review a generic task using isolated agent runs.

Run from packages/python: uv run python examples/agent_flow.py
Set PROVIDER=codex for a separate, schema-free exploration turn per stage.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from _shared import example_artifacts_dir, example_config
from pydantic import BaseModel

from agent_sdk_wrapper import Agent


class Plan(BaseModel):
    steps: list[str]


class Draft(BaseModel):
    text: str


class Review(BaseModel):
    approved: bool
    feedback: str


async def run_stage[Output: BaseModel](
    name: str,
    prompt: str,
    schema: type[Output],
    root: Path,
    provider: str,
    model: str | None,
) -> Output:
    options = None
    if provider == "openai":
        options = {
            "sandbox": "read-only",
            "approval_mode": "deny_all",
            "config": {"config_overrides": ["features.shell_tool=false"]},
        }
    agent = Agent(
        provider=provider,
        model=model,
        system_prompt=f"You are the {name}. Work only with the information provided.",
        continue_session=True,
        extra_options={"tools": []} if provider == "anthropic" else None,
        web_tools=False,
        provider_options=options,
    )
    if provider == "openai":
        exploration = await agent.run(
            prompt, artifacts_dir=root / name / "explore", raise_on_error=True
        )
        if not exploration.session_id:
            raise RuntimeError(f"{name} cannot resume the exploration turn")
        prompt = (
            "Based on your previous response, return only the final structured answer. "
            "Do not use tools."
        )
    result = await agent.run(
        prompt,
        output_schema=schema,
        effort="low" if provider == "openai" else None,
        artifacts_dir=root / name / "result",
        raise_on_error=True,
    )
    if not isinstance(result.structured_output, schema):
        raise RuntimeError(f"{name} did not return {schema.__name__}")
    return result.structured_output


async def main() -> None:
    provider, model = example_config("anthropic", "claude-haiku-4-5")
    root = example_artifacts_dir("agent_flow", provider)
    plan = await run_stage(
        "planner",
        "Plan two steps for writing a short welcome note for a new teammate.",
        Plan,
        root,
        provider,
        model,
    )
    draft = await run_stage(
        "writer",
        f"Write the welcome note using this plan: {plan.model_dump_json()}",
        Draft,
        root,
        provider,
        model,
    )
    review = await run_stage(
        "reviewer",
        f"Review this note against the plan. Plan: {plan.model_dump_json()} "
        f"Draft: {draft.model_dump_json()}",
        Review,
        root,
        provider,
        model,
    )
    print(review.model_dump_json(indent=2))
    print(f"Draft: {draft.text}")
    print(f"Artifacts: {root}")


if __name__ == "__main__":
    asyncio.run(main())
