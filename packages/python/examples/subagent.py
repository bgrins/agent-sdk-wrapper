"""Delegate to a subagent. The parent invokes the subagent as a tool and gets
the result back, with control returning to the parent.

Anthropic maps ``SubagentDef`` to ``AgentDefinition``. Codex maps it to
Codex multi-agent configuration.

    uv run python examples/subagent.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from _shared import ExampleOutput, example_artifacts_dir, example_config

from agent_sdk_wrapper import Agent, ProviderEventEnvelope, SubagentDef

DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODEL = "claude-haiku-4-5"


def _provider_event_summary(env: ProviderEventEnvelope) -> str | None:
    if env.provider == "anthropic":
        return _anthropic_provider_event_summary(env)
    if env.provider == "openai":
        return _codex_provider_event_summary(env)
    return None


def _anthropic_provider_event_summary(env: ProviderEventEnvelope) -> str | None:
    msg = env.message if isinstance(env.message, dict) else {}
    if env.class_name.endswith(".TaskStartedMessage"):
        return f"anthropic task started [{msg.get('task_id')}]: {msg.get('description')}"
    if env.class_name.endswith(".TaskProgressMessage"):
        return (
            f"anthropic task progress [{msg.get('task_id')}]: "
            f"last_tool={msg.get('last_tool_name')}"
        )
    if env.class_name.endswith(".TaskNotificationMessage"):
        return (
            f"anthropic task {msg.get('status')} [{msg.get('task_id')}]: "
            f"{msg.get('summary') or '(no summary)'}"
        )
    return None


def _codex_provider_event_summary(env: ProviderEventEnvelope) -> str | None:
    msg = env.message if isinstance(env.message, dict) else {}
    method = msg.get("method")
    if method not in {"item/started", "item/completed"}:
        return None

    item = _codex_event_item(msg.get("payload"))
    if item.get("type") != "collabAgentToolCall":
        return None

    action = "started" if method == "item/started" else "completed"
    receiver = item.get("tool") or item.get("name") or item.get("id")
    status = item.get("status")
    summary = f"codex subagent {action} [{receiver}]"
    if status:
        summary += f": status={status}"
    return summary


def _codex_event_item(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    item = payload.get("item")
    if not isinstance(item, dict):
        return {}
    root = item.get("root", item)
    return root if isinstance(root, dict) else {}


async def main() -> None:
    provider, model = example_config(DEFAULT_PROVIDER, DEFAULT_MODEL)
    artifacts_dir = example_artifacts_dir("subagent", provider)
    out = ExampleOutput(artifacts_dir)

    def on_provider_event(env: ProviderEventEnvelope) -> None:
        summary = _provider_event_summary(env)
        if summary is not None:
            out.print(f"provider event: {summary}")

    try:
        prompt = (
            "Delegate this to the `reviewer` subagent:\n\n"
            "    def add(a, b):\n"
            "        return a - b\n\n"
            "Return only the reviewer's verdict."
        )
        agent = Agent(
            provider=provider,
            model=model,
            system_prompt="Use the reviewer subagent when the prompt asks for delegation.",
            subagents={
                "reviewer": SubagentDef(
                    description="Reviews short code snippets for bugs.",
                    prompt="You are a terse code reviewer. Reply in one sentence.",
                    model=model,
                )
            },
            artifacts_dir=artifacts_dir,
            on_provider_event=on_provider_event,
        )
        result = await agent.run(prompt)
        out.print(result.final_text)
        out.print(f"artifacts: {result.artifacts_dir}")
        out.print("---")
        for tc in result.tool_calls():
            out.print(f"called: {tc.name}({tc.input})")
    finally:
        out.save()


if __name__ == "__main__":
    asyncio.run(main())
