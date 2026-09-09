"""Run matched scenarios on both providers so their traces can be compared.

Each scenario runs the same prompt, tools, and schema against an Anthropic and
an OpenAI model, writing artifacts to
``results/compare/<timestamp>/<scenario>/<label>/``. Load the pairs side by side
in ``docs/trace-viewer.html`` to see how each backend's native stream lands in
the normalized event model.

    uv run python scripts/compare_providers.py
    uv run python scripts/compare_providers.py --scenario tools --scenario basic

Requires ANTHROPIC_API_KEY and OPENAI_API_KEY; these are live, billed runs.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_sdk_wrapper import Agent, RunResult  # noqa: E402

DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5"
DEFAULT_OPENAI_MODEL = "gpt-5.6-luna"


class Verdict(BaseModel):
    """Structured-output schema for the ``structured`` scenario."""

    verdict: str
    confidence: float


def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def square(n: int) -> int:
    """Square an integer."""
    return n * n


SCENARIOS: dict[str, dict[str, object]] = {
    "basic": {
        "prompt": "Reply with exactly PONG and nothing else.",
        "system_prompt": "Follow the user's formatting instruction exactly.",
    },
    "tools": {
        "prompt": "What is (3 + 4) squared? Use the provided tools, then state the number.",
        "system_prompt": "Use the provided tools to compute. Be brief.",
        "tools": [add, square],
    },
    "structured": {
        "prompt": "Is the statement '2 + 2 = 4' true? Answer with the schema.",
        "system_prompt": "Answer only through the structured output schema.",
        "output_schema": Verdict,
    },
    "thinking": {
        "prompt": (
            "A farmer has 17 sheep and all but 9 run away. How many are left? "
            "Reason it through, then give the number."
        ),
        "system_prompt": "Think before answering. Keep the final answer to one line.",
        "effort": "medium",
    },
}


async def run_scenario(
    name: str,
    label: str,
    provider: str,
    model: str,
    root: Path,
) -> tuple[str, str, RunResult | None, str | None]:
    """Run one scenario against one provider, returning its outcome row."""
    spec = dict(SCENARIOS[name])
    prompt = str(spec.pop("prompt"))
    artifacts_dir = root / name / label
    agent = Agent(
        provider=provider,
        model=model,
        artifacts_dir=artifacts_dir,
        max_turns=8,
        max_retries=1,
        **spec,  # type: ignore[arg-type]
    )
    try:
        result = await agent.run(prompt)
    except Exception as exc:  # noqa: BLE001 - one failure must not stop the sweep
        return name, label, None, f"{type(exc).__name__}: {exc}"
    return name, label, result, result.error


def summarize(rows: Sequence[tuple[str, str, RunResult | None, str | None]]) -> str:
    """Render the sweep outcome as an aligned table."""
    header = (
        f"{'scenario':<12} {'provider':<10} {'status':<9} "
        f"{'events':>6} {'tokens':>14} {'cost':>9}"
    )
    lines = [header, "-" * len(header)]
    for name, label, result, error in rows:
        if result is None:
            lines.append(f"{name:<12} {label:<10} {'crashed':<9} {'-':>6} {'-':>14} {'-':>9}")
            lines.append(f"{'':<23} {error}")
            continue
        usage = result.usage
        tokens = f"{usage.input_tokens}/{usage.output_tokens}" if usage else "-"
        cost = f"${result.cost_usd:.4f}" if result.cost_usd is not None else "-"
        lines.append(
            f"{name:<12} {label:<10} {result.status.value:<9} "
            f"{len(result.events):>6} {tokens:>14} {cost:>9}"
        )
        if error:
            lines.append(f"{'':<23} {error}")
    return "\n".join(lines)


async def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        action="append",
        choices=sorted(SCENARIOS),
        help="Scenario to run; repeatable. Defaults to all.",
    )
    parser.add_argument("--anthropic-model", default=DEFAULT_ANTHROPIC_MODEL)
    parser.add_argument("--openai-model", default=DEFAULT_OPENAI_MODEL)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Artifact root. Defaults to results/compare/<timestamp>.",
    )
    args = parser.parse_args(argv)

    missing = [
        key
        for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")
        if not os.environ.get(key)
    ]
    if missing:
        parser.error(f"missing credentials: {', '.join(missing)}")

    scenarios = args.scenario or sorted(SCENARIOS)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    root = args.output_dir or Path("results") / "compare" / timestamp
    targets = [
        ("anthropic", "anthropic", args.anthropic_model),
        ("codex", "openai", args.openai_model),
    ]

    rows = []
    for name in scenarios:
        # Both providers run the scenario concurrently; scenarios run in order
        # so the console output stays readable as a paired comparison.
        rows.extend(
            await asyncio.gather(
                *(
                    run_scenario(name, label, provider, model, root)
                    for label, provider, model in targets
                )
            )
        )

    print(summarize(rows))
    print(f"\nartifacts: {root}")
    return 0 if all(row[2] is not None and row[2].ok for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
