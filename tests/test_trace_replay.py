"""Replay committed trace fixtures through fake providers."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent_sdk_wrapper import (
    Agent,
    install_fake_providers,
    load_trace_replay,
    run_result_summary,
)

ROOT = Path(__file__).resolve().parents[1]
TRACE_FIXTURES = ROOT / "tests" / "fixtures" / "traces"


@pytest.mark.parametrize(
    "trace_path",
    sorted(TRACE_FIXTURES.glob("*.trace.jsonl")),
    ids=lambda path: path.name,
)
def test_trace_fixture_replays_to_expected_result(monkeypatch, trace_path: Path) -> None:
    replay = load_trace_replay(trace_path)
    install_fake_providers(
        monkeypatch,
        events=replay.events,
        providers=[replay.provider],
    )

    result = asyncio.run(
        Agent(
            provider=replay.provider,
            model=replay.model,
            cwd=replay.cwd,
            max_retries=0,
        ).run("replay fixture")
    )

    assert run_result_summary(result) == replay.expected
