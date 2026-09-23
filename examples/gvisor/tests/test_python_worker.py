"""Offline checks for the Python worker; scripts/test.sh runs them without Docker."""

import asyncio
import importlib.util
import json
import time
from pathlib import Path

from agent_sdk_wrapper import Agent
from agent_sdk_wrapper.events import Text
from agent_sdk_wrapper.testing import FakeProvider

spec = importlib.util.spec_from_file_location(
    "worker", Path(__file__).parents[1] / "workload" / "agent-python" / "main.py"
)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def test_runs_share_one_budget_and_each_prints_a_result(tmp_path, capsys):
    async def events(req):
        yield Text(text=req.prompt)
        await asyncio.sleep(0.6 if req.prompt == "first" else 60)

    agent = Agent(provider="anthropic", model="claude-haiku-4-5")
    agent._provider = FakeProvider(events)
    start = time.monotonic()
    ok = asyncio.run(
        worker.run_prompts(agent, ["first", "second"], str(tmp_path / "job"), budget=1)
    )
    elapsed = time.monotonic() - start
    results = [json.loads(line)["result"] for line in capsys.readouterr().out.splitlines()]
    assert not ok
    assert [result["status"] for result in results] == ["success", "timeout"]
    # A full budget per run would take 1.6 s.
    assert elapsed < 1.4
