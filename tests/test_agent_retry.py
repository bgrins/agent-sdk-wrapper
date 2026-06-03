"""Retry/backoff behavior for the unified Agent."""

from __future__ import annotations

import asyncio

import pytest

from agent_sdk_wrapper import Agent, RunStatus, Text, TransientError
from agent_sdk_wrapper.providers import base
from agent_sdk_wrapper.providers import openai_provider as op_mod


class FlakyProvider(base.ProviderAdapter):
    """Fails the first N attempts with TransientError, then succeeds."""

    name = "openai"
    fail_n = 1
    calls = 0

    async def stream(self, req):  # type: ignore[override]
        FlakyProvider.calls += 1
        if FlakyProvider.calls <= FlakyProvider.fail_n:
            raise TransientError("rate limit (synthetic)")
        yield Text(text="ok")


def test_run_retries_transient_then_succeeds(monkeypatch):
    monkeypatch.setattr(op_mod, "OpenAIProvider", FlakyProvider)
    FlakyProvider.calls = 0
    FlakyProvider.fail_n = 2

    agent = Agent(provider="openai", max_retries=3)
    result = asyncio.run(agent.run("hi"))

    assert FlakyProvider.calls == 3
    assert result.status == RunStatus.SUCCESS
    assert result.final_text == "ok"


def test_run_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr(op_mod, "OpenAIProvider", FlakyProvider)
    FlakyProvider.calls = 0
    FlakyProvider.fail_n = 99

    agent = Agent(provider="openai", max_retries=1)
    result = asyncio.run(agent.run("hi"))

    assert FlakyProvider.calls == 2  # initial + 1 retry
    assert result.status == RunStatus.FAILURE
    assert "rate limit" in (result.error or "").lower()


def test_raise_on_error(monkeypatch):
    monkeypatch.setattr(op_mod, "OpenAIProvider", FlakyProvider)
    FlakyProvider.calls = 0
    FlakyProvider.fail_n = 99

    agent = Agent(provider="openai", max_retries=0, raise_on_error=True)
    from agent_sdk_wrapper import RunFailedError

    with pytest.raises(RunFailedError):
        asyncio.run(agent.run("hi"))
