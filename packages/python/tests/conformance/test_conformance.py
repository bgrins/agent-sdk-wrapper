"""Run the shared conformance cases against the real Claude CLI and Codex app-server.

Offline, local mock APIs answer each case. Cases with a ``live`` section also run
against the real APIs with ``AGENT_SDK_WRAPPER_RUN_INTEGRATION=1`` and the
provider's key; they make billed calls.
"""

from __future__ import annotations

import inspect
import os
from typing import Any

import pytest

from agent_sdk_wrapper import Agent
from agent_sdk_wrapper.agent import _ALLOWED_OVERRIDES
from agent_sdk_wrapper.providers.anthropic_provider import AnthropicProvider
from agent_sdk_wrapper.providers.openai_provider import OpenAIProvider

from .runner import LIVE_KEYS, SPEC, live_view, python_view, run_case

pytest.importorskip("codex_cli_bin")

CASES = SPEC["cases"]
ADAPTERS = {"anthropic": AnthropicProvider, "codex": OpenAIProvider}


def _params(*, live: bool) -> list[Any]:
    params = []
    for case in CASES:
        if live and "live" not in case:
            continue
        view = python_view(case)
        if isinstance(view, str):
            params.append(pytest.param(case, id=case["id"], marks=pytest.mark.skip(reason=view)))
        elif not (live and view["expect"].get("config_error")):
            params.append(pytest.param(view, id=case["id"]))
    return params


@pytest.mark.parametrize("case", _params(live=False))
async def test_case(case, tmp_path, monkeypatch):
    await run_case(case, tmp_path, monkeypatch, live=False)


@pytest.mark.integration
@pytest.mark.parametrize("case", _params(live=True))
async def test_live_case(case, tmp_path, monkeypatch):
    if os.environ.get("AGENT_SDK_WRAPPER_RUN_INTEGRATION") != "1":
        pytest.skip("live cases require AGENT_SDK_WRAPPER_RUN_INTEGRATION=1")
    if not os.environ.get(LIVE_KEYS[case["provider"]]):
        pytest.skip(f"{LIVE_KEYS[case['provider']]} is required")
    await run_case(live_view(case), tmp_path, monkeypatch, live=True)


def _keywords(fn: Any) -> set[str]:
    return {
        name
        for name, param in inspect.signature(fn).parameters.items()
        if param.kind is inspect.Parameter.KEYWORD_ONLY
    }


def _options() -> dict[str, set[str]]:
    """Every option name a caller can pass, per provider."""

    shared = _keywords(Agent.__init__) | _keywords(Agent.run) | _keywords(Agent.stream)
    shared |= _ALLOWED_OVERRIDES
    return {
        provider: shared | {f"provider_options.{name}" for name in _keywords(adapter)}
        for provider, adapter in ADAPTERS.items()
    }


def _used(case: dict[str, Any]) -> set[str]:
    used = {"provider"}
    sections = [case, *case.get("runs", []), case.get("live", {})]
    for section in sections:
        for key in ("options", "run_options"):
            options = section.get(key, {})
            used |= set(options)
            used |= {f"provider_options.{name}" for name in options.get("provider_options", {})}
    return used


def test_every_option_has_a_case_or_an_exemption():
    exemptions = SPEC["coverage_exemptions"]["python"]
    for provider, options in _options().items():
        used = set().union(*(_used(c) for c in CASES if c["provider"] == provider))
        exempt = {key.split(":", 1)[1] for key in exemptions if key.startswith(f"{provider}:")}
        assert not exempt - options, f"{provider} exemptions name unknown options"
        assert not exempt & used, f"{provider} exemptions name options a case uses"
        assert options - used - exempt == set(), f"{provider} options without a case"
