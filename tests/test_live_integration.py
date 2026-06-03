"""Opt-in live integration coverage for both providers.

Run through Docker Compose:

    docker compose run --rm integration

The module is skipped unless ``AGENT_SDK_WRAPPER_RUN_INTEGRATION=1`` is set. Individual
provider cases are skipped when their API key is missing.
Artifacts default to ``results/integration-runs/<timestamp>/`` so Docker
Compose runs leave inspectable traces on the host bind mount. Set
``AGENT_SDK_WRAPPER_TEST_ARTIFACTS_DIR`` to override the root.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel, Field

from agent_sdk_wrapper import Agent, ConfigError, McpStdioServer, SubagentDef, normalize_provider

pytestmark = pytest.mark.integration

if os.environ.get("AGENT_SDK_WRAPPER_RUN_INTEGRATION") != "1":
    pytest.skip(
        "live integration tests require AGENT_SDK_WRAPPER_RUN_INTEGRATION=1",
        allow_module_level=True,
    )


ROOT = Path(__file__).resolve().parents[1]
SIMPLE_MCP_SERVER = ROOT / "tests" / "fixtures" / "simple_mcp_server.py"
_DEFAULT_ARTIFACT_ROOT = (
    ROOT
    / "results"
    / "integration-runs"
    / datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
)


@dataclass(frozen=True)
class LiveProvider:
    provider: str
    label: str
    model: str | None


class LiveToolHandoff(BaseModel):
    handoff_id: str = Field(description="Stable handoff id from the intake stage.")
    score: int = Field(description="Score computed with the live_add tool.")
    signal: str = Field(description="Release-readiness signal returned by a tool.")
    next_agent: Literal["verifier"]


class LiveHandoffDecision(BaseModel):
    handoff_id: str
    accepted: bool
    next_action: Literal["ship", "revise"]
    rationale: str


def live_add(a: int, b: int) -> int:
    """Add two integers."""

    return a + b


def live_double(n: int) -> int:
    """Double an integer."""

    return n * 2


def live_release_signal(topic: str) -> str:
    """Return a deterministic release-readiness signal for a topic."""

    if topic.lower() == "docker":
        return "docker compose integration path is required"
    return f"{topic} requires follow-up"


def _structured(result, schema: type[BaseModel]) -> BaseModel:
    if isinstance(result.structured_output, schema):
        return result.structured_output
    return schema.model_validate(result.structured_output)


def _integration_artifact_root() -> Path:
    configured = os.environ.get("AGENT_SDK_WRAPPER_TEST_ARTIFACTS_DIR")
    if not configured:
        return _DEFAULT_ARTIFACT_ROOT
    path = Path(configured)
    return path if path.is_absolute() else ROOT / path


def _artifacts_dir(live_provider: LiveProvider, *parts: str) -> Path:
    return _integration_artifact_root() / live_provider.label / Path(*parts)


def _selected_providers() -> list[str]:
    raw = (
        os.environ.get("AGENT_SDK_WRAPPER_TEST_PROVIDERS")
        or os.environ.get("PROVIDER")
        or "anthropic,openai"
    )
    providers: list[str] = []
    for part in raw.split(","):
        value = part.strip()
        if not value:
            continue
        try:
            provider = normalize_provider(value)
        except ConfigError:
            continue
        if provider is not None and provider not in providers:
            providers.append(provider)
    return providers or ["anthropic", "openai"]


def _model_for(provider: str) -> str | None:
    selected_provider = None
    try:
        selected_provider = normalize_provider(os.environ.get("PROVIDER"))
    except ConfigError:
        selected_provider = None

    if provider == "anthropic":
        return (
            os.environ.get("AGENT_SDK_WRAPPER_ANTHROPIC_MODEL")
            or os.environ.get("ANTHROPIC_MODEL")
            or (os.environ.get("MODEL") if selected_provider == "anthropic" else None)
            or "claude-haiku-4-5"
        )
    return (
        os.environ.get("AGENT_SDK_WRAPPER_OPENAI_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or (os.environ.get("MODEL") if selected_provider == "openai" else None)
        or None
    )


@pytest.fixture(
    params=_selected_providers(),
    ids=lambda provider: "codex" if provider == "openai" else provider,
)
def live_provider(request: pytest.FixtureRequest) -> LiveProvider:
    provider = str(request.param)
    if provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY is required for Anthropic integration tests")
    if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is required for Codex integration tests")
    return LiveProvider(
        provider=provider,
        label="codex" if provider == "openai" else provider,
        model=_model_for(provider),
    )


@pytest.mark.asyncio
async def test_live_basic_text(live_provider: LiveProvider) -> None:
    agent = Agent(
        provider=live_provider.provider,
        model=live_provider.model,
        system_prompt="Follow the user's formatting instruction exactly.",
        max_retries=0,
    )

    result = await agent.run(
        "Reply with exactly LIVE_PONG and nothing else.",
        artifacts_dir=_artifacts_dir(live_provider, "basic"),
        raise_on_error=True,
    )

    assert result.ok
    assert "LIVE_PONG" in result.final_text


@pytest.mark.asyncio
async def test_live_callable_tools(live_provider: LiveProvider) -> None:
    agent = Agent(
        provider=live_provider.provider,
        model=live_provider.model,
        tools=[live_add, live_double],
        system_prompt="Use the provided tools when arithmetic is requested.",
        max_retries=0,
    )

    result = await agent.run(
        "Use the tools to compute double(add(3, 4)). Reply with ANSWER: 14.",
        artifacts_dir=_artifacts_dir(live_provider, "callable-tools"),
        raise_on_error=True,
    )

    assert result.ok
    assert "14" in result.final_text
    tool_names = {call.name or "" for call in result.tool_calls()}
    assert any("live_add" in name or name.endswith("add") for name in tool_names)
    assert any("live_double" in name or name.endswith("double") for name in tool_names)


@pytest.mark.asyncio
async def test_live_structured_tool_handoff_chain(
    live_provider: LiveProvider,
) -> None:
    intake_agent = Agent(
        provider=live_provider.provider,
        model=live_provider.model,
        tools=[live_add, live_release_signal],
        output_schema=LiveToolHandoff,
        system_prompt=(
            "You are an intake agent. Use the available tools for arithmetic and "
            "release-signal lookup. Return only structured output matching the schema."
        ),
        max_retries=0,
    )

    intake = await intake_agent.run(
        "Create handoff LIVE-HANDOFF-001. Call live_add with a=7 and b=5 for "
        "the score. Call live_release_signal with topic docker for the signal. "
        "Set next_agent to verifier.",
        artifacts_dir=_artifacts_dir(live_provider, "advanced-handoff", "intake"),
        raise_on_error=True,
    )
    handoff = _structured(intake, LiveToolHandoff)

    assert intake.ok
    assert handoff.handoff_id == "LIVE-HANDOFF-001"
    assert handoff.score == 12
    assert "docker" in handoff.signal.lower()
    tool_names = {call.name or "" for call in intake.tool_calls()}
    assert any("live_add" in name or name.endswith("add") for name in tool_names)
    assert any("live_release_signal" in name for name in tool_names)

    verifier_agent = Agent(
        provider=live_provider.provider,
        model=live_provider.model,
        output_schema=LiveHandoffDecision,
        system_prompt=(
            "You are a verifier agent receiving a structured handoff. Preserve "
            "the handoff id. Accept it only when score is 12 and the signal "
            "mentions docker."
        ),
        max_retries=0,
    )
    verifier = await verifier_agent.run(
        "Review this handoff and return a structured decision:\n"
        f"{handoff.model_dump_json()}",
        artifacts_dir=_artifacts_dir(live_provider, "advanced-handoff", "verifier"),
        raise_on_error=True,
    )
    decision = _structured(verifier, LiveHandoffDecision)

    assert verifier.ok
    assert decision.handoff_id == handoff.handoff_id
    assert decision.accepted is True
    assert decision.next_action == "ship"


@pytest.mark.asyncio
async def test_live_external_mcp(live_provider: LiveProvider) -> None:
    mcp_artifacts = _artifacts_dir(live_provider, "mcp-server-artifacts")
    server = McpStdioServer(
        name="brief_tools",
        command=sys.executable,
        args=[str(SIMPLE_MCP_SERVER)],
        env={"SIMPLE_MCP_ARTIFACTS_DIR": str(mcp_artifacts)},
        enabled_tools=["read_brief"],
    )
    agent = Agent(
        provider=live_provider.provider,
        model=live_provider.model,
        mcp_servers=[server],
        allowed_tools=["mcp__brief_tools__read_brief"],
        system_prompt="Use available MCP tools when asked to inspect a brief.",
        max_retries=0,
    )

    result = await agent.run(
        "Use read_brief, then reply with LIVE_MCP_OK.",
        artifacts_dir=_artifacts_dir(live_provider, "external-mcp"),
        raise_on_error=True,
    )

    assert result.ok
    assert "LIVE_MCP_OK" in result.final_text
    assert any("read_brief" in (call.name or "") for call in result.tool_calls())


@pytest.mark.asyncio
async def test_live_session_resume(live_provider: LiveProvider) -> None:
    agent = Agent(
        provider=live_provider.provider,
        model=live_provider.model,
        continue_session=True,
        max_retries=0,
    )

    first = await agent.run(
        "Remember the token ALPHA42 for the next turn. Reply READY.",
        artifacts_dir=_artifacts_dir(live_provider, "session-1"),
        raise_on_error=True,
    )
    second = await agent.run(
        "What token did I ask you to remember? Reply with only the token.",
        artifacts_dir=_artifacts_dir(live_provider, "session-2"),
        raise_on_error=True,
    )

    assert first.ok
    assert second.ok
    assert first.session_id
    assert second.session_id == first.session_id
    assert "ALPHA42" in second.final_text


@pytest.mark.asyncio
async def test_live_subagent(live_provider: LiveProvider) -> None:
    model = live_provider.model
    agent = Agent(
        provider=live_provider.provider,
        model=model,
        subagents={
            "reviewer": SubagentDef(
                description="Reviews tiny Python snippets for obvious bugs.",
                prompt="You are a terse reviewer. Return one sentence.",
                model=model,
            )
        },
        max_retries=0,
    )

    result = await agent.run(
        "Delegate this to reviewer: def add(a, b): return a - b. "
        "Return the reviewer's verdict.",
        artifacts_dir=_artifacts_dir(live_provider, "subagent"),
        raise_on_error=True,
    )

    assert result.ok
    assert result.final_text.strip()
