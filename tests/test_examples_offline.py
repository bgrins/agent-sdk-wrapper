"""Offline validation for example scripts.

These tests run the example ``main()`` functions with fake providers. That keeps
the default test suite API-key free while still checking example control flow,
artifact paths, stdout capture, and provider/model environment switching.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

from agent_sdk_wrapper import (
    AgentUpdated,
    RunRequest,
    SessionInfo,
    StructuredOutput,
    Text,
    TokenUsage,
    ToolCall,
    ToolResult,
    Usage,
)
from agent_sdk_wrapper.artifacts import ProviderEventLogger
from agent_sdk_wrapper.providers import anthropic_provider as anthropic_mod
from agent_sdk_wrapper.providers import base
from agent_sdk_wrapper.providers import openai_provider as openai_mod

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


@dataclass
class TaskStartedMessage:
    task_id: str
    description: str


@dataclass
class TaskNotificationMessage:
    task_id: str
    status: str
    summary: str


class OfflineExampleProvider(base.ProviderAdapter):
    name = "offline"

    def __init__(self, seen_requests: list[RunRequest] | None = None, **_: object) -> None:
        self._seen_requests = seen_requests

    async def stream(self, req: RunRequest):  # type: ignore[override]
        if self._seen_requests is not None:
            self._seen_requests.append(req)

        yield SessionInfo(id=req.session_id or "offline-session")

        for fn in req.tools:
            name = getattr(fn, "__name__", "tool")
            yield ToolCall(id=f"tool-{name}", name=name, input={"offline": True})
            yield ToolResult(id=f"tool-{name}", output="offline tool result")

        for server in req.mcp_servers:
            tool_name = (server.enabled_tools or ["read_file"])[0]
            yield ToolCall(
                id=f"mcp-{server.name}-{tool_name}",
                name=f"mcp__{server.name}__{tool_name}",
                input={"path": "README.md"},
            )
            yield ToolResult(
                id=f"mcp-{server.name}-{tool_name}",
                output="offline mcp result",
            )

        for name in req.subagents:
            _write_offline_subagent_provider_events(req, name)
            yield AgentUpdated(name=name)
            yield ToolCall(
                id=f"subagent-{name}",
                name=f"agent.{name}",
                input={"prompt": "offline"},
            )
            yield ToolResult(id=f"subagent-{name}", output="offline subagent result")

        if req.output_schema is not None:
            structured = _fake_structured_output(req.output_schema)
            yield Text(text="offline structured response")
            yield StructuredOutput(value=structured)
        else:
            yield Text(text="offline example response")

        yield Usage(usage=TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2))


def _write_offline_subagent_provider_events(req: RunRequest, name: str) -> None:
    logger = ProviderEventLogger(req.provider, req.artifacts_dir, req.on_provider_event)
    if req.provider == "anthropic":
        logger.write(
            TaskStartedMessage(
                task_id=f"offline-{name}",
                description=f"Offline subagent {name}",
            )
        )
        logger.write(
            TaskNotificationMessage(
                task_id=f"offline-{name}",
                status="completed",
                summary="Offline subagent finished.",
            )
        )
    elif req.provider == "openai":
        item = {
            "id": f"offline-{name}",
            "status": "completed",
            "tool": name,
            "type": "collabAgentToolCall",
        }
        logger.write({"method": "item/started", "payload": {"item": item}})
        logger.write({"method": "item/completed", "payload": {"item": item}})


def _fake_structured_output(schema: type):
    name = getattr(schema, "__name__", "")
    if name == "Weather":
        return schema(city="Portland", temperature_c=21.0, conditions="clear")
    if name == "WorkflowPlan":
        return schema(
            objective="Check release readiness",
            tasks=[
                {
                    "id": "T1",
                    "question": "Are examples runnable?",
                    "rationale": "Examples are the user-facing workflow.",
                }
            ],
            success_criteria=["Examples are documented"],
        )
    if name == "AnalysisReport":
        return schema(
            observations=[
                {
                    "id": "O1",
                    "summary": "Document Docker examples",
                    "evidence": "The brief prioritizes Docker Compose workflows.",
                    "impact": "medium",
                    "recommendation": "Keep commands in README.",
                }
            ],
            open_questions=[],
        )
    if name == "VerificationReport":
        return schema(
            items=[
                {
                    "observation_id": "O1",
                    "verdict": "supported",
                    "rationale": "The observation follows the brief.",
                }
            ],
            overall_verdict="ready",
        )
    if name == "FixPlan":
        return schema(
            status="ready",
            actions=[
                {
                    "id": "F1",
                    "title": "Document fixture promotion",
                    "owner_role": "docs",
                    "details": "Keep the fixture workflow visible in README.",
                    "validation": "Run docker compose run --rm fixtures.",
                }
            ],
            validation_steps=["Run the offline fixture check."],
            traceability=["O1"],
        )
    if name == "FinalReport":
        return schema(
            status="ready",
            summary="Offline structured chain completed.",
            fix_plan_status="ready",
            next_actions=["Keep Docker example commands documented."],
        )
    return schema()


def load_example_module(name: str, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(EXAMPLES))
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def install_fake_providers(
    monkeypatch: pytest.MonkeyPatch,
    seen_requests: list[RunRequest] | None = None,
) -> None:
    class CapturingOfflineProvider(OfflineExampleProvider):
        def __init__(self, **options: object) -> None:
            super().__init__(seen_requests, **options)

    monkeypatch.setattr(anthropic_mod, "AnthropicProvider", CapturingOfflineProvider)
    monkeypatch.setattr(openai_mod, "OpenAIProvider", CapturingOfflineProvider)


def latest_artifact_dir(root: Path, provider: str, example: str) -> Path:
    matches = sorted((root / "results" / provider / example).glob("*"))
    assert matches, f"no artifact directory for {provider}/{example}"
    return matches[-1]


def assert_artifacts(root: Path, provider: str, example: str) -> Path:
    run_dir = latest_artifact_dir(root, provider, example)
    artifacts_dir = run_dir / "artifacts" if (run_dir / "artifacts").exists() else run_dir
    manifest_path = artifacts_dir / "manifest.json"
    result_path = artifacts_dir / "result.json"
    stdout_path = artifacts_dir / "stdout.txt"

    assert manifest_path.exists()
    assert result_path.exists()
    assert stdout_path.exists()
    assert (artifacts_dir / "trace.jsonl").exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    stdout = stdout_path.read_text(encoding="utf-8")

    assert manifest["provider"] == provider
    assert manifest["status"] == "success"
    assert manifest["files"]["result"] == "result.json"
    assert manifest["files"]["stdout"] == "stdout.txt"
    assert result["status"] == "success"
    assert stdout.strip()
    return artifacts_dir


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "example",
    [
        "run_basic",
        "stream",
        "structured_output",
        "custom_tools",
        "subagent",
        "auditor_style",
    ],
)
async def test_examples_write_artifacts_offline(
    example: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_fake_providers(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PROVIDER", raising=False)
    monkeypatch.delenv("MODEL", raising=False)

    module = load_example_module(example, monkeypatch)

    await module.main()

    artifacts_dir = assert_artifacts(tmp_path, "anthropic", example)
    if example == "subagent":
        stdout = (artifacts_dir / "stdout.txt").read_text(encoding="utf-8")
        assert "provider event: anthropic task started" in stdout
        assert "provider event: anthropic task completed" in stdout
    if example == "auditor_style":
        assert artifacts_dir.name == "artifacts"
        plan = json.loads((artifacts_dir / "plan.json").read_text(encoding="utf-8"))
        analysis = json.loads((artifacts_dir / "analysis.json").read_text(encoding="utf-8"))
        verification = json.loads(
            (artifacts_dir / "verification.json").read_text(encoding="utf-8")
        )
        fix_plan = json.loads((artifacts_dir / "fix_plan.json").read_text(encoding="utf-8"))
        report = json.loads((artifacts_dir / "report.json").read_text(encoding="utf-8"))
        stats = json.loads(
            (artifacts_dir / "workflow_stats.json").read_text(encoding="utf-8")
        )
        workflow_manifest = json.loads(
            (artifacts_dir / "manifest.json").read_text(encoding="utf-8")
        )

        assert plan["tasks"][0]["id"] == "T1"
        assert analysis["observations"][0]["id"] == "O1"
        assert verification["items"][0]["observation_id"] == "O1"
        assert fix_plan["actions"][0]["id"] == "F1"
        assert report["status"] == "ready"
        assert report["fix_plan_status"] == "ready"
        assert stats["stage_statuses"]["fix_planner"] == "success"
        assert stats["tool_calls"] >= 3
        assert workflow_manifest["files"]["analysis_manifest"] == "analysis/manifest.json"
        assert workflow_manifest["files"]["verification_manifest"] == (
            "verification/manifest.json"
        )
        assert workflow_manifest["files"]["fix_plan_manifest"] == (
            "fix-plan/manifest.json"
        )
        assert workflow_manifest["files"]["report_manifest"] == "report/manifest.json"
        assert workflow_manifest["files"]["workflow_events"] == "workflow-events.jsonl"
        assert workflow_manifest["files"]["workflow_stats"] == "workflow_stats.json"
        assert workflow_manifest["files"]["context"] == "context.md"


@pytest.mark.asyncio
async def test_examples_can_switch_to_codex_with_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen_requests: list[RunRequest] = []
    install_fake_providers(monkeypatch, seen_requests)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PROVIDER", "codex")
    monkeypatch.setenv("MODEL", "gpt-5")

    module = load_example_module("run_basic", monkeypatch)

    await module.main()

    assert_artifacts(tmp_path, "openai", "run_basic")
    assert seen_requests
    assert seen_requests[0].provider == "openai"
    assert seen_requests[0].model == "gpt-5"


@pytest.mark.asyncio
async def test_subagent_example_logs_codex_provider_events(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_fake_providers(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PROVIDER", "codex")
    monkeypatch.setenv("MODEL", "gpt-5")

    module = load_example_module("subagent", monkeypatch)

    await module.main()

    artifacts_dir = assert_artifacts(tmp_path, "openai", "subagent")
    stdout = (artifacts_dir / "stdout.txt").read_text(encoding="utf-8")
    assert "provider event: codex subagent started [reviewer]" in stdout
    assert "provider event: codex subagent completed [reviewer]" in stdout
