from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from agent_sdk_wrapper import Error, RunRequest, StructuredOutput, install_fake_providers

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def load_example_module(name: str) -> ModuleType:
    if str(EXAMPLES) not in sys.path:
        sys.path.insert(0, str(EXAMPLES))
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_auditor_style_run_stage_applies_stage_settings(monkeypatch, tmp_path: Path) -> None:
    module = load_example_module("auditor_style")
    seen: list[RunRequest] = []
    plan = {"objective": "Check", "tasks": [], "success_criteria": ["Done"]}
    install_fake_providers(
        monkeypatch, events=[StructuredOutput(value=plan)], seen_requests=seen
    )
    recorder = module.WorkflowRecorder(tmp_path)

    async def run(stage: str, mcp_servers):
        return await module.run_stage(
            provider="anthropic",
            model="claude-haiku-4-5",
            stage=stage,
            prompt="go",
            output_schema=module.WorkflowPlan,
            artifacts_dir=tmp_path / stage,
            recorder=recorder,
            mcp_servers=mcp_servers,
        )

    _, verified = asyncio.run(run("verifier", module.auditor_mcp_servers()))
    asyncio.run(run("planner", None))

    verifier_req, planner_req = seen
    assert verified == module.WorkflowPlan.model_validate(plan)
    assert verifier_req.system_prompt == module.STAGE_SYSTEM_PROMPTS["verifier"]
    assert verifier_req.max_turns == module.MAX_TURNS_BY_STAGE["verifier"]
    assert verifier_req.output_schema is module.WorkflowPlan
    assert verifier_req.allowed_tools == module.ALLOWED_MCP_TOOLS
    assert planner_req.system_prompt == module.STAGE_SYSTEM_PROMPTS["planner"]
    assert (planner_req.allowed_tools, planner_req.mcp_servers) == ([], [])
    assert recorder.stage_statuses == {"verifier": "success", "planner": "success"}


def test_auditor_style_run_stage_raises_provider_error(monkeypatch, tmp_path: Path) -> None:
    module = load_example_module("auditor_style")
    install_fake_providers(
        monkeypatch, events=[Error(message="provider rejected model", error_type="api_error_404")]
    )

    with pytest.raises(RuntimeError, match="analyst failed with error: provider rejected model"):
        asyncio.run(
            module.run_stage(
                provider="anthropic",
                model="claude-haiku-4-5",
                stage="analyst",
                prompt="go",
                output_schema=module.AnalysisReport,
                artifacts_dir=tmp_path,
                recorder=module.WorkflowRecorder(tmp_path),
            )
        )


def test_auditor_style_mcp_tools_are_read_only() -> None:
    module = load_example_module("auditor_style")

    [server] = module.auditor_mcp_servers()

    assert server.name == "auditor_demo"
    assert server.enabled_tools == ["read_project_brief", "read_artifact_policy"]
    assert "auditor_mcp_server.py" in server.args[0]


def test_auditor_style_prompts_chain_structured_outputs() -> None:
    module = load_example_module("auditor_style")
    plan = module.WorkflowPlan(
        objective="Check release readiness",
        tasks=[
            module.PlanTask(
                id="T1",
                question="Are examples runnable?",
                rationale="Examples are the user-facing workflow.",
            )
        ],
        success_criteria=["Examples are documented"],
    )
    analysis = module.AnalysisReport(
        observations=[
            module.Observation(
                id="O1",
                summary="Document Docker examples",
                evidence="The brief prioritizes Docker Compose workflows.",
                impact="medium",
                recommendation="Keep commands in README.",
            )
        ],
        open_questions=[],
    )
    verification = module.VerificationReport(
        items=[
            module.VerificationItem(
                observation_id="O1",
                verdict="supported",
                rationale="The recommendation follows the brief.",
            )
        ],
        overall_verdict="ready",
    )
    fix_plan = module.FixPlan(
        status="ready",
        actions=[
            module.FixAction(
                id="F1",
                title="Update docs",
                owner_role="docs",
                details="Document the workflow.",
                validation="Run examples offline.",
            )
        ],
        validation_steps=["Run docker compose run --rm python-fixtures"],
        traceability=["O1"],
    )

    analyst_prompt = module.analyst_prompt(plan)
    verifier_prompt = module.verifier_prompt(plan, analysis)
    reporter_prompt = module.reporter_prompt(plan, analysis, verification, fix_plan)

    assert '"objective": "Check release readiness"' in analyst_prompt
    assert '"summary": "Document Docker examples"' in verifier_prompt
    assert '"title": "Update docs"' in reporter_prompt
    assert '"overall_verdict": "ready"' in reporter_prompt


def test_auditor_style_writes_structured_artifact(tmp_path: Path) -> None:
    module = load_example_module("auditor_style")
    manifest = {
        "schema_version": 1,
        "files": {},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    report = module.FinalReport(
        status="ready",
        summary="Ready for release.",
        fix_plan_status="ready",
        next_actions=["Ship it"],
    )

    path = module.write_json_artifact(tmp_path, "report", report)

    saved = json.loads(path.read_text(encoding="utf-8"))
    updated_manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert saved["summary"] == "Ready for release."
    assert updated_manifest["files"]["report"] == "report.json"
