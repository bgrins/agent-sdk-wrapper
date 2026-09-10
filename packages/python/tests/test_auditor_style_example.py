from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

from agent_sdk_wrapper import Error, EventEnvelope, Text

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


def test_auditor_style_uses_structured_agent_chain() -> None:
    module = load_example_module("auditor_style")

    assert set(module.STAGE_SYSTEM_PROMPTS) == {
        "planner",
        "analyst",
        "verifier",
        "fix_planner",
        "reporter",
    }
    assert "Do not perform a security or vulnerability audit" in module.WORKFLOW_BRIEF
    assert ".env files" in module.SCOPE_BOUNDARY
    assert module.MAX_TURNS_BY_STAGE["fix_planner"] > 0


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
        validation_steps=["Run docker compose run --rm fixtures"],
        traceability=["O1"],
    )

    analyst_prompt = module.analyst_prompt(plan)
    verifier_prompt = module.verifier_prompt(plan, analysis)
    fix_prompt = module.fix_planner_prompt(plan, analysis, verification)
    reporter_prompt = module.reporter_prompt(plan, analysis, verification, fix_plan)
    prompts = [
        module.planner_prompt(),
        analyst_prompt,
        verifier_prompt,
        fix_prompt,
        reporter_prompt,
    ]

    assert '"objective": "Check release readiness"' in analyst_prompt
    assert '"summary": "Document Docker examples"' in verifier_prompt
    assert '"title": "Update docs"' in reporter_prompt
    assert '"overall_verdict": "ready"' in reporter_prompt
    assert "Do not look for vulnerabilities" in analyst_prompt
    assert "rather than a source patch" in fix_prompt
    assert all(module.SCOPE_BOUNDARY in prompt for prompt in prompts)


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


def test_auditor_style_recorder_prints_live_progress(tmp_path: Path) -> None:
    module = load_example_module("auditor_style")
    messages: list[str] = []
    recorder = module.WorkflowRecorder(tmp_path, live_print=messages.append)

    recorder.start_stage("planner", tmp_path / "planner", mcp_enabled=False)
    recorder.on_event("planner")(
        EventEnvelope(
            run_id="run",
            sequence=0,
            timestamp="2026-06-03T00:00:00+00:00",
            event=Text(text="planning complete"),
        )
    )
    recorder.on_event("planner")(
        EventEnvelope(
            run_id="run",
            sequence=1,
            timestamp="2026-06-03T00:00:01+00:00",
            event=Error(message="provider rejected model"),
        )
    )

    assert messages[0].startswith("planner")
    assert "start" in messages[0]
    assert "planning complete" in messages[1]
    assert "provider rejected model" in messages[2]
