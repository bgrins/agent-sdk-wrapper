"""Run a structured planner/analyst/verifier/fix-planner/reporter workflow.

The example exercises multi-agent chaining and structured artifacts without a
custom MCP harness or vulnerability-search setup.

    uv run python examples/auditor_style.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from _shared import (
    ExampleOutput,
    example_artifacts_dir,
    example_config,
    record_manifest_file,
)
from pydantic import BaseModel, Field

from agent_sdk_wrapper import Agent, EventEnvelope, McpStdioServer, RunResult, TokenUsage

DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODEL = "claude-haiku-4-5"
EXAMPLE_NAME = "auditor_style"
MCP_SERVER_NAME = "auditor_demo"
MCP_TOOLS = ["read_project_brief", "read_artifact_policy"]
ALLOWED_MCP_TOOLS = [f"mcp__{MCP_SERVER_NAME}__{name}" for name in MCP_TOOLS]
MAX_TURNS_BY_STAGE = {
    "planner": 6,
    "analyst": 8,
    "verifier": 8,
    "fix_planner": 8,
    "reporter": 6,
}
SCOPE_BOUNDARY = (
    "Do not inspect, request, summarize, or rely on .env files or environment "
    "secrets from the project."
)

WORKFLOW_BRIEF = """\
Project brief:

The agent-sdk-wrapper project is adding a local trace viewer, Docker Compose workflows,
timestamped example artifacts, and live integration tests for Anthropic and
OpenAI Codex. The project should remain a small SDK wrapper rather than a
third-party orchestration harness.

Review goals:
- keep examples easy to run through Docker Compose
- keep default tests offline and deterministic
- make artifacts useful for later trace replay
- clarify which behavior belongs to the wrapper versus provider SDKs
- avoid broad product claims that are not covered by tests

Do not perform a security or vulnerability audit. Treat this as release-readiness
workflow planning and review.
"""


class PlanTask(BaseModel):
    id: str = Field(description="Stable short id, such as T1")
    question: str = Field(description="Review question to answer")
    rationale: str = Field(description="Why this task matters")


class WorkflowPlan(BaseModel):
    objective: str
    tasks: list[PlanTask]
    success_criteria: list[str]


class Observation(BaseModel):
    id: str
    summary: str
    evidence: str
    impact: Literal["low", "medium", "high"]
    recommendation: str


class AnalysisReport(BaseModel):
    observations: list[Observation]
    open_questions: list[str]


class VerificationItem(BaseModel):
    observation_id: str
    verdict: Literal["supported", "unsupported", "needs_followup"]
    rationale: str


class VerificationReport(BaseModel):
    items: list[VerificationItem]
    overall_verdict: Literal["ready", "needs_changes", "needs_followup"]


class FixAction(BaseModel):
    id: str
    title: str
    owner_role: str
    details: str
    validation: str


class FixPlan(BaseModel):
    status: Literal["ready", "needs_changes", "blocked"]
    actions: list[FixAction]
    validation_steps: list[str]
    traceability: list[str]


class FinalReport(BaseModel):
    status: Literal["ready", "needs_changes", "needs_followup"]
    summary: str
    fix_plan_status: Literal["ready", "needs_changes", "blocked", "not_run"]
    next_actions: list[str]


STAGE_SYSTEM_PROMPTS = {
    "planner": (
        "You are a workflow planner. Produce a compact structured plan that a "
        "review chain can execute."
    ),
    "analyst": (
        "You are a pragmatic release-readiness analyst. Identify concrete, "
        "non-security observations grounded only in the brief and plan."
    ),
    "verifier": (
        "You are a verifier. Check whether each observation is supported by the "
        "brief and prior structured outputs. Do not invent external facts."
    ),
    "fix_planner": (
        "You are a rollout planner. Convert verified observations into concrete "
        "non-code actions and validation steps. Do not produce a source patch."
    ),
    "reporter": (
        "You are the final reporter. Synthesize the verified work into a short "
        "structured decision and next-action list."
    ),
}


def _json_block(value: BaseModel) -> str:
    return json.dumps(value.model_dump(mode="json"), ensure_ascii=False, indent=2)


def auditor_mcp_servers() -> list[McpStdioServer]:
    server_path = Path(__file__).with_name("auditor_mcp_server.py")
    return [
        McpStdioServer(
            name=MCP_SERVER_NAME,
            command=sys.executable,
            args=[str(server_path)],
            enabled_tools=MCP_TOOLS,
        )
    ]


@dataclass
class WorkflowRecorder:
    artifacts_dir: Path
    live_print: Callable[[str], None] | None = None
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    usage: TokenUsage = field(default_factory=TokenUsage)
    stage_statuses: dict[str, str] = field(default_factory=dict)
    stage_reasons: dict[str, str] = field(default_factory=dict)
    events: int = 0
    tool_calls: int = 0
    tool_errors: int = 0

    @property
    def event_log_path(self) -> Path:
        return self.artifacts_dir / "workflow-events.jsonl"

    def start_stage(
        self,
        stage: str,
        artifacts_dir: Path,
        *,
        mcp_enabled: bool,
    ) -> None:
        self._print(
            f"{stage:<12} start max_turns={MAX_TURNS_BY_STAGE[stage]} "
            f"mcp={'yes' if mcp_enabled else 'no'} artifacts={artifacts_dir}"
        )

    def on_event(self, stage: str):
        def record(envelope: EventEnvelope) -> None:
            event = envelope.event.to_dict()
            event_type = event["type"]
            summary = _event_summary(event)
            self.events += 1
            if event_type == "tool_call":
                self.tool_calls += 1
            elif event_type == "tool_result" and event.get("is_error"):
                self.tool_errors += 1

            self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
            line = {
                "stage": stage,
                "sequence": envelope.sequence,
                "timestamp": envelope.timestamp,
                "type": event_type,
                "summary": summary,
            }
            with self.event_log_path.open("a", encoding="utf-8") as out:
                out.write(json.dumps(line, ensure_ascii=False, separators=(",", ":")) + "\n")
            if _prints_live(event_type):
                self._print(f"{stage:<12} {event_type:<17} {summary}")

        return record

    def record_result(self, stage: str, result: RunResult) -> None:
        self.stage_statuses[stage] = result.status.value
        self.stage_reasons[stage] = result.ended_reason.value
        if result.usage is not None:
            self.usage = self.usage + result.usage

    def to_dict(self) -> dict[str, object]:
        return {
            "started_at": self.started_at,
            "updated_at": datetime.now(UTC).isoformat(),
            "events": self.events,
            "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors,
            "stage_statuses": self.stage_statuses,
            "stage_ended_reasons": self.stage_reasons,
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
                "total_tokens": self.usage.total_tokens,
                "cache_read_tokens": self.usage.cache_read_tokens,
                "cache_write_tokens": self.usage.cache_write_tokens,
                "reasoning_output_tokens": self.usage.reasoning_output_tokens,
                "requests": self.usage.requests,
            },
        }

    def _print(self, message: str) -> None:
        if self.live_print is not None:
            self.live_print(message)


def _event_summary(event: dict[str, object]) -> str:
    typ = str(event["type"])
    if typ in {"text", "thinking"}:
        return _truncate(str(event.get("text") or ""))
    if typ in {"warning", "error"}:
        return _truncate(str(event.get("message") or typ))
    if typ == "tool_call":
        return f"{event.get('name') or 'tool'}"
    if typ == "tool_result":
        return "tool error" if event.get("is_error") else "tool result"
    if typ == "usage":
        usage = event.get("usage")
        if isinstance(usage, dict):
            return f"{usage.get('total_tokens', 0)} tokens"
    if typ == "run_finished":
        return str(event.get("ended_reason") or event.get("status") or "")
    return typ


def _prints_live(event_type: str) -> bool:
    return event_type in {
        "session_info",
        "thinking",
        "text",
        "tool_call",
        "tool_result",
        "structured_output",
        "usage",
        "warning",
        "error",
        "run_finished",
    }


def _truncate(text: str, limit: int = 180) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def planner_prompt() -> str:
    return f"""\
Create a release-readiness review plan for this brief.

{SCOPE_BOUNDARY}

{WORKFLOW_BRIEF}

Return only structured output matching the schema.
"""


def analyst_prompt(plan: WorkflowPlan) -> str:
    return f"""\
Analyze the brief using the plan. Do not look for vulnerabilities and do not
claim repository access. Focus on product/workflow readiness. If MCP tools are
available, call read_project_brief and read_artifact_policy before writing the
structured output.

{SCOPE_BOUNDARY}

Brief:
{WORKFLOW_BRIEF}

Plan:
{_json_block(plan)}

Return only structured output matching the schema.
"""


def verifier_prompt(plan: WorkflowPlan, analysis: AnalysisReport) -> str:
    return f"""\
Verify that each observation is grounded in the supplied brief and plan.
Mark speculative observations as unsupported or needs_followup. If MCP tools are
available, call read_artifact_policy to check trace/artifact claims.

{SCOPE_BOUNDARY}

Brief:
{WORKFLOW_BRIEF}

Plan:
{_json_block(plan)}

Analysis:
{_json_block(analysis)}

Return only structured output matching the schema.
"""


def fix_planner_prompt(
    plan: WorkflowPlan,
    analysis: AnalysisReport,
    verification: VerificationReport,
) -> str:
    return f"""\
Create a non-code fix plan for the verified observations. This stage simulates
the downstream patch-agent branch, but it must produce operational actions,
documentation updates, and validation steps rather than a source patch.

If MCP tools are available, call read_project_brief and read_artifact_policy
before writing the structured output.

{SCOPE_BOUNDARY}

Plan:
{_json_block(plan)}

Analysis:
{_json_block(analysis)}

Verification:
{_json_block(verification)}

Return only structured output matching the schema.
"""


def reporter_prompt(
    plan: WorkflowPlan,
    analysis: AnalysisReport,
    verification: VerificationReport,
    fix_plan: FixPlan | None,
) -> str:
    fix_plan_text = (
        _json_block(fix_plan)
        if fix_plan is not None
        else json.dumps({"status": "not_run"}, indent=2)
    )
    return f"""\
Prepare the final review summary from the verified chain.

{SCOPE_BOUNDARY}

Plan:
{_json_block(plan)}

Analysis:
{_json_block(analysis)}

Verification:
{_json_block(verification)}

Fix plan:
{fix_plan_text}

Return only structured output matching the schema.
"""


def _structured(result: RunResult, schema: type[BaseModel]) -> BaseModel:
    if isinstance(result.structured_output, schema):
        return result.structured_output
    return schema.model_validate(result.structured_output)


def write_json_artifact(artifacts_dir: Path, key: str, value: BaseModel) -> Path:
    path = artifacts_dir / f"{key}.json"
    path.write_text(_json_block(value) + "\n", encoding="utf-8")
    record_manifest_file(artifacts_dir, key, path)
    return path


def write_json_value_artifact(artifacts_dir: Path, key: str, value: object) -> Path:
    path = artifacts_dir / f"{key}.json"
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    record_manifest_file(artifacts_dir, key, path)
    return path


def write_context_artifact(
    artifacts_dir: Path,
    *,
    plan: WorkflowPlan,
    analysis: AnalysisReport,
    verification: VerificationReport,
    fix_plan: FixPlan | None,
    recorder: WorkflowRecorder,
) -> Path:
    fix_status = fix_plan.status if fix_plan is not None else "not_run"
    lines = [
        "# Auditor Demo Context",
        "",
        f"Objective: {plan.objective}",
        f"Verification: {verification.overall_verdict}",
        f"Fix plan: {fix_status}",
        (
            "Usage: "
            f"{recorder.usage.input_tokens} input, "
            f"{recorder.usage.output_tokens} output, "
            f"{recorder.usage.total_tokens} total tokens"
        ),
        f"Tool calls: {recorder.tool_calls} ({recorder.tool_errors} errors)",
        "",
        "## Observations",
    ]
    for observation in analysis.observations:
        lines.append(f"- {observation.id}: {observation.summary}")
    if fix_plan is not None:
        lines.extend(["", "## Planned Actions"])
        for action in fix_plan.actions:
            lines.append(f"- {action.id}: {action.title} ({action.owner_role})")
    path = artifacts_dir / "context.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    record_manifest_file(artifacts_dir, "context", path)
    return path


async def run_stage(
    *,
    provider: str,
    model: str | None,
    stage: str,
    prompt: str,
    output_schema: type[BaseModel],
    artifacts_dir: Path,
    recorder: WorkflowRecorder,
    mcp_servers: list[McpStdioServer] | None = None,
) -> tuple[RunResult, BaseModel]:
    recorder.start_stage(stage, artifacts_dir, mcp_enabled=mcp_servers is not None)
    agent = Agent(
        provider=provider,
        model=model,
        system_prompt=STAGE_SYSTEM_PROMPTS[stage],
        output_schema=output_schema,
        artifacts_dir=artifacts_dir,
        max_turns=MAX_TURNS_BY_STAGE[stage],
        mcp_servers=mcp_servers,
        allowed_tools=ALLOWED_MCP_TOOLS if mcp_servers else None,
        on_event=recorder.on_event(stage),
        include_events_in_result=False,
    )
    result = await agent.run(prompt)
    recorder.record_result(stage, result)
    if not result.ok:
        message = result.error or result.final_text or result.ended_reason.value
        raise RuntimeError(
            f"{stage} failed with {result.ended_reason.value}: {message}\n"
            f"Artifacts: {result.artifacts_dir}"
        )
    if result.structured_output is None:
        raise RuntimeError(
            f"{stage} did not produce structured output for {output_schema.__name__}.\n"
            f"Artifacts: {result.artifacts_dir}"
        )
    return result, _structured(result, output_schema)


async def main() -> None:
    provider, model = example_config(DEFAULT_PROVIDER, DEFAULT_MODEL)
    run_dir = example_artifacts_dir(EXAMPLE_NAME, provider)
    artifacts_dir = run_dir / "artifacts"
    out = ExampleOutput(artifacts_dir)
    recorder = WorkflowRecorder(artifacts_dir, live_print=out.print)
    mcp_servers = auditor_mcp_servers()

    try:
        plan_result, plan_value = await run_stage(
            provider=provider,
            model=model,
            stage="planner",
            prompt=planner_prompt(),
            output_schema=WorkflowPlan,
            artifacts_dir=artifacts_dir,
            recorder=recorder,
        )
        plan = WorkflowPlan.model_validate(plan_value)
        write_json_artifact(artifacts_dir, "plan", plan)
        record_manifest_file(artifacts_dir, "workflow_events", recorder.event_log_path)
        out.print(f"planner_status : {plan_result.status.value}")
        out.print(f"artifacts      : {plan_result.artifacts_dir}")

        analysis_dir = artifacts_dir / "analysis"
        analysis_result, analysis_value = await run_stage(
            provider=provider,
            model=model,
            stage="analyst",
            prompt=analyst_prompt(plan),
            output_schema=AnalysisReport,
            artifacts_dir=analysis_dir,
            recorder=recorder,
            mcp_servers=mcp_servers,
        )
        analysis = AnalysisReport.model_validate(analysis_value)
        write_json_artifact(artifacts_dir, "analysis", analysis)
        record_manifest_file(artifacts_dir, "analysis_manifest", analysis_dir / "manifest.json")
        out.print(f"analysis_status: {analysis_result.status.value}")

        verification_dir = artifacts_dir / "verification"
        verification_result, verification_value = await run_stage(
            provider=provider,
            model=model,
            stage="verifier",
            prompt=verifier_prompt(plan, analysis),
            output_schema=VerificationReport,
            artifacts_dir=verification_dir,
            recorder=recorder,
            mcp_servers=mcp_servers,
        )
        verification = VerificationReport.model_validate(verification_value)
        write_json_artifact(artifacts_dir, "verification", verification)
        record_manifest_file(
            artifacts_dir,
            "verification_manifest",
            verification_dir / "manifest.json",
        )
        out.print(f"verifier_status: {verification_result.status.value}")

        fix_plan: FixPlan | None = None
        if verification.overall_verdict == "ready":
            fix_dir = artifacts_dir / "fix-plan"
            fix_result, fix_value = await run_stage(
                provider=provider,
                model=model,
                stage="fix_planner",
                prompt=fix_planner_prompt(plan, analysis, verification),
                output_schema=FixPlan,
                artifacts_dir=fix_dir,
                recorder=recorder,
                mcp_servers=mcp_servers,
            )
            fix_plan = FixPlan.model_validate(fix_value)
            write_json_artifact(artifacts_dir, "fix_plan", fix_plan)
            record_manifest_file(artifacts_dir, "fix_plan_manifest", fix_dir / "manifest.json")
            out.print(f"fix_status     : {fix_result.status.value}")
        else:
            out.print(f"fix_status     : skipped ({verification.overall_verdict})")

        report_dir = artifacts_dir / "report"
        report_result, report_value = await run_stage(
            provider=provider,
            model=model,
            stage="reporter",
            prompt=reporter_prompt(plan, analysis, verification, fix_plan),
            output_schema=FinalReport,
            artifacts_dir=report_dir,
            recorder=recorder,
        )
        report = FinalReport.model_validate(report_value)
        write_json_artifact(artifacts_dir, "report", report)
        record_manifest_file(artifacts_dir, "report_manifest", report_dir / "manifest.json")
        write_json_value_artifact(artifacts_dir, "workflow_stats", recorder.to_dict())
        write_context_artifact(
            artifacts_dir,
            plan=plan,
            analysis=analysis,
            verification=verification,
            fix_plan=fix_plan,
            recorder=recorder,
        )
        out.print(f"report_status  : {report_result.status.value}")
        out.print("--- final")
        out.print(f"status: {report.status}")
        out.print(f"fix_plan_status: {report.fix_plan_status}")
        out.print(report.summary)
        out.print("next_actions:")
        for action in report.next_actions:
            out.print(f"- {action}")
    finally:
        out.save()


if __name__ == "__main__":
    asyncio.run(main())
