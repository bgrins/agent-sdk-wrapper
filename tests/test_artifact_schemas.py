"""JSON Schema coverage for trace and artifact files."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

from agent_sdk_wrapper import (
    Agent,
    AgentUpdated,
    Error,
    EventEnvelope,
    RunFinished,
    RunStarted,
    RunStatus,
    SessionInfo,
    StructuredOutput,
    Text,
    Thinking,
    TokenUsage,
    ToolCall,
    ToolResult,
    Usage,
)
from agent_sdk_wrapper.events import WarningEvent
from agent_sdk_wrapper.providers import base
from agent_sdk_wrapper.providers import openai_provider as op_mod

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "docs" / "schemas"
TRACE_FIXTURES = ROOT / "tests" / "fixtures" / "traces"


def load_schema(name: str) -> dict[str, Any]:
    return json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))


def schema_registry() -> Registry:
    resources: list[tuple[str, Resource[dict[str, Any]]]] = []
    for path in sorted(SCHEMA_DIR.glob("*.schema.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        resource = Resource.from_contents(schema, default_specification=DRAFT202012)
        resources.append((schema["$id"], resource))
        resources.append((path.as_uri(), resource))
    return Registry().with_resources(resources)


def validator(name: str) -> Draft202012Validator:
    schema = load_schema(name)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(
        schema,
        registry=schema_registry(),
        format_checker=FormatChecker(),
    )


def test_schema_files_are_valid_json_schema() -> None:
    for path in sorted(SCHEMA_DIR.glob("*.schema.json")):
        Draft202012Validator.check_schema(json.loads(path.read_text(encoding="utf-8")))


def test_each_event_type_validates_against_trace_schema() -> None:
    trace_validator = validator("agent-sdk-wrapper.event-envelope-jsonl.v1.schema.json")
    events = [
        RunStarted(provider="openai", model="gpt-5", cwd="/workspace"),
        Text(text="hello", raw={"method": "item/agentMessage/delta"}),
        Thinking(text="plan"),
        ToolCall(id="tool-1", name="repo.read_file", input={"path": "app.py"}),
        ToolResult(id="tool-1", output="contents", is_error=False),
        AgentUpdated(name="reviewer"),
        Usage(
            usage=TokenUsage(
                requests=1,
                input_tokens=10,
                output_tokens=2,
                total_tokens=12,
                cache_read_tokens=3,
                reasoning_output_tokens=1,
            ),
            cost_usd=0.01,
            raw={"total": {"inputTokens": 10}},
        ),
        SessionInfo(id="sess-1"),
        StructuredOutput(value={"ok": True}),
        WarningEvent(message="retrying"),
        Error(message="missing runtime", error_type="ProviderNotAvailableError"),
        RunFinished(status=RunStatus.SUCCESS, duration_ms=12),
    ]

    for sequence, event in enumerate(events):
        envelope = EventEnvelope(
            run_id="schema-event-types",
            sequence=sequence,
            timestamp="2026-05-31T00:00:00+00:00",
            event=event,
        )
        trace_validator.validate(envelope.to_dict())


def test_golden_trace_fixtures_validate_against_trace_schema() -> None:
    trace_validator = validator("agent-sdk-wrapper.event-envelope-jsonl.v1.schema.json")

    for path in sorted(TRACE_FIXTURES.glob("*.trace.jsonl")):
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines, f"{path} should not be empty"
        for expected_sequence, line in enumerate(lines):
            payload = json.loads(line)
            assert payload["sequence"] == expected_sequence
            trace_validator.validate(payload)


def test_generated_artifacts_validate_schemas(monkeypatch, tmp_path: Path) -> None:
    class FakeProvider(base.ProviderAdapter):
        name = "openai"

        async def stream(self, req):  # type: ignore[override]
            yield Text(text="artifact schema")
            yield ToolCall(id="tool-1", name="repo.read_file", input={"path": "app.py"})
            yield ToolResult(id="tool-1", output="contents", is_error=False)
            yield SessionInfo(id="sess-schema")
            yield Usage(usage=TokenUsage(input_tokens=4, output_tokens=2, total_tokens=6))
            yield StructuredOutput(value={"ok": True})

    monkeypatch.setattr(op_mod, "OpenAIProvider", FakeProvider)

    artifacts_dir = tmp_path / "artifacts"
    result = asyncio.run(Agent(provider="openai", artifacts_dir=artifacts_dir).run("ignored"))

    assert result.ok
    trace_validator = validator("agent-sdk-wrapper.event-envelope-jsonl.v1.schema.json")
    manifest_validator = validator("agent-sdk-wrapper.manifest.v1.schema.json")
    result_validator = validator("agent-sdk-wrapper.run-result.v1.schema.json")

    for line in (artifacts_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines():
        trace_validator.validate(json.loads(line))
    manifest_validator.validate(
        json.loads((artifacts_dir / "manifest.json").read_text(encoding="utf-8"))
    )
    result_validator.validate(
        json.loads((artifacts_dir / "result.json").read_text(encoding="utf-8"))
    )
