"""Both implementations replay this language-neutral contract, without requiring Node."""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

from agent_sdk_wrapper import Agent
from agent_sdk_wrapper.testing import event_from_dict, install_fake_providers

ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "docs" / "schemas").is_dir()
)
FIXTURES = json.loads((ROOT / "docs/fixtures/native-twin-v1.json").read_text())


@pytest.mark.parametrize("expected", FIXTURES, ids=lambda item: item["status"])
async def test_native_twin_shared_result(expected, monkeypatch):
    schemas = [
        json.loads(path.read_text()) for path in (ROOT / "docs/schemas").glob("*.schema.json")
    ]
    registry = Registry().with_resources(
        (schema["$id"], Resource.from_contents(schema)) for schema in schemas
    )
    result_schema = next(schema for schema in schemas if "run-result" in schema["$id"])
    Draft202012Validator(result_schema, registry=registry, format_checker=FormatChecker()).validate(
        expected
    )
    events = [event_from_dict(env["event"]) for env in expected["events"][1:-1]]
    install_fake_providers(monkeypatch, events=events)
    result = await Agent(provider=expected["provider"], model=expected["model"]).run(
        expected["events"][0]["event"]["prompt"]
    )
    actual = result.to_dict()
    for key in expected.keys() - {"run_id", "duration_ms", "events"}:
        assert actual[key] == expected[key], key
    for actual_env, expected_env in zip(actual["events"], expected["events"], strict=True):
        actual_event = dict(actual_env["event"])
        expected_event = dict(expected_env["event"])
        actual_event.pop("duration_ms", None)
        expected_event.pop("duration_ms", None)
        assert actual_event == expected_event
