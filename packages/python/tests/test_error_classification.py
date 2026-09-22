"""The shared classification cases; TypeScript's core tests replay the same file."""

import json
from pathlib import Path

import pytest

from agent_sdk_wrapper.classify import classify

ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "docs" / "schemas").is_dir()
)
CASES = json.loads(
    (ROOT / "docs/fixtures/error-classification-v1.json").read_text(encoding="utf-8")
)["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["message"][:60])
def test_shared_error_classification(case):
    assert classify(case["message"], case.get("status")) == case["error_type"]
