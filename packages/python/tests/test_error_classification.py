"""The shared classification cases; TypeScript's core tests replay the same file."""

import json
from pathlib import Path

import pytest

from agent_sdk_wrapper.classify import TRANSIENT, classify

ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "docs" / "schemas").is_dir()
)
CASES = json.loads(
    (ROOT / "docs/fixtures/error-classification-v1.json").read_text(encoding="utf-8")
)["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["message"][:60])
def test_shared_error_classification(case):
    assert classify(case["message"], case.get("status")) == case["error_type"]


# What TypeScript's classify returns for non-ASCII text.
@pytest.mark.parametrize(
    ("message", "error_type"),
    [
        ("HTTP ٥٠٠ upstream", None),
        ("status: ٤٠١", None),
        ("Übilling failed", "billing_error"),
        ("model foo\r not found", None),
        ("model foo  not found", None),
        ("model: gpt-x not found", None),
        ("model x not found", "model_not_found"),
        ("status:﻿503", TRANSIENT),
        ("status: 503", TRANSIENT),
        ("status　503", TRANSIENT),
        ("status\x1c503", None),
        ("İnvalid api key", None),
        ("overloadedſ", TRANSIENT),
    ],
)
def test_non_ascii_text_classifies_as_in_typescript(message, error_type):
    assert classify(message) == error_type
