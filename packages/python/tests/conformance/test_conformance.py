"""Run the shared conformance cases against the real Claude CLI and Codex app-server.

Offline, local mock APIs answer each case. Cases with a ``live`` section also run
against the real APIs with ``AGENT_SDK_WRAPPER_RUN_INTEGRATION=1`` and the
provider's key; they make billed calls.
"""

from __future__ import annotations

import copy
import http.client
import inspect
import json
import os
from typing import Any
from urllib.parse import urlsplit

import pytest
from jsonschema import Draft202012Validator

from agent_sdk_wrapper import Agent
from agent_sdk_wrapper.agent import _ALLOWED_OVERRIDES
from agent_sdk_wrapper.providers.anthropic_provider import AnthropicProvider
from agent_sdk_wrapper.providers.openai_provider import OpenAIProvider

from .mocks import EXHAUSTED, REPEATS, MockClaude, MockCodex
from .runner import (
    LIVE_KEYS,
    ROOT,
    SCHEMA,
    SPEC,
    check_match,
    live_view,
    python_view,
    run_case,
)

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


def _schema_errors(spec: dict[str, Any]) -> list[str]:
    return [
        f"{'/'.join(map(str, error.absolute_path))}: {error.message}"
        for error in Draft202012Validator(SCHEMA).iter_errors(spec)
    ]


def test_spec_matches_its_schema():
    Draft202012Validator.check_schema(SCHEMA)
    assert _schema_errors(SPEC) == []
    ids = [case["id"] for case in CASES]
    assert sorted(ids) == sorted(set(ids)), "duplicate case ids"
    envelope = json.loads(
        (ROOT / "docs/schemas/agent-sdk-wrapper.event-envelope-jsonl.v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    types = [
        envelope["$defs"][branch["$ref"].rsplit("/", 1)[1]]["properties"]["type"]["const"]
        for branch in envelope["properties"]["event"]["oneOf"]
    ]
    assert SCHEMA["$defs"]["event_types"]["items"]["enum"] == types


@pytest.mark.parametrize(
    "typo",
    [
        {"final_txt": "ok"},
        {"request": {"count": 1}},
        {"requests": {"match": [{"path": "model", "equal": "x"}]}},
        {"setup_error": "authentication_failed", "status": "failure"},
    ],
)
def test_schema_rejects_unknown_expectations_in_skipped_cases(typo):
    spec = copy.deepcopy(SPEC)
    case = spec["cases"][0]
    case["languages"] = {"python": "unsupported: probe", "typescript": "unsupported: probe"}
    case["expect"] = {**case["expect"], **typo}
    assert _schema_errors(spec)


REQUEST = {
    "headers": {"x-api-key": "sk"},
    "body": {"model": "m", "stream": True, "thinking": {"type": "enabled", "budget_tokens": 2}},
}


@pytest.mark.parametrize(
    ("match", "requests", "passes"),
    [
        ({"path": "thinking", "equals": {"budget_tokens": 2, "type": "enabled"}}, [REQUEST], True),
        ({"path": "stream", "equals": 1}, [REQUEST], False),
        ({"request": -1, "path": "model", "equals": "m"}, [REQUEST], True),
        ({"request": -2, "path": "model", "equals": "m"}, [REQUEST], False),
        ({"request": 1, "path": "model", "excludes": "x"}, [REQUEST], False),
        ({"path": "model", "excludes": "m"}, [], True),
        ({"header": "x-api-key", "absent": True}, [], True),
        ({"path": "model", "contains": "m"}, [], False),
    ],
)
def test_request_match_semantics(match, requests, passes):
    if passes:
        check_match(match, requests)
    else:
        with pytest.raises(AssertionError):
            check_match(match, requests)


def _post(api: MockClaude | MockCodex, path: str) -> http.client.HTTPResponse:
    url = urlsplit(api.base_url)
    connection = http.client.HTTPConnection(url.hostname, url.port, timeout=10)
    body = json.dumps({"model": "m", "stream": True})
    connection.request("POST", path, body, {"content-type": "application/json"})
    return connection.getresponse()


def test_mock_fails_once_the_last_step_has_repeated():
    api = MockCodex([{"text": "a"}, {"text": "b"}]).start()
    try:
        statuses = [_post(api, "/v1/responses").status for _ in range(2 + REPEATS + 1)]
        response = _post(api, "/v1/responses")
        assert statuses == [200] * (2 + REPEATS) + [400]
        assert EXHAUSTED in response.read().decode()
    finally:
        api.stop()


def test_mock_truncate_drops_a_chunked_stream():
    api = MockClaude([{"truncate": True}]).start()
    try:
        response = _post(api, "/v1/messages")
        assert response.getheader("transfer-encoding") == "chunked"
        with pytest.raises(http.client.IncompleteRead):
            response.read()
    finally:
        api.stop()


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


def _used(view: dict[str, Any]) -> set[str]:
    """Options a case Python runs sets, including its live runs when Python runs them."""

    used = {"provider"}
    sections = [view, *view.get("runs", [])]
    if "live" in view and not view["expect"].get("config_error"):
        sections += [view["live"], *view["live"].get("runs", [])]
    for section in sections:
        for key in ("options", "run_options"):
            options = section.get(key, {})
            used |= set(options)
            used |= {f"provider_options.{name}" for name in options.get("provider_options", {})}
    return used


def test_every_option_has_a_case_or_an_exemption():
    exemptions = SPEC["coverage_exemptions"]["python"]
    views = [view for view in map(python_view, CASES) if not isinstance(view, str)]
    for provider, options in _options().items():
        used = set().union(*(_used(view) for view in views if view["provider"] == provider))
        exempt = {key.split(":", 1)[1] for key in exemptions if key.startswith(f"{provider}:")}
        assert not exempt - options, f"{provider} exemptions name unknown options"
        assert not exempt & used, f"{provider} exemptions name options a case uses"
        assert options - used - exempt == set(), f"{provider} options without a case"
