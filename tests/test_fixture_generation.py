"""Fixture regeneration script coverage."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "regenerate_fixtures.py"
SPEC = importlib.util.spec_from_file_location("regenerate_fixtures", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
regenerate_fixtures = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(regenerate_fixtures)
EXPECTED_FIXTURES = {
    "provider_error.trace.jsonl",
    "retry.trace.jsonl",
    "stream.trace.jsonl",
    "structured_output.trace.jsonl",
    "success.trace.jsonl",
    "tool_call.trace.jsonl",
}


def test_regenerate_trace_fixtures_check_matches_committed() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_regenerate_trace_fixtures_writes_expected_files(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output-dir", str(tmp_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert {path.name for path in tmp_path.glob("*.trace.jsonl")} == EXPECTED_FIXTURES


def test_committed_trace_fixtures_do_not_contain_secrets() -> None:
    for path in (ROOT / "tests" / "fixtures" / "traces").glob("*.trace.jsonl"):
        regenerate_fixtures.assert_trace_has_no_secrets(path)


def test_promoted_live_trace_fixtures_are_additive(tmp_path: Path) -> None:
    fixture_dir = tmp_path / "fixtures"
    live_root = tmp_path / "live"
    _write_live_trace(live_root / "anthropic" / "basic" / "trace.jsonl", "anthropic")
    _write_live_trace(live_root / "codex" / "basic" / "trace.jsonl", "openai")

    generate = subprocess.run(
        [sys.executable, str(SCRIPT), "--output-dir", str(fixture_dir)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert generate.returncode == 0, generate.stdout + generate.stderr

    promote = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--promote-live-from",
            str(live_root),
            "--fixture-dir",
            str(fixture_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert promote.returncode == 0, promote.stdout + promote.stderr

    assert (fixture_dir / "live-anthropic-basic.trace.jsonl").exists()
    assert (fixture_dir / "live-codex-basic.trace.jsonl").exists()

    check = subprocess.run(
        [sys.executable, str(SCRIPT), "--check", "--output-dir", str(fixture_dir)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert check.returncode == 0, check.stdout + check.stderr

    promoted = (fixture_dir / "live-anthropic-basic.trace.jsonl").read_text()
    assert "source-session" not in promoted
    assert '"raw"' not in promoted


def test_live_fixture_promotion_redacts_secrets(tmp_path: Path, capsys) -> None:
    live_root = tmp_path / "live"
    fixture_dir = tmp_path / "fixtures"
    secret = "sk-ant-api03-" + ("a" * 32)
    _write_live_trace(
        live_root / "anthropic" / "basic" / "trace.jsonl",
        "anthropic",
        extra_event={
            "type": "tool_call",
            "id": "secret-check",
            "name": "debug",
            "input": {
                "OPENAI_API_KEY": "sk-proj-" + ("b" * 32),
                "authorization": "Bearer " + ("c" * 32),
                "note": f"provider echoed {secret}",
                "input_tokens": 11,
            },
        },
    )

    [promoted] = regenerate_fixtures.promote_live_fixtures(live_root, fixture_dir)

    contents = promoted.read_text(encoding="utf-8")
    assert "sk-ant-api03-" not in contents
    assert "sk-proj-" not in contents
    assert "Bearer " not in contents
    assert regenerate_fixtures.REDACTED_SECRET in contents
    assert '"input_tokens":11' in contents
    regenerate_fixtures.assert_trace_has_no_secrets(promoted)

    captured = capsys.readouterr()
    assert "redacted potential fixture secrets" in captured.err


def test_live_fixture_promotion_selection_ignores_generic_provider_env(monkeypatch) -> None:
    monkeypatch.setenv("PROVIDER", "codex")

    assert regenerate_fixtures.selected_live_providers(None, use_env=False) == [
        "anthropic",
        "openai",
    ]
    assert regenerate_fixtures.selected_live_providers(["codex"], use_env=False) == [
        "openai"
    ]


def test_live_fixture_promotion_models_ignore_generic_model_env(monkeypatch) -> None:
    monkeypatch.setenv("PROVIDER", "codex")
    monkeypatch.setenv("MODEL", "claude-haiku-4-5")

    assert (
        regenerate_fixtures._live_model_for(  # noqa: SLF001
            "openai",
            use_generic_model_env=False,
        )
        is None
    )

    monkeypatch.setenv("PROVIDER", "anthropic")
    monkeypatch.setenv("MODEL", "gpt-5")
    assert regenerate_fixtures._live_model_for(  # noqa: SLF001
        "anthropic",
        use_generic_model_env=False,
    ) == "claude-haiku-4-5"

    monkeypatch.setenv("OPENAI_MODEL", "gpt-5")
    assert regenerate_fixtures._live_model_for(  # noqa: SLF001
        "openai",
        use_generic_model_env=False,
    ) == "gpt-5"


def test_live_fixture_promotion_requires_selected_provider_credentials(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="missing credentials"):
        asyncio.run(
            regenerate_fixtures.generate_live_fixtures(  # type: ignore[attr-defined]
                tmp_path,
                ["anthropic"],
                require_all=True,
            )
        )


def _write_live_trace(
    path: Path,
    provider: str,
    *,
    extra_event: dict[str, object] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        {
            "run_id": "source-run",
            "sequence": 0,
            "timestamp": "2026-06-02T00:00:00+00:00",
            "event": {"type": "run_started", "provider": provider},
        },
        {
            "run_id": "source-run",
            "sequence": 1,
            "timestamp": "2026-06-02T00:00:01+00:00",
            "event": {"type": "session_info", "id": "source-session"},
        },
        {
            "run_id": "source-run",
            "sequence": 2,
            "timestamp": "2026-06-02T00:00:02+00:00",
            "event": {"type": "text", "text": "LIVE_FIXTURE_PONG"},
        },
        *(
            [
                {
                    "run_id": "source-run",
                    "sequence": 3,
                    "timestamp": "2026-06-02T00:00:03+00:00",
                    "event": extra_event,
                }
            ]
            if extra_event is not None
            else []
        ),
        {
            "run_id": "source-run",
            "sequence": 4 if extra_event is not None else 3,
            "timestamp": "2026-06-02T00:00:04+00:00",
            "event": {
                "type": "usage",
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 2,
                    "total_tokens": 3,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                    "reasoning_output_tokens": 0,
                    "requests": 0,
                },
                "raw": {"provider": provider},
            },
        },
        {
            "run_id": "source-run",
            "sequence": 5 if extra_event is not None else 4,
            "timestamp": "2026-06-02T00:00:05+00:00",
            "event": {
                "type": "run_finished",
                "status": "success",
                "duration_ms": 1,
                "ended_reason": "success",
            },
        },
    ]
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
