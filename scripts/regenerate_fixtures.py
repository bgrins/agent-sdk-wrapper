#!/usr/bin/env python3
"""Regenerate trace fixtures.

Offline fixtures are deterministic and safe to commit. Live fixtures are
integration-only and write timestamped artifacts under ``results/`` by default.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent_sdk_wrapper import (  # noqa: E402
    Agent,
    AgentEvent,
    AgentUpdated,
    ConfigError,
    ProviderNotAvailableError,
    RunRequest,
    SessionInfo,
    StructuredOutput,
    Text,
    Thinking,
    TokenUsage,
    ToolCall,
    ToolResult,
    TransientError,
    Usage,
    normalize_provider,
)
from agent_sdk_wrapper.providers import anthropic_provider as anthropic_mod  # noqa: E402
from agent_sdk_wrapper.providers import openai_provider as openai_mod  # noqa: E402
from agent_sdk_wrapper.providers.base import ProviderAdapter  # noqa: E402

DEFAULT_OFFLINE_DIR = ROOT / "tests" / "fixtures" / "traces"
DEFAULT_LIVE_ROOT = ROOT / "results" / "fixture-runs"
SCHEMA_DIR = ROOT / "docs" / "schemas"
TRACE_SCHEMA_NAME = "agent-sdk-wrapper.event-envelope-jsonl.v1.schema.json"
BASE_TIME = datetime(2026, 5, 31, tzinfo=UTC)
OFFLINE_FIXTURE_NAMES = {
    "provider_error.trace.jsonl",
    "retry.trace.jsonl",
    "stream.trace.jsonl",
    "structured_output.trace.jsonl",
    "success.trace.jsonl",
    "tool_call.trace.jsonl",
}
LIVE_FIXTURE_SPECS = {
    "anthropic": ("live-anthropic-basic.trace.jsonl", 10, 1010),
    "codex": ("live-codex-basic.trace.jsonl", 11, 1011),
}
REDACTED_SECRET = "[REDACTED_SECRET]"
SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("anthropic_api_key", re.compile(r"sk-ant-[A-Za-z0-9_-]{16,}")),
    ("openai_api_key", re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{24,}")),
    ("github_token", re.compile(r"gh[pousr]_[A-Za-z0-9_]{24,}")),
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("bearer_token", re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]{16,}", re.IGNORECASE)),
)
SENSITIVE_KEY_NAMES = {
    "api_key",
    "apikey",
    "authorization",
    "bearer_token",
    "client_secret",
    "credential",
    "id_token",
    "password",
    "refresh_token",
    "secret",
    "token",
    "x_api_key",
}
SAFE_TOKEN_FIELD_NAMES = {
    "cache_read_tokens",
    "cache_write_tokens",
    "cached_input_tokens",
    "completion_tokens",
    "input_tokens",
    "output_tokens",
    "prompt_tokens",
    "reasoning_output_tokens",
    "token_usage",
    "tokenusage",
    "total_tokens",
}


class StaticProvider(ProviderAdapter):
    """Fake provider that yields a fixed event list."""

    def __init__(self, provider: str, events: Sequence[AgentEvent]) -> None:
        self.name = provider
        self.events = list(events)

    async def stream(self, req: RunRequest):
        for event in self.events:
            yield event


class RetryOnceProvider(ProviderAdapter):
    """Fake provider that fails once, then succeeds."""

    name = "openai"

    def __init__(self) -> None:
        self.attempts = 0

    async def stream(self, req: RunRequest):
        self.attempts += 1
        if self.attempts == 1:
            raise TransientError("rate limit")
        yield Text(text="Recovered.")


class UnavailableProvider(ProviderAdapter):
    """Fake provider that behaves like a missing runtime."""

    name = "anthropic"

    async def stream(self, req: RunRequest):
        raise ProviderNotAvailableError("missing runtime")
        yield Text(text="unreachable")


@contextlib.contextmanager
def patched_provider(
    provider: str,
    factory: Callable[[], ProviderAdapter],
) -> Iterator[None]:
    if provider == "anthropic":
        original = anthropic_mod.AnthropicProvider
        anthropic_mod.AnthropicProvider = lambda **_: factory()  # type: ignore[assignment]
        try:
            yield
        finally:
            anthropic_mod.AnthropicProvider = original  # type: ignore[assignment]
        return

    original = openai_mod.OpenAIProvider
    openai_mod.OpenAIProvider = lambda **_: factory()  # type: ignore[assignment]
    try:
        yield
    finally:
        openai_mod.OpenAIProvider = original  # type: ignore[assignment]


@contextlib.contextmanager
def deterministic_retry_delay() -> Iterator[None]:
    from agent_sdk_wrapper import agent as agent_mod

    original = agent_mod._backoff
    agent_mod._backoff = lambda attempt: 0.0  # type: ignore[assignment]
    try:
        yield
    finally:
        agent_mod._backoff = original  # type: ignore[assignment]


async def generate_offline_fixtures(output_dir: Path) -> list[Path]:
    """Generate deterministic trace fixtures into ``output_dir``."""

    output_dir.mkdir(parents=True, exist_ok=True)
    fixtures = [
        (
            "success.trace.jsonl",
            "golden-success",
            0,
            123,
            _run_static_fixture(
                provider="openai",
                model="gpt-5",
                events=[
                    Thinking(text="checked constraints"),
                    Text(text="Done."),
                    Usage(
                        usage=TokenUsage(
                            requests=1,
                            input_tokens=10,
                            output_tokens=2,
                            total_tokens=12,
                        )
                    ),
                ],
            ),
        ),
        (
            "tool_call.trace.jsonl",
            "golden-tool-call",
            1,
            456,
            _run_static_fixture(
                provider="anthropic",
                model="claude-haiku-4-5",
                events=[
                    AgentUpdated(name="reviewer"),
                    ToolCall(id="tool-1", name="repo.read_file", input={"path": "app.py"}),
                    ToolResult(id="tool-1", output="contents", is_error=False),
                    Text(text="Reviewed app.py."),
                ],
            ),
        ),
        (
            "retry.trace.jsonl",
            "golden-retry",
            2,
            789,
            _run_retry_fixture,
        ),
        (
            "provider_error.trace.jsonl",
            "golden-provider-error",
            3,
            12,
            _run_provider_error_fixture,
        ),
        (
            "structured_output.trace.jsonl",
            "golden-structured-output",
            4,
            234,
            _run_static_fixture(
                provider="openai",
                model="gpt-5",
                events=[
                    Text(text='{"ok":true}'),
                    StructuredOutput(value={"ok": True}),
                ],
            ),
        ),
        (
            "stream.trace.jsonl",
            "golden-stream",
            5,
            345,
            _run_stream_fixture,
        ),
    ]

    written: list[Path] = []
    for filename, run_id, minute_offset, duration_ms, generate in fixtures:
        path = output_dir / filename
        await generate(path)
        normalize_trace(
            path,
            run_id=run_id,
            start=BASE_TIME + timedelta(minutes=minute_offset),
            duration_ms=duration_ms,
        )
        validate_trace(path)
        written.append(path)
    return written


def _run_static_fixture(
    *,
    provider: str,
    model: str | None,
    events: Sequence[AgentEvent],
) -> Callable[[Path], Any]:
    async def generate(path: Path) -> None:
        with patched_provider(provider, lambda: StaticProvider(provider, events)):
            agent = Agent(provider=provider, model=model, max_retries=0)
            await agent.run("offline fixture", trace_file=path)

    return generate


async def _run_retry_fixture(path: Path) -> None:
    provider = RetryOnceProvider()
    with deterministic_retry_delay(), patched_provider("openai", lambda: provider):
        agent = Agent(provider="openai", max_retries=1)
        await agent.run("offline retry fixture", trace_file=path)


async def _run_provider_error_fixture(path: Path) -> None:
    with patched_provider("anthropic", UnavailableProvider):
        agent = Agent(provider="anthropic", max_retries=0)
        await agent.run("offline provider error fixture", trace_file=path)


async def _run_stream_fixture(path: Path) -> None:
    events = [
        SessionInfo(id="stream-session"),
        Thinking(text="stream plan"),
        Text(text="Streamed."),
        Usage(usage=TokenUsage(input_tokens=3, output_tokens=1, total_tokens=4)),
    ]
    with patched_provider("openai", lambda: StaticProvider("openai", events)):
        agent = Agent(provider="openai", model="gpt-5-mini", max_retries=0)
        async for _ in agent.stream("offline stream fixture", trace_file=path):
            pass


def normalize_trace(path: Path, *, run_id: str, start: datetime, duration_ms: int) -> None:
    lines: list[str] = []
    for sequence, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        payload = json.loads(line)
        payload["run_id"] = run_id
        payload["sequence"] = sequence
        payload["timestamp"] = (start + timedelta(seconds=sequence)).isoformat()
        event = payload.get("event", {})
        if event.get("type") == "run_finished":
            event["duration_ms"] = duration_ms
        lines.append(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def check_offline_fixtures(expected_dir: Path) -> int:
    with tempfile.TemporaryDirectory(prefix="agent-sdk-wrapper-fixtures-") as tmp:
        generated_dir = Path(tmp)
        generated = await generate_offline_fixtures(generated_dir)
        generated_names = {path.name for path in generated}
        expected_names = {
            path.name
            for path in expected_dir.glob("*.trace.jsonl")
            if path.name in OFFLINE_FIXTURE_NAMES
        }
        failures: list[str] = []

        missing = sorted(generated_names - expected_names)
        extra = sorted(expected_names - generated_names)
        if missing:
            failures.append(f"missing committed fixtures: {', '.join(missing)}")
        if extra:
            failures.append(f"extra committed fixtures: {', '.join(extra)}")

        for name in sorted(generated_names & expected_names):
            expected = (expected_dir / name).read_text(encoding="utf-8")
            actual = (generated_dir / name).read_text(encoding="utf-8")
            if expected != actual:
                failures.append(f"{name} is stale; regenerate fixtures")

        if failures:
            for failure in failures:
                print(failure, file=sys.stderr)
            return 1
    print(f"offline fixtures match {display_path(expected_dir)}")
    return 0


def promote_live_fixtures(source_root: Path, output_dir: Path) -> list[Path]:
    """Promote captured live artifact traces into committed fixture files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    promoted: list[Path] = []
    for label, (filename, minute_offset, duration_ms) in LIVE_FIXTURE_SPECS.items():
        trace = source_root / label / "basic" / "trace.jsonl"
        if not trace.exists():
            continue
        run_id = f"live-{label}-basic"
        output = output_dir / filename
        write_promoted_live_trace(
            trace,
            output,
            run_id=run_id,
            start=BASE_TIME + timedelta(minutes=minute_offset),
            duration_ms=duration_ms,
        )
        validate_trace(output)
        promoted.append(output)
    if not promoted:
        raise RuntimeError(f"no live traces found under {source_root}")
    return promoted


def write_promoted_live_trace(
    source: Path,
    output: Path,
    *,
    run_id: str,
    start: datetime,
    duration_ms: int,
) -> None:
    lines: list[str] = []
    secret_findings: list[str] = []
    for sequence, line in enumerate(source.read_text(encoding="utf-8").splitlines()):
        payload = json.loads(line)
        payload["run_id"] = run_id
        payload["sequence"] = sequence
        payload["timestamp"] = (start + timedelta(seconds=sequence)).isoformat()
        event = payload.get("event", {})
        if event.get("type") == "session_info":
            event["id"] = f"{run_id}-session"
        elif event.get("type") == "usage":
            event.pop("raw", None)
        elif event.get("type") == "run_finished":
            event["duration_ms"] = duration_ms
        payload, findings = scrub_fixture_payload(payload)
        secret_findings.extend(f"line {sequence + 1}: {finding}" for finding in findings)
        lines.append(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    if secret_findings:
        print(
            f"redacted potential fixture secrets from {display_path(source)}",
            file=sys.stderr,
        )
        for finding in secret_findings:
            print(f"  - {finding}", file=sys.stderr)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def generate_live_fixtures(
    output_dir: Path,
    providers: Sequence[str],
    *,
    require_all: bool = False,
    use_generic_model_env: bool = True,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[Path] = []
    skipped: list[str] = []

    for provider in providers:
        if provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
            skipped.append("anthropic (missing ANTHROPIC_API_KEY)")
            continue
        if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
            skipped.append("codex (missing OPENAI_API_KEY)")
            continue

        label = "codex" if provider == "openai" else provider
        model = _live_model_for(provider, use_generic_model_env=use_generic_model_env)
        run_dir = output_dir / label / "basic"
        agent = Agent(
            provider=provider,
            model=model,
            system_prompt="Follow the user's formatting instruction exactly.",
            max_retries=0,
        )
        result = await agent.run(
            "Reply with exactly LIVE_FIXTURE_PONG and nothing else.",
            artifacts_dir=run_dir,
            raise_on_error=True,
        )
        if "LIVE_FIXTURE_PONG" not in result.final_text:
            raise RuntimeError(f"{label} live fixture returned unexpected text")
        trace = run_dir / "trace.jsonl"
        validate_trace(trace, check_secrets=False)
        warn_trace_secret_findings(trace)
        artifacts.append(run_dir)

    if skipped:
        print("Skipped live providers: " + "; ".join(skipped), file=sys.stderr)
    if skipped and require_all:
        raise RuntimeError(
            "missing credentials for selected live fixture providers: "
            + "; ".join(skipped)
        )
    if not artifacts:
        raise RuntimeError("no live fixtures were generated")
    return artifacts


def _live_model_for(provider: str, *, use_generic_model_env: bool = True) -> str | None:
    selected_provider = _provider_from_env() if use_generic_model_env else None
    if provider == "anthropic":
        return (
            os.environ.get("ANTHROPIC_MODEL")
            or (os.environ.get("MODEL") if selected_provider == "anthropic" else None)
            or "claude-haiku-4-5"
        )
    return (
        os.environ.get("OPENAI_MODEL")
        or (os.environ.get("MODEL") if selected_provider == "openai" else None)
        or None
    )


def _provider_from_env() -> str | None:
    try:
        return normalize_provider(os.environ.get("PROVIDER"))
    except ConfigError:
        return None


def selected_live_providers(
    raw: Sequence[str] | None,
    *,
    use_env: bool = True,
) -> list[str]:
    if raw:
        values = raw
    elif use_env and os.environ.get("PROVIDER"):
        values = [os.environ["PROVIDER"]]
    else:
        values = ["anthropic", "openai"]

    providers: list[str] = []
    for value in values:
        provider = normalize_provider(value)
        if provider is not None and provider not in providers:
            providers.append(provider)
    return providers


def default_live_output_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return DEFAULT_LIVE_ROOT / stamp


def display_path(path: Path) -> Path:
    return path.relative_to(ROOT) if path.is_relative_to(ROOT) else path


def validate_trace(path: Path, *, check_secrets: bool = True) -> None:
    validator = trace_validator()
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"{path} is empty")
    for expected_sequence, line in enumerate(lines):
        payload = json.loads(line)
        if payload.get("sequence") != expected_sequence:
            raise ValueError(f"{path} sequence mismatch at line {expected_sequence + 1}")
        validator.validate(payload)
    if check_secrets:
        assert_trace_has_no_secrets(path)


def assert_trace_has_no_secrets(path: Path) -> None:
    findings = trace_secret_findings(path)
    if not findings:
        return
    details = "\n".join(f"- {finding}" for finding in findings)
    raise ValueError(f"potential fixture secrets found in {display_path(path)}:\n{details}")


def warn_trace_secret_findings(path: Path) -> None:
    findings = trace_secret_findings(path)
    if not findings:
        return
    print(
        f"potential secrets found in live trace {display_path(path)}; "
        "promotion will redact matching values",
        file=sys.stderr,
    )
    for finding in findings:
        print(f"  - {finding}", file=sys.stderr)


def trace_secret_findings(path: Path) -> list[str]:
    findings: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        _, line_findings = scrub_fixture_payload(json.loads(line))
        findings.extend(f"line {line_number}: {finding}" for finding in line_findings)
    return findings


def scrub_fixture_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    scrubbed, findings = _scrub_fixture_value(payload, "$")
    if not isinstance(scrubbed, dict):
        raise TypeError("trace payload must remain a JSON object after scrubbing")
    return scrubbed, findings


def _scrub_fixture_value(value: Any, path: str) -> tuple[Any, list[str]]:
    if isinstance(value, dict):
        scrubbed: dict[str, Any] = {}
        findings: list[str] = []
        for key, child in value.items():
            key_path = f"{path}.{key}"
            if _is_sensitive_key(key):
                redacted = _redact_secret_value(child)
                if redacted != child:
                    findings.append(f"{key_path} ({_normalize_key(key)})")
                scrubbed[key] = redacted
                continue
            scrubbed_child, child_findings = _scrub_fixture_value(child, key_path)
            scrubbed[key] = scrubbed_child
            findings.extend(child_findings)
        return scrubbed, findings

    if isinstance(value, list):
        scrubbed_items: list[Any] = []
        findings: list[str] = []
        for index, item in enumerate(value):
            scrubbed_item, item_findings = _scrub_fixture_value(item, f"{path}[{index}]")
            scrubbed_items.append(scrubbed_item)
            findings.extend(item_findings)
        return scrubbed_items, findings

    if isinstance(value, str):
        return _redact_secret_patterns(value, path)

    return value, []


def _redact_secret_patterns(value: str, path: str) -> tuple[str, list[str]]:
    redacted = value
    findings: list[str] = []
    for kind, pattern in SECRET_PATTERNS:
        if pattern.search(redacted):
            redacted = pattern.sub(REDACTED_SECRET, redacted)
            findings.append(f"{path} ({kind})")
    return redacted, findings


def _redact_secret_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _redact_secret_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_redact_secret_value(item) for item in value]
    if value is None or value == "" or value == REDACTED_SECRET:
        return value
    return REDACTED_SECRET


def _is_sensitive_key(key: str) -> bool:
    normalized = _normalize_key(key)
    if normalized in SAFE_TOKEN_FIELD_NAMES:
        return False
    if normalized in SENSITIVE_KEY_NAMES:
        return True
    return normalized.endswith(
        (
            "_api_key",
            "_apikey",
            "_authorization",
            "_bearer_token",
            "_client_secret",
            "_credential",
            "_id_token",
            "_password",
            "_refresh_token",
            "_secret",
            "_token",
        )
    )


def _normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")


def trace_validator():
    try:
        from jsonschema import Draft202012Validator, FormatChecker
        from referencing import Registry, Resource
        from referencing.jsonschema import DRAFT202012
    except ImportError as exc:
        raise RuntimeError("fixture validation requires dev dependencies") from exc

    resources: list[tuple[str, Any]] = []
    for path in sorted(SCHEMA_DIR.glob("*.schema.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        resource = Resource.from_contents(schema, default_specification=DRAFT202012)
        resources.append((schema["$id"], resource))
        resources.append((path.as_uri(), resource))

    schema = json.loads((SCHEMA_DIR / TRACE_SCHEMA_NAME).read_text(encoding="utf-8"))
    return Draft202012Validator(
        schema,
        registry=Registry().with_resources(resources),
        format_checker=FormatChecker(),
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Offline trace fixture directory. With --live, artifact output root. "
            "Defaults to tests/fixtures/traces for offline and results/fixture-runs/<timestamp> "
            "for live."
        ),
    )
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=DEFAULT_OFFLINE_DIR,
        help="Committed trace fixture directory for live promotion.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Generate offline fixtures in a temp directory and compare with committed fixtures.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "Generate live provider artifacts. Requires credentials and should "
            "be run via Docker Compose."
        ),
    )
    parser.add_argument(
        "--provider",
        action="append",
        help="Live provider to run: anthropic, openai, or codex. Repeatable.",
    )
    parser.add_argument(
        "--promote-live",
        action="store_true",
        help="With --live, also promote generated live traces into committed fixtures.",
    )
    parser.add_argument(
        "--promote-live-from",
        type=Path,
        help="Promote traces from an existing live artifact root into committed fixtures.",
    )
    args = parser.parse_args(argv)
    if args.check and args.live:
        parser.error("--check cannot be combined with --live")
    if args.promote_live and not args.live:
        parser.error("--promote-live requires --live")
    if args.promote_live_from is not None and args.live:
        parser.error("--promote-live-from cannot be combined with --live")
    if args.promote_live_from is not None and args.check:
        parser.error("--promote-live-from cannot be combined with --check")
    if args.check and args.output_dir is None:
        args.output_dir = DEFAULT_OFFLINE_DIR
    elif args.output_dir is None:
        args.output_dir = default_live_output_dir() if args.live else DEFAULT_OFFLINE_DIR
    return args


async def async_main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    if args.promote_live_from is not None:
        promoted = promote_live_fixtures(args.promote_live_from, args.fixture_dir)
        for path in promoted:
            print(display_path(path))
        return 0
    if args.live:
        use_generic_env = not args.promote_live
        providers = selected_live_providers(args.provider, use_env=use_generic_env)
        artifacts = await generate_live_fixtures(
            args.output_dir,
            providers,
            require_all=args.promote_live,
            use_generic_model_env=use_generic_env,
        )
        for path in artifacts:
            print(display_path(path))
        if args.promote_live:
            promoted = promote_live_fixtures(args.output_dir, args.fixture_dir)
            for path in promoted:
                print(display_path(path))
        return 0
    if args.check:
        return await check_offline_fixtures(args.output_dir)
    written = await generate_offline_fixtures(args.output_dir)
    for path in written:
        print(display_path(path))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return asyncio.run(async_main(sys.argv[1:] if argv is None else argv))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
