"""Write trace, result, manifest and native-event files.

``trace.jsonl`` contains one normalized ``EventEnvelope`` per line.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import uuid
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any

from .events import RunResult, _jsonable, utcnow_iso
from .logging import JSON_TEXT_ERRORS, get_logger

ARTIFACT_SCHEMA_VERSION = 1
TRACE_FORMAT = "agent-sdk-wrapper.event-envelope-jsonl.v1"


@dataclasses.dataclass(frozen=True)
class ProviderEventEnvelope:
    """A provider-native SDK message observed before normalized mapping."""

    sequence: int
    timestamp: str
    provider: str
    class_name: str
    message: Any
    raw: Any = dataclasses.field(default=None, repr=False, compare=False)
    run_id: str | None = None

    @classmethod
    def from_message(
        cls,
        provider: str,
        sequence: int,
        message: Any,
        *,
        run_id: str | None = None,
    ) -> ProviderEventEnvelope:
        typ = type(message)
        return cls(
            sequence=sequence,
            timestamp=utcnow_iso(),
            provider=provider,
            class_name=f"{typ.__module__}.{typ.__qualname__}",
            message=_provider_jsonable(message),
            raw=message,
            run_id=run_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "provider": self.provider,
            "class": self.class_name,
            "message": self.message,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, separators=(",", ":")
        )


ProviderEventCallback = Callable[[ProviderEventEnvelope], None]


def normalize_artifacts_dir(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path


def trace_file_for(artifacts_dir: Path) -> Path:
    return artifacts_dir / "trace.jsonl"


def result_file_for(artifacts_dir: Path) -> Path:
    return artifacts_dir / "result.json"


def manifest_file_for(artifacts_dir: Path) -> Path:
    return artifacts_dir / "manifest.json"


def sdk_dir_for(artifacts_dir: str | Path) -> Path:
    path = Path(artifacts_dir) / "sdk"
    path.mkdir(parents=True, exist_ok=True)
    return path


def provider_events_file_for(artifacts_dir: str | Path) -> Path:
    path = Path(artifacts_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path / "provider-events.jsonl"


class ProviderEventLogger:
    """Append provider-native SDK messages before normalized adapter mapping.

    The runner clears the file at run start.
    """

    def __init__(
        self,
        provider: str,
        artifacts_dir: str | Path | None,
        on_provider_event: ProviderEventCallback | None = None,
        *,
        run_id: str | None = None,
    ) -> None:
        self.provider = provider
        self.path = (
            provider_events_file_for(artifacts_dir)
            if artifacts_dir is not None
            else None
        )
        self.on_provider_event = on_provider_event
        self.run_id = run_id
        self.sequence = 0

    def write(self, message: Any) -> None:
        if self.path is None and self.on_provider_event is None:
            return
        envelope = ProviderEventEnvelope.from_message(
            self.provider,
            self.sequence,
            message,
            run_id=self.run_id,
        )
        self.sequence += 1
        if self.path is not None:
            with self.path.open("a", encoding="utf-8", errors=JSON_TEXT_ERRORS) as out:
                out.write(envelope.to_json() + "\n")
        if self.on_provider_event is not None:
            try:
                self.on_provider_event(envelope)
            except Exception:
                get_logger().exception(
                    "on_provider_event callback raised; continuing run"
                )


def clear_stale_artifacts(artifacts_dir: Path) -> None:
    """Remove files from a previous run that the new run would not overwrite first."""
    result_file_for(artifacts_dir).unlink(missing_ok=True)
    provider_events_file_for(artifacts_dir).unlink(missing_ok=True)
    sdk_dir = artifacts_dir / "sdk"
    if sdk_dir.is_dir() and not sdk_dir.is_symlink():
        shutil.rmtree(sdk_dir)
    else:
        sdk_dir.unlink(missing_ok=True)


def collect_side_files(artifacts_dir: Path) -> dict[str, Path]:
    """Return provider-specific side files for manifest discovery."""
    files: dict[str, Path] = {}
    provider_events = artifacts_dir / "provider-events.jsonl"
    if provider_events.exists():
        files["provider_events"] = provider_events

    sdk_dir = artifacts_dir / "sdk"
    if not sdk_dir.exists():
        return files
    for path in sorted(sdk_dir.rglob("*")):
        if path.is_file():
            rel = path.relative_to(artifacts_dir).as_posix()
            files[rel.replace("/", ".")] = path
    return files


def write_result_artifact(artifacts_dir: Path, result: RunResult) -> Path:
    path = result_file_for(artifacts_dir)
    _write_json_atomic(path, result.to_dict())
    return path


def write_manifest(
    artifacts_dir: Path,
    *,
    run_id: str,
    provider: str,
    model: str | None,
    status: str,
    trace_file: str | Path | None,
    ended_reason: str | None = None,
    result_file: str | Path | None = None,
    duration_ms: int | None = None,
    error: str | None = None,
    error_type: str | None = None,
    extra_files: dict[str, str | Path] | None = None,
) -> Path:
    files: dict[str, str] = {}
    if trace_file is not None:
        files["trace"] = _relpath(trace_file, artifacts_dir)
    if result_file is not None:
        files["result"] = _relpath(result_file, artifacts_dir)
    for key, path in (extra_files or {}).items():
        files[key] = _relpath(path, artifacts_dir)

    manifest: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "trace_format": TRACE_FORMAT,
        "run_id": run_id,
        "provider": provider,
        "model": model,
        "status": status,
        "ended_reason": ended_reason,
        "updated_at": utcnow_iso(),
        "duration_ms": duration_ms,
        "error": error,
        "error_type": error_type,
        "files": files,
    }
    path = manifest_file_for(artifacts_dir)
    _write_json_atomic(path, _jsonable(manifest))
    return path


def _write_json_atomic(path: Path, payload: Any) -> None:
    """Replace ``path`` in one step so readers never see a partial file."""
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("x", encoding="utf-8", errors=JSON_TEXT_ERRORS) as out:
            out.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _relpath(path: str | Path, base: Path) -> str:
    path = Path(path)
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def _provider_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return _provider_jsonable(value.value)
    if hasattr(value, "model_dump"):
        try:
            return _provider_jsonable(value.model_dump(mode="json", by_alias=True))
        except Exception:
            try:
                return _provider_jsonable(value.model_dump())
            except Exception:
                return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        try:
            return _provider_jsonable(dataclasses.asdict(value))
        except Exception:
            return str(value)
    if hasattr(value, "__dict__"):
        try:
            return {
                str(k): _provider_jsonable(v)
                for k, v in vars(value).items()
                if not str(k).startswith("_")
            }
        except Exception:
            return str(value)
    if isinstance(value, dict):
        return {str(k): _provider_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_provider_jsonable(v) for v in value]
    return str(value)
