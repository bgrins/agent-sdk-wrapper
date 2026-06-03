"""Shared helpers for example scripts."""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from agent_sdk_wrapper import (
    Provider,
    normalize_provider,
    resolve_provider,
)


def example_config(default_provider: str, default_model: str) -> tuple[Provider, str | None]:
    """Resolve provider/model from env, then fall back to the example defaults."""

    raw_provider = os.environ.get("PROVIDER")
    raw_model = os.environ.get("MODEL")
    model = raw_model.strip() if raw_model is not None and raw_model.strip() else None

    provider = normalize_provider(raw_provider)
    if provider is None and model is not None:
        provider = resolve_provider(None, model)
    if provider is None:
        provider = resolve_provider(default_provider, default_model)

    if model is None:
        model = None if provider == "openai" else default_model
    return provider, model


def example_artifacts_dir(example: str, provider: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return Path("results") / provider / example / timestamp


class ExampleOutput:
    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = artifacts_dir
        self._parts: list[str] = []

    def print(self, *values: object, sep: str = " ", end: str = "\n") -> None:
        self.write(sep.join(str(value) for value in values) + end)

    def write(self, text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()
        self._parts.append(text)

    def save(self) -> Path:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        path = self.artifacts_dir / "stdout.txt"
        path.write_text("".join(self._parts), encoding="utf-8")
        record_manifest_file(self.artifacts_dir, "stdout", path)
        return path


def record_manifest_file(artifacts_dir: Path, key: str, path: Path) -> None:
    manifest_path = artifacts_dir / "manifest.json"
    if not manifest_path.exists():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    manifest.setdefault("files", {})[key] = path.relative_to(artifacts_dir).as_posix()
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
