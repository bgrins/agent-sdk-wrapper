"""Helpers for structured output: build a JSON schema from a type, and
validate a raw value back into that type.

Accepts Pydantic models, dataclasses, and TypedDicts — anything Pydantic's
``TypeAdapter`` understands.
"""

from __future__ import annotations

from typing import Any

from .errors import AgentSdkWrapperError, ConfigError


def json_schema_of_type(tp: type) -> dict[str, Any]:
    if hasattr(tp, "model_json_schema"):  # pydantic BaseModel
        return tp.model_json_schema()
    try:
        from pydantic import TypeAdapter

        return TypeAdapter(tp).json_schema()
    except Exception as exc:  # pragma: no cover - surfaced as config error
        raise ConfigError(f"cannot derive a JSON schema from {tp!r}: {exc}", cause=exc) from exc


def validate_output(tp: type | None, value: Any) -> Any:
    if tp is None or value is None:
        return value
    if hasattr(tp, "model_validate"):  # pydantic BaseModel
        try:
            return tp.model_validate(value)
        except Exception as exc:
            raise AgentSdkWrapperError(
                f"structured output did not match {tp!r}: {exc}", cause=exc
            ) from exc
    try:
        from pydantic import TypeAdapter

        return TypeAdapter(tp).validate_python(value)
    except Exception as exc:
        raise AgentSdkWrapperError(
            f"structured output did not match {tp!r}: {exc}", cause=exc
        ) from exc
