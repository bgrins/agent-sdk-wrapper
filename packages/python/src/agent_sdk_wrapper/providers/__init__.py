"""Map native SDK streams to normalized events."""

from __future__ import annotations

import inspect
from typing import Any

from ..errors import ConfigError
from ..request import Provider
from .base import ProviderAdapter


def build_provider(provider: Provider, **options: Any) -> ProviderAdapter:
    if provider == "anthropic":
        from .anthropic_provider import AnthropicProvider as adapter
    elif provider == "openai":
        from .openai_provider import OpenAIProvider as adapter
    else:
        raise ConfigError(f"unknown provider {provider!r}; expected 'anthropic' or 'openai'")
    try:
        inspect.signature(adapter).bind(**options)
    except TypeError as exc:
        raise ConfigError(f"invalid provider_options for {provider!r}: {exc}") from None
    return adapter(**options)


__all__ = ["ProviderAdapter", "build_provider"]
