"""Provider adapters: translate each backend SDK's stream into the unified
event model defined in :mod:`agent_sdk_wrapper.events`.
"""

from __future__ import annotations

from typing import Any

from ..errors import ConfigError
from ..request import Provider
from .base import ProviderAdapter


def build_provider(provider: Provider, **options: Any) -> ProviderAdapter:
    if provider == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(**options)
    if provider == "openai":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider(**options)
    raise ConfigError(f"unknown provider {provider!r}; expected 'anthropic' or 'openai'")


__all__ = ["ProviderAdapter", "build_provider"]
