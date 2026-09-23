"""The provider adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from ..events import AgentEvent
from ..request import RunRequest


class ProviderAdapter(ABC):
    """Translate a ``RunRequest`` into native calls and normalized ``AgentEvent`` values.

    Raise ``ProviderNotAvailableError`` for an unavailable runtime; the runner classifies
    other exceptions by their message. It also handles deadlines and event envelopes.
    """

    name: str

    def ensure_available(self) -> None:  # noqa: B027
        """Raise ProviderNotAvailableError if the backend can't be used."""

    def validate_request(self, req: RunRequest) -> None:  # noqa: B027
        """Raise ConfigError for unsupported request options."""

    def check_credentials(self, req: RunRequest) -> str | None:
        """Explain why ``req.cli_login`` can't be satisfied, or return None."""
        return None

    @abstractmethod
    def stream(self, req: RunRequest) -> AsyncIterator[AgentEvent]:
        """Yield normalized events for ``req``. An async generator."""
        raise NotImplementedError
