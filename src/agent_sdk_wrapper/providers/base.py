"""The provider adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from ..events import AgentEvent
from ..request import RunRequest


class ProviderAdapter(ABC):
    """Translates one backend SDK into a stream of normalized events.

    Implementations are thin: build the backend's native request from a
    :class:`~agent_sdk_wrapper.request.RunRequest`, drive its async stream, and
    ``yield`` normalized :data:`~agent_sdk_wrapper.events.AgentEvent` values. Raise
    :class:`~agent_sdk_wrapper.errors.TransientError` for retryable failures and
    :class:`~agent_sdk_wrapper.errors.ProviderNotAvailableError` when the backend
    can't run at all; the unified runner handles retries and event framing.
    """

    name: str

    def ensure_available(self) -> None:  # noqa: B027
        """Raise ProviderNotAvailableError if the backend can't be used."""

    def validate_request(self, req: RunRequest) -> None:  # noqa: B027
        """Raise ConfigError for provider-specific request options."""

    @abstractmethod
    def stream(self, req: RunRequest) -> AsyncIterator[AgentEvent]:
        """Yield normalized events for ``req``. An async generator."""
        raise NotImplementedError
