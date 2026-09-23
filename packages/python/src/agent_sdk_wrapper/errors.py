"""Provider-independent errors."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .events import RunResult


class AgentSdkWrapperError(Exception):
    """Base class for every error this library raises."""

    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.__cause__ = cause


class ConfigError(AgentSdkWrapperError):
    """Invalid configuration — missing key, unknown provider, bad schema."""


class ProviderNotAvailableError(AgentSdkWrapperError):
    """The backend can't run: missing dependency, CLI, or credentials."""


class ProcessTerminatedError(AgentSdkWrapperError):
    """The runtime was killed by a signal. Stop the batch; do not retry in place.

    ``result`` is the failed ``RunResult`` when ``run()`` raised it.
    """

    def __init__(
        self, signal: int, *, message: str | None = None, cause: BaseException | None = None
    ) -> None:
        super().__init__(
            message or f"provider runtime killed by signal {signal}", cause=cause
        )
        self.signal = signal
        self.result: RunResult | None = None


class RunFailedError(AgentSdkWrapperError):
    """``run()`` ended in a non-success state and raise_on_error was set.

    ``result`` is the failed ``RunResult``.
    """

    def __init__(self, result: RunResult) -> None:
        super().__init__(result.error or f"run ended with status {result.status.value}")
        self.result = result
