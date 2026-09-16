"""Provider-independent errors. ``transient`` marks retryable failures."""

from __future__ import annotations


class AgentSdkWrapperError(Exception):
    """Base class for every error this library raises."""

    transient: bool = False

    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.__cause__ = cause


class ConfigError(AgentSdkWrapperError):
    """Invalid configuration — missing key, unknown provider, bad schema."""


class ProviderNotAvailableError(AgentSdkWrapperError):
    """The backend can't run: missing dependency, CLI, or credentials."""


class TransientError(AgentSdkWrapperError):
    """A retryable failure (rate limit, timeout, transient upstream 5xx)."""

    transient = True


class ProcessTerminatedError(AgentSdkWrapperError):
    """The runtime was killed by a signal. Stop the batch; do not retry in place."""

    def __init__(
        self, signal: int, *, message: str | None = None, cause: BaseException | None = None
    ) -> None:
        super().__init__(
            message or f"provider runtime killed by signal {signal}", cause=cause
        )
        self.signal = signal


class RunFailedError(AgentSdkWrapperError):
    """The agent run completed in a non-success state and raise_on_error was set."""

    def __init__(self, message: str, *, status: str, cause: BaseException | None = None) -> None:
        super().__init__(message, cause=cause)
        self.status = status
