"""Exception hierarchy.

Provider-specific exceptions (from the Claude Agent SDK or the OpenAI Codex
SDK) are wrapped into these so callers can catch one family regardless of
backend. ``transient`` marks errors worth retrying at the run level.
"""

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
    """The provider runtime was killed by a signal (e.g. SIGTERM, SIGKILL).

    Deliberately not a :class:`TransientError`: an external kill aborts the
    whole process tree, so retrying the run in-place only burns the shutdown
    window. Callers running a batch should treat it as a stop signal for the
    batch, not a per-item failure.
    """

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
