"""The unified request surface that both provider adapters consume."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from .artifacts import ProviderEventCallback
from .errors import ConfigError
from .mcp import McpServer

Provider = Literal["anthropic", "openai"]
PROVIDERS: tuple[Provider, ...] = ("anthropic", "openai")
ProviderInput = str | None
BuiltinTools = Literal["none"] | list[str]
BuiltinToolsInput = Literal["none"] | Sequence[str] | None
INHERIT_MODEL = "inherit"
Effort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]

_PROVIDER_ALIASES: dict[str, Provider] = {
    "anthropic": "anthropic",
    "openai": "openai",
    "codex": "openai",
}
_OPENAI_MODEL_PREFIXES = ("gpt-", "o1", "o3", "o4", "o5", "codex", "chatgpt-")
_ANTHROPIC_MODEL_PREFIXES = ("claude",)
_ANTHROPIC_EFFORTS = {"low", "medium", "high", "xhigh", "max"}
_OPENAI_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}


def normalize_provider(provider: ProviderInput) -> Provider | None:
    """Normalize user-facing provider names into canonical adapter names."""

    if provider is None:
        return None
    if not isinstance(provider, str):
        raise ConfigError(
            f"unknown provider {provider!r}; expected one of {PROVIDERS} or 'codex'"
        )
    normalized = provider.strip().lower()
    if normalized == "":
        return None
    if normalized in _PROVIDER_ALIASES:
        return _PROVIDER_ALIASES[normalized]
    raise ConfigError(
        f"unknown provider {provider!r}; expected one of {PROVIDERS} or 'codex'"
    )


def parse_model_spec(model: str | None) -> tuple[Provider | None, str | None]:
    """Parse optional ``provider:model`` syntax without inferring bare names."""

    if model is None:
        return None, None
    if not isinstance(model, str):
        raise ConfigError(f"model must be a string or None, got {type(model).__name__}")
    spec = model.strip()
    if spec == "":
        return None, None

    prefix, sep, name = spec.partition(":")
    if not sep:
        return None, spec
    provider = normalize_provider(prefix)
    model_name = name.strip()
    if provider is None or model_name == "":
        raise ConfigError(f"invalid model spec {model!r}; expected 'provider:model'")
    return provider, model_name


def infer_provider_from_model(model: str | None) -> Provider | None:
    """Infer the provider for common Anthropic and OpenAI model names."""

    spec_provider, bare_model = parse_model_spec(model)
    if spec_provider is not None:
        return spec_provider
    if bare_model is None:
        return None
    normalized = bare_model.strip().lower()
    if normalized == "":
        return None

    if normalized.startswith(_ANTHROPIC_MODEL_PREFIXES):
        return "anthropic"
    if normalized.startswith(_OPENAI_MODEL_PREFIXES):
        return "openai"
    return None


def resolve_provider(provider: ProviderInput, model: str | None = None) -> Provider:
    """Resolve an explicit provider or infer one from a model name."""

    spec_provider, bare_model = parse_model_spec(model)
    explicit = normalize_provider(provider)
    if explicit is not None:
        if spec_provider is not None and spec_provider != explicit:
            raise ConfigError(
                f"model spec provider {spec_provider!r} conflicts with provider "
                f"{explicit!r}"
            )
        return explicit

    if spec_provider is not None:
        return spec_provider

    inferred = infer_provider_from_model(bare_model)
    if inferred is not None:
        return inferred

    if bare_model is not None and bare_model.strip():
        raise ConfigError(
            f"could not infer provider from model {bare_model!r}; pass provider='anthropic' "
            "or provider='openai'"
        )
    raise ConfigError(
        "provider is required when model is omitted; pass provider='anthropic', "
        "provider='openai', or a provider-identifying model name"
    )


def normalize_model_for_provider(provider: Provider, model: str | None) -> str | None:
    """Return the bare model name, rejecting conflicting provider-prefixed specs."""

    spec_provider, bare_model = parse_model_spec(model)
    if spec_provider is not None and spec_provider != provider:
        raise ConfigError(
            f"model spec provider {spec_provider!r} conflicts with provider {provider!r}"
        )
    return bare_model


def normalize_builtin_tools(value: BuiltinToolsInput) -> BuiltinTools | None:
    """Normalize built-in tool controls.

    ``None`` keeps the provider default. ``"none"`` or ``[]`` request no
    built-in tools where the active provider can enforce that hard guarantee.
    A non-empty list requests a provider-native built-in allowlist.
    """

    if value is None:
        return None
    if isinstance(value, str):
        if value.strip().lower() == "none":
            return "none"
        raise ConfigError("builtin_tools string value must be 'none'")
    if not isinstance(value, Sequence):
        raise ConfigError("builtin_tools must be 'none' or a sequence of strings")
    tools = list(value)
    if not all(isinstance(tool, str) for tool in tools):
        raise ConfigError("builtin_tools must be 'none' or a sequence of strings")
    return tools or "none"


def normalize_effort_for_provider(provider: Provider, value: str | None) -> Effort | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError("effort must be a string or None")
    effort = value.strip().lower()
    if not effort:
        return None
    allowed = _ANTHROPIC_EFFORTS if provider == "anthropic" else _OPENAI_EFFORTS
    if effort not in allowed:
        raise ConfigError(
            f"effort {value!r} is not supported by provider {provider!r}; "
            f"expected one of: {', '.join(sorted(allowed))}"
        )
    if provider == "openai" and effort == "max":
        return "xhigh"
    return cast(Effort, effort)


@dataclass
class SubagentDef:
    """A subagent the parent can delegate a task to and get a result back.

    anthropic: mapped to a Claude Agent SDK ``AgentDefinition`` (invoked via the
    built-in ``Agent``/``Task`` tool).
    openai: mapped to Codex multi-agent config. ``tools`` and subagent
    ``max_turns`` have no direct Codex equivalent. For Codex, ``tools=None`` and
    ``tools=[]`` are both accepted and mean provider defaults; non-empty
    ``tools`` lists and any ``max_turns`` value raise ``ConfigError``.
    ``model=None`` inherits the parent model; ``model=INHERIT_MODEL`` is also
    accepted for YAML/config files that need an explicit inherit sentinel.
    """

    description: str
    prompt: str
    model: str | None = None
    # Built-in tool names the subagent may use (anthropic only, e.g. ["Read"]).
    tools: list[str] | None = None
    max_turns: int | None = None


def normalize_subagents_for_provider(
    provider: Provider, subagents: dict[str, SubagentDef]
) -> dict[str, SubagentDef]:
    """Normalize subagent model specs for the already-selected parent provider."""

    normalized: dict[str, SubagentDef] = {}
    for name, subagent in subagents.items():
        model = subagent.model
        if isinstance(model, str) and model.strip().lower() == INHERIT_MODEL:
            normalized[name] = dataclasses.replace(subagent, model=None)
            continue
        normalized[name] = dataclasses.replace(
            subagent,
            model=normalize_model_for_provider(provider, model),
        )
    return normalized


@dataclass
class RunRequest:
    """A fully-resolved request. Built by :class:`agent_sdk_wrapper.Agent` per call."""

    provider: Provider
    prompt: str
    model: str | None = None
    system_prompt: str | None = None

    # Custom Python-function tools. Plain callables; schema is derived from
    # type hints + docstring. See agent_sdk_wrapper.tools.
    tools: list[Callable[..., Any]] = field(default_factory=list)

    # Subagents keyed by name.
    subagents: dict[str, SubagentDef] = field(default_factory=dict)

    # External MCP servers exposed to the provider runtime.
    mcp_servers: list[McpServer] = field(default_factory=list)

    # Structured output: a Pydantic model, dataclass, or TypedDict.
    output_schema: type | None = None

    max_turns: int | None = None
    effort: Effort | None = None
    cwd: str | Path | None = None
    env: dict[str, str] = field(default_factory=dict)

    # Run-level controls.
    timeout: float | None = None  # seconds, wall-clock for the whole run
    max_retries: int = 2  # whole-run retries on transient errors (backoff)
    include_raw: bool = False
    include_events_in_result: bool = True
    artifacts_dir: str | Path | None = None
    on_provider_event: ProviderEventCallback | None = None
    # Provider built-in tools. ``None`` keeps provider defaults; ``"none"`` is a
    # hard no-builtins request and providers that cannot enforce it must reject.
    builtin_tools: BuiltinTools | None = None
    # Whether the model is offered web tools. ``None`` keeps provider defaults.
    # ``False`` removes WebSearch/WebFetch (Anthropic) and sets
    # ``tools.web_search=false`` (Codex). ``True`` explicitly enables them.
    # Controls tool affordances only; does not enforce network egress (a model
    # with shell access can still fetch URLs).
    web_tools: bool | None = None

    # ── Provider-specific passthrough ──────────────────────────────────
    # anthropic: built-in tools to allow (e.g. ["Read", "Bash"]). The
    # subagent ("Agent") tool is added automatically when subagents exist.
    allowed_tools: list[str] = field(default_factory=list)
    # anthropic: built-in/MCP tools to deny.
    # openai: applied to wrapper-managed callable/external MCP servers.
    disallowed_tools: list[str] = field(default_factory=list)
    # Provider session/thread to resume. ``Agent(continue_session=True)`` stores
    # the latest emitted session id and reuses it on later calls.
    session_id: str | None = None
    continue_session: bool = False
    # anthropic: 'default' | 'acceptEdits' | 'plan' | 'bypassPermissions' | 'dontAsk'
    permission_mode: str | None = None
    # anthropic: which on-disk settings to load. Default [] = none (isolated).
    setting_sources: list[str] | None = None
    # Escape hatch merged into the backend's native options object.
    extra_options: dict[str, Any] = field(default_factory=dict)
