"""agent-sdk-wrapper — a unified wrapper over the Claude Agent SDK and the OpenAI
Codex SDK.

Quick start::

    from agent_sdk_wrapper import Agent

    agent = Agent(model="claude-sonnet-4-6")
    result = agent.run_sync("Say hello")
    print(result.final_text)
"""

from __future__ import annotations

from .agent import Agent
from .artifacts import ProviderEventCallback, ProviderEventEnvelope
from .errors import (
    AgentSdkWrapperError,
    ConfigError,
    ProviderNotAvailableError,
    RunFailedError,
    TransientError,
)
from .events import (
    AgentEvent,
    AgentUpdated,
    Error,
    EventEnvelope,
    RunEndedReason,
    RunFinished,
    RunResult,
    RunStarted,
    RunStatus,
    SessionInfo,
    StructuredOutput,
    Text,
    Thinking,
    TokenUsage,
    ToolCall,
    ToolResult,
    Usage,
    WarningEvent,
)
from .logging import LOGGER_NAME, TraceWriter, get_logger
from .mcp import McpHttpServer, McpServer, McpStdioServer, McpToolApprovalMode
from .request import (
    INHERIT_MODEL,
    PROVIDERS,
    BuiltinTools,
    BuiltinToolsInput,
    Effort,
    Provider,
    ProviderInput,
    RunRequest,
    SubagentDef,
    infer_provider_from_model,
    normalize_builtin_tools,
    normalize_effort_for_provider,
    normalize_model_for_provider,
    normalize_provider,
    parse_model_spec,
    resolve_provider,
)
from .testing import (
    EventFactory,
    EventSource,
    FakeProvider,
    TraceReplay,
    envelope_from_dict,
    event_from_dict,
    install_fake_providers,
    load_trace,
    load_trace_replay,
    run_result_summary,
    trace_summary,
)

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentEvent",
    "AgentUpdated",
    "BuiltinTools",
    "BuiltinToolsInput",
    "ConfigError",
    "Error",
    "EventEnvelope",
    "EventFactory",
    "EventSource",
    "Effort",
    "FakeProvider",
    "INHERIT_MODEL",
    "LOGGER_NAME",
    "Text",
    "McpHttpServer",
    "McpServer",
    "McpStdioServer",
    "McpToolApprovalMode",
    "PROVIDERS",
    "Provider",
    "ProviderEventCallback",
    "ProviderEventEnvelope",
    "ProviderInput",
    "ProviderNotAvailableError",
    "RunFailedError",
    "RunEndedReason",
    "RunFinished",
    "RunRequest",
    "RunResult",
    "RunStarted",
    "RunStatus",
    "SessionInfo",
    "StructuredOutput",
    "SubagentDef",
    "AgentSdkWrapperError",
    "Thinking",
    "TokenUsage",
    "ToolCall",
    "ToolResult",
    "TraceWriter",
    "TraceReplay",
    "TransientError",
    "Usage",
    "WarningEvent",
    "__version__",
    "envelope_from_dict",
    "event_from_dict",
    "get_logger",
    "infer_provider_from_model",
    "install_fake_providers",
    "load_trace",
    "load_trace_replay",
    "normalize_builtin_tools",
    "normalize_effort_for_provider",
    "normalize_model_for_provider",
    "normalize_provider",
    "parse_model_spec",
    "run_result_summary",
    "resolve_provider",
    "trace_summary",
]
