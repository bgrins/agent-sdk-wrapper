# Python API

Requires Python 3.12+. Run `uv sync --extra dev` from this directory.

## Run and stream

```python
import asyncio
from agent_sdk_wrapper import Agent

async def main():
    agent = Agent(provider="codex", continue_session=True)  # or "anthropic"
    result = await agent.run("Remember ALPHA42. Reply READY.")
    if result.status != "success":
        raise RuntimeError(result.error)
    print(result.final_text)

    async for event in agent.stream("Repeat the token."):
        print(event.to_json())

asyncio.run(main())
```

Claude needs `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN` or a cloud-provider flag;
Codex needs `OPENAI_API_KEY` or `provider_options={"api_key": ...}`. With the default
`cli_login="deny"`, a run never uses a runtime's stored login and fails before
starting without credentials. Codex accepts `cli_login="require"` to use its stored
ChatGPT login instead; Claude rejects it. Codex keeps API keys in memory, not `auth.json`.
Claude loads no on-disk settings or `CLAUDE.md` unless `setting_sources` names them.

To resume in another process, save `result.session_id` and pass it as
`Agent(provider=..., session_id=...)` with the same working directory.

## Options

Constructor keywords are defaults; `run()` and `stream()` accept per-call overrides.
`RunRequest` is the resolved request passed to adapters.

| Options | Purpose |
|---|---|
| `provider`, `model`, `effort`, `cwd` | Provider/model selection and execution settings |
| `tools`, `mcp_servers`, `subagents` | Callable tools, external MCP and subagents |
| `output_schema` | Validated structured output |
| `session_id`, `continue_session` | Explicit resume or automatic continuation |
| `max_retries`, `timeout`, `max_turns` | Retries (default 2), provider-wait deadline and action limit |
| `cli_login`, `setting_sources` | Stored-login policy; Claude on-disk settings (default none) |
| `provider_options`, `extra_options` | Provider-specific settings; unsupported combinations fail |

`run()` and `stream()` raise `ConfigError` for invalid settings before any event.
Other failures produce failed results; check `result.status`, or use
`raise_on_error=True` for `RunFailedError`. Signal-killed runtimes are recorded,
then raise `ProcessTerminatedError`. A run retries only before its first progress
event (text, thinking, tools, subagents, structured output), for transient errors
and retryable error events. `timeout` bounds waits on the provider, never consumer
code. One `continue_session` run may be active per Agent. `check_runtime()`
validates settings, runtime and credentials. `run_sync()` works outside an event loop.

`result.final_text` is the last assistant message. `Error.error_type` uses the
[shared vocabulary](../typescript/PARITY.md#error-types); only
`transient_api_error` is retryable.

| Capability | Claude | Codex |
|---|---|---|
| Callable tools | In-process MCP; pydantic-validated arguments | Stdio MCP found through Codex tool search; functions must be importable or self-contained |
| Structured output | Supported | Strict JSON schema: object roots, no free-form dicts |
| External MCP, resume | Supported | Supported |
| Subagents | Native definitions; background tasks disabled | Native multi-agent config; per-subagent tools/turn limits rejected |
| `max_turns` | Native turn limit | Wrapper limit on completed action items |
| Built-in tool filtering | Native controls | Unsupported; `web_tools` sets the `web_search` mode |

Claude `effort` also sets `CLAUDE_CODE_EFFORT_LEVEL`, which the CLI ranks above
`--effort`. `extra_options` cannot replace keys that first-class options set.
Codex `sandbox` applies per thread; caller `config_overrides` take precedence.

See [examples](examples/) and [API differences](../typescript/PARITY.md).

## Events and traces

`EventEnvelope` contains `run_id`, `sequence`, `timestamp` and `event`.
`RunResult` contains status, text, usage/cost, session ID and retained events.

- `on_event`: normalized envelopes.
- `on_provider_event`: native envelopes; `.raw` is the SDK object, `.message` is serialized.
- `trace_file`: normalized JSONL.
- `artifacts_dir`: trace, result, manifest and native-event files.

From the repository root, run `npm run trace-viewer -- packages/python/results`.
Open the printed URL and select a trace.
Claude token totals include subagents via `model_usage`; `requests` remains a
main-loop turn-count proxy. Codex usage is per turn; its cost is unavailable.
`SessionInfo.model` reports the model the runtime actually used.
[Accounting limits](../typescript/PARITY.md).

CLI: `uv run agent-sdk-wrapper run --provider codex --prompt "Say hello" --output jsonl`.
`--stream` requires `--output text`.
Run tests with `uv run pytest`; see [validation](../typescript/VALIDATION.md) for Compose and live tests.
