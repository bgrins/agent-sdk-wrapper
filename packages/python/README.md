# Python API

Python 3.12+. Calls the native Claude Agent SDK or Codex SDK directly.
Run `uv sync --extra dev` from this directory to install locally.

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

Supply provider credentials through the native SDK's environment or login.
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
| `max_retries`, `timeout`, `max_turns` | Retry budget, deadline and action limit |
| `provider_options`, `extra_options` | Provider-specific settings; unsupported combinations fail |

Check `result.status`; runtime errors can produce failed results without raising.
Use `raise_on_error=True` for `RunFailedError`. `check_runtime()` validates settings
and runtime availability. `run_sync()` is available outside an event loop.

| Capability | Claude | Codex |
|---|---|---|
| Callable tools | In-process MCP | Temporary stdio MCP; functions must be importable or source-extractable |
| Structured output, external MCP, resume | Supported | Supported |
| Subagents | Native definitions | Native multi-agent config; per-subagent tools/turn limits rejected |
| `max_turns` | Native turn limit | Wrapper limit on completed action items |
| Built-in tool filtering | Native controls | Unsupported; `web_tools` and MCP controls are separate |

See [examples](examples/) for tools and structured output, and
[API differences](../typescript/PARITY.md) for limits shared with TypeScript.

## Events and traces

`EventEnvelope` contains `run_id`, `sequence`, `timestamp` and `event`.
`RunResult` contains status, text, usage/cost, session ID and retained events.

- `on_event`: normalized envelopes.
- `on_provider_event`: native envelopes; `.raw` is the SDK object, `.message` is serialized.
- `trace_file`: normalized JSONL.
- `artifacts_dir`: trace, result, manifest and native-event files.

Open root `docs/trace-viewer.html` and select **Open Files** or **Open Artifact Directory**.
Claude token totals include subagents via `model_usage`; `requests` remains a
main-loop turn-count proxy. Codex cost is unavailable. [Accounting limits](../typescript/PARITY.md).

CLI: `uv run agent-sdk-wrapper run --provider codex --prompt "Say hello" --output jsonl`.
Run tests with `uv run pytest`; see [validation](../typescript/VALIDATION.md) for Compose and live tests.
