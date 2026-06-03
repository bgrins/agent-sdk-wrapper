# agent-sdk-wrapper

`agent-sdk-wrapper` is a Python wrapper with one `Agent` API over:

- `anthropic`: Claude Agent SDK
- `openai`: OpenAI Codex SDK

It builds provider-specific requests, converts SDK streams into normalized
events, and aggregates `run()` calls into `RunResult` objects.

## Install

Requires Python `>=3.12`.

```bash
uv sync
uv sync --extra dev
```

The package exposes the `agent-sdk-wrapper` console script.

## Quick Start

```python
import asyncio
from agent_sdk_wrapper import Agent

async def main() -> None:
    agent = Agent(model="claude-haiku-4-5")
    result = await agent.run("Say hello.")
    print(result.final_text)

    async for env in agent.stream("Count to three."):
        print(env.to_json())

asyncio.run(main())
```

Provider can be explicit or inferred from common model names:

```python
Agent(provider="anthropic", model="claude-haiku-4-5")
Agent(provider="openai", model="gpt-5")
Agent(provider="codex")          # alias for provider="openai"
Agent(model="codex:gpt-5")       # provider:model syntax
```

## Core Features

| Feature | Anthropic | OpenAI / Codex |
|---|---|---|
| `run()` / `stream()` | Yes | Yes |
| Text, thinking, tools, usage | Normalized events | Normalized events |
| Python callable tools | In-process MCP server | Temporary stdio MCP server |
| External MCP servers | stdio/http | stdio/http via Codex config |
| Subagents | Claude `AgentDefinition` | Codex multi-agent config |
| Structured output | Native structured output + validation | Codex `outputSchema` + validation |
| `max_turns` | Native option | Wrapper-enforced over completed action items |
| Session resume | Claude resume | Codex thread resume |
| Provider-native event sidecar | Yes | Yes |

Provider-specific controls stay provider-specific when there is no reliable
cross-provider meaning. Unsupported combinations raise `ConfigError` rather
than being silently ignored.

Important configuration notes:

- `provider="codex"` is an alias for the `openai` adapter.
- `web_tools=False` disables Claude `WebSearch`/`WebFetch` and Codex
  `tools.web_search`.
- Codex rejects `builtin_tools`, provider-native tool filters, non-empty
  `SubagentDef.tools`, and `SubagentDef.max_turns`.
- Codex callable tools must be importable or simple enough for
  `inspect.getsource()`.
- `Agent.check_runtime()` validates the request before checking provider
  availability.

## MCP And Subagents

```python
from agent_sdk_wrapper import Agent, McpStdioServer, SubagentDef

agent = Agent(
    model="gpt-5",
    mcp_servers=[
        McpStdioServer(
            name="repo",
            command="uv",
            args=["run", "python", "-m", "repo_tools.server"],
            enabled_tools=["search", "summarize"],
        )
    ],
    subagents={
        "reviewer": SubagentDef(
            description="Reviews short code snippets.",
            prompt="Return a concise review verdict.",
        )
    },
)
```

## Traces And Artifacts

- `trace_file=...` writes normalized `EventEnvelope` JSONL.
- `artifacts_dir=...` writes `trace.jsonl`, `manifest.json`, `result.json`,
  and `provider-events.jsonl`.
- `provider-events.jsonl` records serialized provider-native SDK messages before
  wrapper normalization.
- `on_provider_event=...` receives those provider events live. Use
  `envelope.message` for the serialized payload or `envelope.raw` for the SDK
  object.
- Versioned JSON Schemas live under `docs/schemas/`.
- `docs/trace-viewer.html` is a dependency-free local viewer. Served from the
  repo root, it auto-discovers runs under `/results/`.

Examples write timestamped artifacts under the gitignored
`results/<provider>/<example>/<timestamp>/`.

## CLI

```bash
uv run agent-sdk-wrapper run --provider anthropic --prompt "Say hello" --output text
uv run agent-sdk-wrapper run --model gpt-5 --prompt "Say hello" --output jsonl
uv run agent-sdk-wrapper run --config agent-sdk-wrapper.toml --prompt "Audit this change"
```

Output modes:

| Mode | Output |
|---|---|
| `jsonl` | one serialized `EventEnvelope` per line |
| `text` | final assistant text |
| `json` | serialized `RunResult` |
| `--stream` | streamed text output |

Longer run definitions can live in TOML or JSON. CLI env/options override
config values, and repeatable tool filters append to config lists.

## Docker

The development image is Ubuntu 24.04.

```bash
docker compose build
docker compose run --rm verify
docker compose run --rm examples
docker compose run --rm integration    # live; requires credentials
docker compose run --rm fixtures
```

Run individual examples with services such as `example-run-basic`,
`example-subagent`, and `example-auditor-style`. Set `PROVIDER=codex` or
`MODEL=...` to switch providers.

## Tests

```bash
uv run pytest
docker compose run --rm verify
docker compose run --rm integration
```

Default tests are offline. Live integration tests require
`AGENT_SDK_WRAPPER_RUN_INTEGRATION=1` and provider credentials; Docker Compose
sets the integration flag for the `integration` service.

Downstream projects can use fake providers for offline tests:

```python
from agent_sdk_wrapper import Agent, Text, install_fake_providers

def test_agent_flow(monkeypatch):
    seen = []
    install_fake_providers(
        monkeypatch,
        events=lambda req: [Text(text=f"prompt={req.prompt}")],
        seen_requests=seen,
    )

    result = Agent(provider="codex").run_sync("hello")

    assert result.final_text == "prompt=hello"
    assert seen[0].provider == "openai"
```
