# agent-sdk-wrapper

Thin Python and TypeScript shims over the **Claude Agent SDK** and **OpenAI Codex SDK**.

The native SDKs own the agent loops, tool execution and sessions. This package
handles provider/model selection, strict configuration checks, normalized events
and results, and wrapper-level retries. Each language calls its native SDK
directly; neither implementation depends on the other.

The shared idea is small: `Agent.run()`, `Agent.stream()`, one request shape,
and predictable results. Provider-specific options stay explicit. Unsupported
combinations fail instead of silently approximating another provider's behavior.

| Package | Status | Guide |
|---|---|---|
| Python · `packages/python/` | Tools, structured output, sessions, traces | [Python API](packages/python/README.md) |
| TypeScript · `packages/typescript/` | Initial run/stream/resume slice; tools and structured output pending | [TypeScript API](packages/typescript/README.md) |

Both packages use the name `agent-sdk-wrapper` in their respective registries.
Provider names are `anthropic` and `openai`; `codex` aliases `openai`.
See [API parity](packages/typescript/PARITY.md) for the current differences.

## Use

```python
import asyncio
from agent_sdk_wrapper import Agent

result = asyncio.run(Agent(provider="codex").run("Summarize this repository."))
print(result.final_text)
```

```ts
import { Agent } from "agent-sdk-wrapper";

const result = await new Agent({ provider: "codex" }).run("Summarize this repository.");
console.log(result.final_text);
```

These calls use the provider's credentials and native runtime. Check
`result.status` before treating output as a successful answer.

## Develop

```sh
uv --directory packages/python sync --extra dev
uv --directory packages/python run pytest

npm ci
npm run verify
```

Or validate both in separate Ubuntu containers:

```sh
docker compose up --build --abort-on-container-failure python-verify typescript-verify
```

Default tests are offline. See [validation](packages/typescript/VALIDATION.md)
for container services, packaging checks and explicitly gated live tests.
