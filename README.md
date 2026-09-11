# agent-sdk-wrapper

Python and TypeScript interfaces to the **Claude Agent SDK** and **OpenAI Codex SDK**.
The native SDKs run the agent loops, tools and sessions. This thin shim handles
provider selection, validation, normalized events/results and retries.
Each language calls its native SDK directly and runs independently.

| Package | Guide |
|---|---|
| `packages/python/` | [Python API](packages/python/README.md) |
| `packages/typescript/` | [TypeScript API](packages/typescript/README.md) |

Both expose `Agent.run()` and `Agent.stream()`. Providers are `anthropic` and
`openai`; `codex` aliases `openai`. Unsupported settings raise `ConfigError`.
See [capabilities and differences](packages/typescript/PARITY.md) and
[SDK versions](docs/sdk-versions.md).

## Develop

```sh
uv --directory packages/python sync --extra dev
uv --directory packages/python run pytest
npm ci
npm run verify
```

Run both offline suites in Ubuntu containers:

```sh
docker compose up --build --abort-on-container-failure python-verify typescript-verify
```

[Validation commands](packages/typescript/VALIDATION.md) include packaging and opt-in live tests.
