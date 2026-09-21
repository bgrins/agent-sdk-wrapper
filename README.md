# agent-sdk-wrapper

Native Python and TypeScript interfaces for the **Claude Agent SDK** and **Codex SDK**.
The SDKs manage tools, sessions and agent loops. The wrapper adds provider selection,
validation and normalized events/results. Install either language independently.

| Package | Guide |
|---|---|
| `packages/python/` | [Python API](packages/python/README.md) |
| `packages/typescript/` | [TypeScript API](packages/typescript/README.md) |

Both expose `Agent.run()` and `Agent.stream()`. Use provider `anthropic` or `codex`
(an alias for `openai`). Unsupported settings raise `ConfigError`. Runs use API or
cloud-provider credentials and never a runtime's stored login unless Codex
`cli_login="require"` asks for it.
See [API differences](packages/typescript/PARITY.md) and [SDK versions](docs/sdk-versions.md).

## Develop

```sh
uv --directory packages/python sync --extra dev
uv --directory packages/python run pytest
npm ci
npm run verify
```

Or use Docker:

```sh
docker compose up --build --abort-on-container-failure python-verify typescript-verify
```

[Validation commands](packages/typescript/VALIDATION.md) cover packaging and live tests.

## View traces

Run `npm run trace-viewer -- results` and open the printed URL. It lists the 500 most
recently updated runs from at most 5000 directories, read newest first, and `?index=`
must be same-origin. Except on macOS and on Linux with `/proc`, it serves only files
directly in the results directory.
For the [gVisor example](examples/gvisor/README.md), use
`npm run trace-viewer -- results/gvisor-output --depth 1`.
