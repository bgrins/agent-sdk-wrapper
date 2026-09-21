# Validation

Run from the repository root. Default tests are offline.

## Local checks

```sh
uv --directory packages/python sync --extra dev
uv --directory packages/python run pytest
uv --directory packages/python run ruff check
npm ci
npm run verify
```

`npm run verify` runs unit/viewer tests, typecheck, lint, format and package checks.
Package checks require `tar` and symlinks on macOS/Linux.

## Containers

```sh
docker compose up --build --abort-on-container-failure python-verify typescript-verify
```

Builds need network access; verify services have none. Rebuild after TypeScript
edits. Python source uses a bind mount.

## Live tests

```sh
docker compose run --rm --build python-integration
AGENT_SDK_WRAPPER_TS_RUN_INTEGRATION=1 docker compose run --rm --build typescript-integration
# TypeScript on the host:
AGENT_SDK_WRAPPER_TS_RUN_INTEGRATION=1 npm run test:integration
```

Set `ANTHROPIC_API_KEY` and/or `OPENAI_API_KEY`; live tests make billed calls.
Missing keys/flags skip tests. Python Compose sets its flag; TypeScript requires
the flag above. Override models with `AGENT_SDK_WRAPPER_ANTHROPIC_MODEL` or
`AGENT_SDK_WRAPPER_OPENAI_MODEL` (Python, which defaults to `claude-haiku-4-5` and
`gpt-5.6-luna`) and `AGENT_SDK_WRAPPER_TS_ANTHROPIC_MODEL` or
`AGENT_SDK_WRAPPER_TS_OPENAI_MODEL` (TypeScript). Compose loads `.env`; host commands do not.
On the host, an unset model follows inherited `ANTHROPIC_MODEL` and
`ANTHROPIC_DEFAULT_*_MODEL`; set the overrides above when those are exported.

## Build packages

```sh
uv --directory packages/python build
npm run build
npm pack --workspace agent-sdk-wrapper --ignore-scripts --pack-destination /tmp
```

Build before packing: `.npmrc` disables lifecycle scripts, including `prepack`.
After dependency updates, run `npm audit` and `npm audit signatures`.
