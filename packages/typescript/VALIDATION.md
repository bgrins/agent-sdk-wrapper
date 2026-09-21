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
the flag above. Compose loads `.env`; host commands do not.

Python's live tests are the [conformance cases](../../docs/fixtures/CONFORMANCE.md) with a
`live` section; offline, `pytest` runs every case against local mock APIs. On the host:

```sh
AGENT_SDK_WRAPPER_RUN_INTEGRATION=1 uv --directory packages/python run pytest -m integration tests/conformance
```

Each case runs with scratch `HOME`, `CODEX_HOME` and Claude config directories, and without
inherited `ANTHROPIC_*`, `OPENAI_*`, `CODEX_*` and `CLAUDE_CODE_*` variables other than the two
API keys. Python models default to `claude-haiku-4-5` and `gpt-5.6-luna`; override them with
`AGENT_SDK_WRAPPER_ANTHROPIC_MODEL` or `AGENT_SDK_WRAPPER_OPENAI_MODEL`. Python writes each live
run's artifacts to `packages/python/results/integration-runs/<timestamp>/<case>/run-<n>`, or under
`AGENT_SDK_WRAPPER_TEST_ARTIFACTS_DIR`. TypeScript models come from
`AGENT_SDK_WRAPPER_TS_ANTHROPIC_MODEL` or `AGENT_SDK_WRAPPER_TS_OPENAI_MODEL`; on the host, an
unset TypeScript model follows inherited `ANTHROPIC_MODEL` and `ANTHROPIC_DEFAULT_*_MODEL`.

## Build packages

```sh
uv --directory packages/python build
npm run build
npm pack --workspace agent-sdk-wrapper --ignore-scripts --pack-destination /tmp
```

Build before packing: `.npmrc` disables lifecycle scripts, including `prepack`.
After dependency updates, run `npm audit` and `npm audit signatures`.
