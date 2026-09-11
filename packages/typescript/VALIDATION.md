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

`npm run verify` runs unit tests, typecheck, lint, format and a packed-consumer check.
Individual commands: `npm test`, `npm run typecheck`, `npm run lint`,
`npm run format:check`, `npm run test:package`.
Shared JSON fixtures test normalized aggregation; native streams are mocked.
The package check uses existing dependencies, `tar` and symlinks on macOS/Linux.

## Containers

```sh
docker compose up --build --abort-on-container-failure python-verify typescript-verify
```

Both images use Ubuntu; verify services have no network. Builds need network access.
Services use `python-` or `typescript-` prefixes. TypeScript source is copied into
its image, so rebuild after edits; Python uses a repository bind mount.

## Live tests

```sh
docker compose run --rm --build python-integration
AGENT_SDK_WRAPPER_TS_RUN_INTEGRATION=1 docker compose run --rm --build typescript-integration
# TypeScript on the host:
AGENT_SDK_WRAPPER_TS_RUN_INTEGRATION=1 npm run test:integration
```

These make billed calls and require `ANTHROPIC_API_KEY` and/or `OPENAI_API_KEY`.
Python Compose sets its integration flag; TypeScript requires the explicit flag above.
Missing flags/keys skip tests. TypeScript tests cover stream, continue and resume.
Optional model overrides: `AGENT_SDK_WRAPPER_TS_ANTHROPIC_MODEL` and
`AGENT_SDK_WRAPPER_TS_OPENAI_MODEL`. Host commands do not load `.env`; Compose does.

## Build packages

```sh
uv --directory packages/python build
npm run build
npm pack --workspace agent-sdk-wrapper --ignore-scripts --pack-destination /tmp
```

Build before packing: `.npmrc` disables lifecycle scripts, including `prepack`.
After dependency updates, run `npm audit` and `npm audit signatures`.
