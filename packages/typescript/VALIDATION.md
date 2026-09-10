# Validation

Run these commands from a repository checkout, at its root. TypeScript requires
Node 22.14+ in the 22.x line (or Node 24+) and npm; Python uses the existing uv
environment. Default tests do not need credentials. No command below publishes
either package.

The packages are siblings under `packages/python/` and `packages/typescript/`.
Shared schemas, fixtures and the viewer remain under root `docs/`. Python's sdist
includes the shared contracts so its tests also work outside this monorepo.

## Offline checks

```sh
npm ci
npm test
npm run typecheck
npm run lint
npm run format:check
npm run test:package
uv --directory packages/python run pytest -q -m 'not integration'
```

`npm ci` installs the exact lockfile with lifecycle scripts disabled by `.npmrc`.
It needs registry access or a populated cache; the tests themselves are offline.
No separate JavaScript test framework or schema-validator dependency is used.

Verification on 2026-09-10 passed in both Ubuntu 24.04 containers: 52 TypeScript
tests on Node 22.23.2/npm 10.9.8 and 186 Python tests on Python 3.12.3, plus
typecheck, lint/format and the packed-consumer check. This covers the declared
Node 22 minimum. Host checks also passed on macOS with Node 25.6.1/npm 11.9.0
and Python 3.13.3 after the package move. Distribution builds passed; all 186
Python tests also passed against an extracted source distribution outside the
repository. The two TypeScript live cases (including the Compose service) and
Python live module skipped without live opt-in.
The Python MCP handshake test was corrected to await initialize/list responses
before closing stdin, eliminating a shutdown race observed during this pass.

The Claude availability review adds three offline regressions (55 TypeScript
tests total): an isolated SDK dependency tree with a pnpm-style symlink,
readable interpreter scripts without executable bits, and native binary
permission checks. Availability checks do not launch a query. Runtime lookup
uses Node's [findPackageJSON](https://nodejs.org/download/release/v22.14.0/docs/api/module.html#modulefindpackagejsonspecifier-base)
from the SDK's ESM entrypoint; that API sets the Node 22.14 minimum. No resolver
dependency or experimental Node flag is needed.

| Check | What it establishes |
|---|---|
| `npm test` | Fake-adapter contract tests and mocked native streams: provider selection, strict validation, envelope ordering, aggregation, retries, sessions, cancellation/cleanup, native error mapping, retraction rejection and usage normalization. It also compiles source, unit tests and the live example. |
| Shared JSON replay | Both languages aggregate the same normalized fixtures. Python validates them with the existing JSON Schema dependency. TypeScript's exhaustive event/field checks prevent its union from silently outgrowing fixture/schema coverage. This is not proof of native-provider equivalence. |
| `npm run typecheck` | Strict source/test/example typing, including negative assertions for unsupported native option shapes. |
| `npm run lint` / `format:check` | Biome checks source, tests, examples, package consumer fixture and validation script. |
| `npm run test:package` | Builds and packs the npm package with scripts disabled; checks its file list; extracts it into a temporary consumer project; compiles against public declarations and runs stream/collect/continue/resume for both provider selections using fake adapters. |
| Python pytest | Existing offline Python behavior and shared-schema tests continue to pass independently of Node. |

The package check reuses `node_modules` from the existing locked installation;
it makes no registry request and does not reinstall dependencies. It tests
packaged files/exports, not a clean-machine install or real provider invocation.
It uses `tar` and directory symlinks and is intended for macOS/Linux. Temporary
files are removed after either success or failure. npm still writes its normal
cache, so restricted environments may need permission for that cache directory.

## Build artifacts

```sh
npm run build
npm pack --workspace agent-sdk-wrapper --ignore-scripts --pack-destination /tmp
uv --directory packages/python build
```

Build before packing: repository `.npmrc` disables the `prepack` lifecycle hook.
The npm tarball contains `dist/`, package metadata, LICENSE, README, PARITY and
this guide. It excludes tests, source, examples, node_modules and repository
state. The Python sdist/wheel use their existing package layout; npm files and
TypeScript source are excluded. Do not stage generated distributions.

## Containers

Each package owns its Dockerfile and build-context ignore file. Both final images
use Ubuntu 24.04. Python retains its SDK-provided runtimes and no Node install;
TypeScript copies Node/npm from the official Node image and uses npm SDK runtimes.
Node 22.23.2 was selected from the [official release index](https://nodejs.org/dist/index.json)
on 2026-09-10; its 2026-07-28 release predates the seven-day cooldown.

```sh
# Build and run both offline suites; a failure stops the other service.
docker compose up --build --abort-on-container-failure python-verify typescript-verify

# Run one suite, or one TypeScript command.
docker compose run --rm --build python-verify
docker compose run --rm --build typescript-verify
docker compose run --rm typescript-verify npm test
```

Both verify services run with networking disabled. Image builds need registry
and apt access. The TypeScript service uses the source copied into its image;
rebuild after edits. It does not mount host node_modules. Python retains the
existing repository bind mount and `/opt/venv` environment. Services and images
use explicit `python-` or `typescript-` names. Both languages use Compose profiles;
there is no default language. Name the desired services in each command; doing
so enables them regardless of profile. Both live test services use the
`integration` profile.

## Gated live smoke tests

```sh
npm run test:integration
AGENT_SDK_WRAPPER_TS_RUN_INTEGRATION=1 npm run test:integration

# Equivalent Node container; still requires the flag and provider key(s).
AGENT_SDK_WRAPPER_TS_RUN_INTEGRATION=1 docker compose run --rm --build typescript-integration
```

The first command reports two skipped cases unless the flag is already enabled.
The second enables only cases with their corresponding `ANTHROPIC_API_KEY` or
`OPENAI_API_KEY` present in the environment. `.env` is not loaded. Login-only
authentication does not enable these tests; they deliberately require keys.
Optional `AGENT_SDK_WRAPPER_TS_ANTHROPIC_MODEL` and
`AGENT_SDK_WRAPPER_TS_OPENAI_MODEL` select the model; otherwise its native default
is used. Each enabled case makes three real, billed calls in a temporary working
directory: stream/collect, continue, and explicit resume through another Agent.
Compose reads its usual root `.env`/shell substitutions and forwards only the
declared keys, integration flag and optional model overrides. The TypeScript
integration service does not set the flag automatically and has network access.

Python live tests retain their existing workflow:

```sh
docker compose run --rm python-integration
```

Do not enable a live flag merely to eliminate skipped tests. A successful offline
run does not establish provider authentication, model access, runtime compatibility
or permission behavior. Record enabled provider/model/SDK versions when running
live tests; keep credentials and raw traces out of git.

## Remaining validation before Foofrix adoption

- Run the gated basic smoke tests with each provider's credentials.
- Add the verified Node 22/container checks to CI.
- Verify the Python Codex runtime's raw output/reasoning accounting before
  changing its normalization; see PARITY.md for the unresolved difference.
- Implement one actual host callback, then test MCP initialize/list/call,
  invalid input, callback failures and cleanup offline; prove the model invokes
  it through both live SDKs. Current smoke tests make no callable-tool claim.
- Add structured output as a separate tested feature, with explicit validation.

For dependency changes, use the README's registry-metadata/cooldown process,
then repeat offline checks plus `npm audit` and `npm audit signatures`. Do not
substitute remembered version numbers or treat a clean audit as proof that a
package cannot contain malicious code.
