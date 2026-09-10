# Agent Notes

Repo-specific invariants for editing `agent-sdk-wrapper`.

- Keep provider SDK imports inside `packages/python/src/agent_sdk_wrapper/providers/`.
  TypeScript production SDK imports belong in `packages/typescript/src/providers/`;
  adapter tests may import SDK types to check fixtures against pinned declarations.
- Keep Docker Ubuntu-based. Do not switch to Alpine/musl.
- Keep the Python Dockerfile free of Node/npm, Claude Code, and standalone Codex
  CLI installs; its runtimes come from Python SDK packages. The separate
  `packages/typescript/Dockerfile` supplies Node/npm and obtains runtimes from npm SDKs.
  Both final images stay Ubuntu-based. TypeScript has its own build-context
  ignore file so host node_modules and generated artifacts cannot enter it.
- Keep `.env`, `.claude/`, `results/`, caches, and generated artifacts out of
  git.
- Use Bash or Node for shared repository automation. Keep Python-specific
  tooling under `packages/python/`.
- Prefix Compose services with `python-` or `typescript-`; neither language
  gets unprefixed services or a default profile.
- Provider/model resolution belongs in the wrapper. `codex` is an alias for the
  `openai` adapter.
- Unsupported provider combinations should raise `ConfigError`, not silently
  degrade.
- `Agent.check_runtime()` validates the request before checking provider
  availability.
- Default tests must stay offline. Python live tests run through Docker Compose
  with provider credentials. The native TypeScript smoke tests run on a Node host
  or the explicit `typescript-integration` Compose service
  only with `AGENT_SDK_WRAPPER_TS_RUN_INTEGRATION=1` and provider credentials;
  do not add Node to the Python Docker image.
- Keep TypeScript dependencies minimal. Use Node's test runner. Select stable
  releases from registry metadata after a seven-day cooldown, check advisories,
  and commit exact SDK pins and the npm lockfile. Keep install scripts disabled.
- `TokenUsage` fields mean the same thing across providers: `input_tokens`
  includes cache, `output_tokens` includes reasoning. Adapters normalize; they
  do not pass provider counters through raw.
- Every member of the `AgentEvent` union needs a branch in
  `docs/schemas/agent-sdk-wrapper.event-envelope-jsonl.v1.schema.json`, a case
  in `testing.event_from_dict`, and rendering in `docs/trace-viewer.html`.
  `test_trace_schema_covers_every_event_type` enforces the first.
  The TypeScript event vocabulary is a subset of this same schema. Shared
  fixtures in `docs/fixtures/native-twin-v1.json` are replayed by both suites.
  TypeScript's `test/schema.test.ts` checks exhaustive event/field coverage;
  `npm run test:package` verifies the packed public API with offline adapters.
- A provider error that never raises still needs classifying. Map the terminal
  state onto a specific `error_type` (`max_turns`, `refused`,
  `transient_api_error`) and set `retryable`; do not emit a generic error.
- A signal-killed runtime raises `ProcessTerminatedError`, never
  `TransientError`. Retrying in-place only burns the shutdown window.
- The Codex callable-tool server is generated source run in a subprocess, so
  offline unit tests do not import it. Changing `_tool_server_script()` or
  bumping `mcp` needs `test_codex_tool_server_script_completes_an_mcp_handshake`
  to pass; it drives the real stdio protocol. `mcp` 2.x renamed `FastMCP` to
  `MCPServer`, and the script supports both.
- `RunStarted` carries the prompt and system prompt. A trace that records every
  answer and none of the questions cannot be read on its own.
- Neither SDK ships fault injection. To exercise error paths, point the runtime
  at a local mock: Anthropic via `env={"ANTHROPIC_BASE_URL": ...}` (forwarded to
  the spawned CLI), Codex via a `model_providers.<id>.base_url` config override.
  The CLI retries 429/5xx internally, so a retryable status stalls rather than
  surfacing; use a 4xx to reach the terminal-error path.
- Reasoning is on by default for both providers (Anthropic `thinking`
  adaptive/summarized, Codex `summary="auto"`). A reasoning item that exposes no
  text still emits a `Thinking` event: the tokens were billed, so a dropped item
  would make a run that reasoned look like one that did not.
