# Agent notes

- Keep docs short: API usage, limits and commands. Never mention downstream
  projects or narrate development history.
- Python SDK imports belong in `packages/python/src/agent_sdk_wrapper/providers/`;
  TypeScript SDK imports belong in `packages/typescript/src/providers/`.
  Adapter tests may import SDK types.
- Keep final Docker images Ubuntu-based. Python gets runtimes from Python SDKs;
  do not install Node/npm or standalone CLIs there. TypeScript has its own image
  and build-context ignore file; exclude host node_modules and generated files.
  The gVisor Python example also includes Node/git to run its target Node project;
  its provider runtimes still come from Python SDKs. Its gateway image has
  neither, and runs as non-root.
- Keep `.env`, `.claude/`, `results/`, caches and generated artifacts out of git.
- Shared automation uses Bash or Node. Python tooling stays in `packages/python/`.
- Prefix language-specific Compose services with `python-` or `typescript-`;
  neither language is the default. Shared example services use neutral names.
- Keep dependencies minimal: Node's test runner, exact SDK pins and lockfiles,
  npm install scripts disabled. Verify registry metadata/advisories. Use a
  seven-day cooldown unless a newer release is explicitly requested.
- Provider/model resolution belongs in the wrapper; `codex` aliases `openai`.
  Reject unsupported combinations with `ConfigError`.
- `check_runtime()` / `checkRuntime()` validate requests before availability.
- Default tests stay offline. Python live tests use Compose. TypeScript live
  tests require `AGENT_SDK_WRAPPER_TS_RUN_INTEGRATION=1` and provider keys.
- Input token totals include cache; output totals include reasoning.
  Claude `usage` excludes subagents: prefer `model_usage` / `modelUsage`, using
  `usage` only when that map is absent/empty. Never add both; cost includes subagents.
- Every Python event union member needs a shared schema branch,
  `testing.event_from_dict` case and trace-viewer rendering. TypeScript uses a
  subset; keep its exhaustive schema checks and shared replay fixtures aligned.
- `RunStarted` includes prompt and system prompt. Preserve empty/redacted
  `Thinking` events when reasoning occurred; both providers reason by default.
- Classify terminal provider failures even when the SDK never raises:
  a specific `error_type`, not a generic error. Use the PARITY.md
  vocabulary and prefer structured native signals over message text.
- Signal-killed runtimes record a `process_terminated` error, then raise
  `ProcessTerminatedError`; never retry them.
- Keep runs isolated from the host: Claude `setting_sources` defaults to `[]`,
  `effort` is pinned through the child env, `cli_login` defaults to `deny`, and
  credentials are never persisted.
- `final_text` is the last `Text`; adapters emit one `Text` per contiguous text run of
  an assistant message, even across frames.
- Changes to Python `_tool_server_script()` or `mcp` require the real offline
  MCP handshake test. The generated server uses mcp 2's low-level `Server` with
  `agent_sdk_wrapper.tools` schemas and calls.
- Fault tests use local mock endpoints: Claude `ANTHROPIC_BASE_URL` via `env`;
  Codex `model_providers.<id>.base_url`. Runtimes retry 429/5xx internally; set
  `CLAUDE_CODE_MAX_RETRIES=0` or Codex `request_max_retries`/`stream_max_retries=0`
  for fast failures. A failed Claude stream still gets one non-streaming request.
  Neither SDK provides fault injection.
