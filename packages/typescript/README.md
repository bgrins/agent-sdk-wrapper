# TypeScript API

ESM; Node 22.14+ in the 22.x line, or Node 24+.
Run `npm ci && npm run build` from the repository root.

## Run, stream and resume

```ts
import { Agent, collectRun } from "agent-sdk-wrapper";

const defaults = {
  provider: "codex" as const, // or "anthropic"
  cwd: process.cwd(),
  continueSession: true,
};
const agent = new Agent(defaults);
const result = await collectRun(agent.stream("Remember ALPHA42. Reply READY."), (env) => {
  if (env.event.type === "text") process.stdout.write(env.event.text);
});
if (result.status !== "success") throw new Error(result.error ?? "Run failed");

// Save the provider and session ID to resume after restarting.
if (result.session_id) {
  const resumed = new Agent({ ...defaults, sessionId: result.session_id });
  const next = await resumed.run("Repeat the token.");
  if (next.status !== "success") throw new Error(next.error ?? "Run failed");
  console.log(next.final_text);
}
```

Claude needs `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN` or an enabled cloud-provider
flag such as `CLAUDE_CODE_USE_BEDROCK=1`; a keyless `ANTHROPIC_BASE_URL` gateway needs a
placeholder key. Codex needs `OPENAI_API_KEY` or `client.apiKey`, including with
`baseUrl`. With the default `cliLogin: "deny"`,
a run never uses a runtime's stored login and throws before starting without
credentials. Codex accepts `cliLogin: "require"` to use its stored ChatGPT login
(checked with `codex login status`); Claude rejects it. The wrapper does not load `.env`.

## Contract

`run()` and `stream()` accept a prompt or `RunRequest`. Constructor defaults are
overridden per field, without deep merging. One active run per Agent.

| Options | Meaning |
|---|---|
| `provider`, `model`, `effort`, `cwd` | Provider/model selection and execution settings; conflicts fail. A `provider:` prefix counts only for `anthropic`, `openai` or `codex`, so Bedrock IDs and ARNs stay whole; blank values mean omitted; `cwd` must be an existing directory |
| `sessionId`, `continueSession` | Explicit resume or automatic reuse of the latest ID per provider; a constructor `sessionId` gives way to the latest reported session |
| `signal` | Cancellation or `AbortSignal.timeout(ms)`; ignored after the terminal frame |
| `cliLogin` | `"deny"` (default) or Codex-only `"require"` for the runtime's stored login |
| `traceFile` | Write normalized JSONL during `run()` or `stream()` |
| `providerOptions` | Native options below; unknown keys fail |
| `onProviderEvent`, `includeRaw` | Original SDK-event callback, or raw data on mapped events |

`EventEnvelope` has `run_id`, zero-based `sequence`, `timestamp` and `event`.
Events cover run boundaries, sessions, completed text/thinking, tool activity,
usage, warnings and errors. `RunResult` contains status, text, usage/cost,
session ID, the first error and its `error_type`, and events. `final_text` is the last
assistant message. Runs are never retried; see [retrying](PARITY.md#retrying). For Claude,
`session_info.model` reports the model the runtime used, including a fallback model after
`model_fallback` or a `model_refusal_fallback` that retracts nothing (a warning precedes
it); Codex exec doesn't expose it. `error_type` uses the
[shared vocabulary](PARITY.md#error-types). `collectRun` rejects incomplete or
misordered streams.

`checkRuntime()` validates configuration, runtime and credentials without a model call.
Setup throws `ConfigError`, `RuntimeUnavailableError` or `ProviderError`; runtime
failures usually produce failed results. Signal-killed processes record the failure,
then throw `ProcessTerminatedError`; a kill after the caller's abort is cancelled.
Trace I/O or serialization failures throw `TraceWriteError`.
Implement `ProviderAdapter` for custom validation, runtime checks and streaming.

## Capabilities

| Capability | Claude | Codex |
|---|---|---|
| Run, stream, resume, effort, cwd | Supported | Supported |
| Trace files | Supported | Supported |
| Permissions and built-in tool controls | Provider-specific | Provider-specific |
| System prompt and max turns | Provider-specific | Not yet implemented |
| Host callbacks, MCP registration, structured output | Not yet implemented | Not yet implemented |
| Token deltas, subagent lifecycle, managed artifacts | Not yet implemented | Not yet implemented |

Native options use `providerOptions.provider: "anthropic"` or `"openai"`:

| Group | Accepted keys |
|---|---|
| Claude `options` | `permissionMode`, `allowDangerouslySkipPermissions`, `tools`, `allowedTools`, `disallowedTools`, `settingSources`, `env`, `systemPrompt`, `maxTurns`, `thinking`, `pathToClaudeCodeExecutable` |
| Codex `client` | `apiKey`, `baseUrl`, `env`, `codexPathOverride` |
| Codex `thread` | `sandboxMode`, `approvalPolicy`, `skipGitRepoCheck`, `networkAccessEnabled`, `webSearchMode`, `additionalDirectories` |

Claude permission bypass requires `allowDangerouslySkipPermissions: true`.
`allowedTools` grants approval, not a hard filter. Native `env` replaces inheritance
for both providers; without it, the child gets `process.env`. Claude also receives
`CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1` unless `env` sets it and
`CLAUDE_CODE_EFFORT_LEVEL` set to `effort` (blank without it) and a blank
`CLAUDE_CODE_SUBAGENT_MODEL` unless `env` sets them; claude.ai login tokens are removed,
and a login token in `env` is rejected. Claude server-tool results (web
search, advisor) are `tool_result` events. Codex drops `CODEX_ACCESS_TOKEN` and
`OPENAI_API_KEY` from the child (the key reaches it only as `CODEX_API_KEY`, which model
commands see blank; `require` drops both API keys too) and disables shell snapshots, which
would copy the env to `CODEX_HOME`. Codex `error` notices are warnings, emitted with the
next event or before the run's final error; the shell tool is named `command`,
web-search calls are emitted when the search completes, and todo lists appear as thinking.
Host tool callbacks are unsupported. See [API limits](PARITY.md).

## Native events and traces

`onProviderEvent` receives original SDK events, typed `unknown`, including unmapped
frames. Callback exceptions, and rejections of a promise the callback returns, propagate
unchanged from `run()`/`stream()` after the runtime is closed; the trace then has no
`run_finished`. Stopping iteration at `run_finished` releases the Agent and trace file. Import native SDK symbols from their
packages; the wrapper does not re-export them or expose client handles.

```ts
await agent.run({ prompt: "Inspect the project", traceFile: "results/run/trace.jsonl" });
```

Use a distinct path per call, including resumes: each call overwrites its file.
Relative paths use the caller's working directory. Parent directories are created.
Envelopes are written before yielding; interrupted streams leave partial traces.
Validation and runtime checks leave files untouched. Managed bundles are unsupported.

The [session example](examples/continue-session.ts) writes a trace for each call.
View them with `npm run trace-viewer -- results/typescript` from the repository root.

[Test and build commands](VALIDATION.md).
