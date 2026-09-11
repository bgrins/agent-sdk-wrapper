# TypeScript API

ESM; Node 22.14+ in the 22.x line, or Node 24+.
Calls the native Claude Agent SDK or Codex SDK directly.
From the repository root: `npm ci && npm run build`.

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

Supply native SDK credentials through environment/login. Codex also accepts
`OPENAI_API_KEY`. The wrapper does not load `.env`.

## Contract

`run()` and `stream()` accept a prompt or `RunRequest`. Constructor defaults are
overridden per field, without deep merging. One active run per Agent.

| Options | Meaning |
|---|---|
| `provider`, `model`, `effort`, `cwd` | Provider/model selection and execution settings; conflicts fail |
| `sessionId`, `continueSession` | Explicit resume or automatic reuse of the latest ID per provider |
| `maxRetries`, `retryDelayMs` | Default 0 retries; transient failures retry only before any events |
| `signal` | Cancellation or `AbortSignal.timeout(ms)` |
| `providerOptions` | Curated native controls below; unknown keys fail |
| `onProviderEvent`, `includeRaw` | Original SDK-event callback, or raw data on mapped events |

`EventEnvelope` has `run_id`, zero-based `sequence`, `timestamp` and `event`.
Events cover run boundaries, sessions, completed text/thinking, tool activity,
usage, warnings and errors. `RunResult` contains status, text, usage/cost,
session ID and events. `collectRun` rejects incomplete or misordered streams.

`checkRuntime()` validates configuration and runtime availability without a model call.
Setup throws `ConfigError` or `RuntimeUnavailableError`; runtime failures usually
produce failed results. Signal-killed processes throw `ProcessTerminatedError`.
Custom `ProviderAdapter`s implement validation, availability and streaming.

## Capabilities

| Capability | Claude | Codex |
|---|---|---|
| Run, stream, resume, effort, cwd | Supported | Supported |
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
`allowedTools` grants approval, not a hard filter. Codex `env` replaces inheritance.
Native tool activity does not imply host callback support. See [API limits](PARITY.md).

## Native events and traces

`onProviderEvent` receives original SDK events, typed `unknown`, including unmapped
frames. Callback exceptions fail the run. Native SDK symbols and client handles
are not re-exported; import native symbols directly when needed.

Write each envelope as a JSONL line, then open it with root `docs/trace-viewer.html`.
The [session example](examples/continue-session.ts) saves its first successful
trace to `results/typescript/trace.jsonl`; later turns are printed only.
There is no automatic artifact manifest or Results-sidebar discovery.

[Validation commands](VALIDATION.md) cover tests, packaging and opt-in live runs.
