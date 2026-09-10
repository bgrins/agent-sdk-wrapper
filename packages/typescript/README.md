# agent-sdk-wrapper for TypeScript

A native Node/TypeScript twin of the Python abstraction. The adapters call
`@anthropic-ai/claude-agent-sdk` and `@openai/codex-sdk` directly. Their packaged
native runtimes execute locally; there is no Python process, service, or
cross-language RPC. Requires Node 22.14+ in the 22.x line, or Node 24+; ESM only.

This is the first vertical slice, not full Python feature parity. In particular,
**host-defined callable tools and structured output are not implemented**.

## Build and use locally

From the repository root:

```sh
npm ci
npm run build
npm test
npm run typecheck
npm run lint
npm run format:check
npm run test:package
```

The private root npm workspace delegates to the publishable package here.
The npm name is `agent-sdk-wrapper` (separate registry from PyPI); this change
does not publish a release. To inspect/build a tarball:

```sh
npm run build
npm pack --workspace agent-sdk-wrapper --ignore-scripts --pack-destination /tmp
# In another Node project, install the resulting /tmp/agent-sdk-wrapper-0.1.0.tgz.
```

The tarball contains ESM JavaScript, declarations, the API/parity/validation
guides, and the license. `test:package` compiles and runs a separate consumer
against the extracted tarball using the existing locked dependencies, offline.
Python wheels and sdists do not include the TypeScript package or npm tooling.

```ts
import { Agent, collectRun } from "agent-sdk-wrapper";

const defaults = {
  provider: "codex" as const, // or "anthropic"; alias resolves to "openai"
  cwd: process.cwd(),
  continueSession: true,
};
const agent = new Agent(defaults);

// One model run: render its stream and collect the same events into a result.
const first = await collectRun(agent.stream("Remember the token ALPHA42. Reply READY."), (env) => {
  if (env.event.type === "text") process.stdout.write(env.event.text);
});
if (first.status !== "success") throw new Error(first.error ?? "Run failed");

const savedSessionId = first.session_id; // persist with its canonical provider
const second = await agent.run("What token did I ask you to remember?");
if (second.status !== "success") throw new Error(second.error ?? "Run failed");
console.log(second.status, second.final_text);

// After restarting the application, use the same provider, cwd and native settings:
if (savedSessionId) {
  const resumed = new Agent({ ...defaults, sessionId: savedSessionId });
  const third = await resumed.run("Repeat the token.");
  if (third.status !== "success") throw new Error(third.error ?? "Run failed");
  console.log(third.final_text);
}
```

In a repository checkout, `examples/continue-session.ts` also writes a session record
and viewer-compatible JSONL under the gitignored `results/typescript/` directory:

```sh
npm test
PROVIDER=anthropic node packages/typescript/.test-dist/examples/continue-session.js
PROVIDER=codex node packages/typescript/.test-dist/examples/continue-session.js
```

Open the repository's `docs/trace-viewer.html` in a browser, choose **Open Files**,
and select `results/typescript/trace.jsonl`. The example saves only the first
successful run; continuation/resume turns are printed but not saved. TypeScript
does not yet create artifact manifests, so runs do not appear automatically in
the viewer's Results sidebar. Normalized events already use its JSONL format.

Examples are live/billed. Supply the provider's credentials through its supported
environment/login mechanism. The Codex adapter maps `OPENAI_API_KEY` to native
`apiKey`; an explicit `providerOptions.client.apiKey` takes precedence. Neither
the example nor tests load `.env` automatically.

## Small public contract

`new Agent(defaults, adapters?)` accepts constructor defaults. `run()` and
`stream()` accept a prompt string or a `RunRequest` with `prompt` plus overrides.
Overrides replace entire fields, including `providerOptions`; they are not deep
merged. One Agent supports one active run at a time. Create separate instances
for concurrency. The optional adapter map makes downstream tests fully offline.

| Field | Meaning |
|---|---|
| `provider`, `model` | Provider selection, inference from common model names, or `provider:model`. Conflicting provider/model selections fail. Unknown model names need an explicit provider. The remote service validates model availability. |
| `effort` | Claude: low/medium/high/xhigh/max. Codex: minimal/low/medium/high/xhigh/max/ultra/persistent. Valid enum values may still be rejected by a particular model. |
| `cwd` | Native working directory. Codex requires a Git repository unless its native `skipGitRepoCheck` is enabled. |
| `sessionId`, `continueSession` | Explicit resume, or reuse the most recent emitted ID for that provider. Explicit IDs win; IDs persist after failures too. `agent.sessionId` initially exposes the constructor ID, then the latest observed ID. Sessions remain in the provider's local storage. |
| `maxRetries`, `retryDelayMs` | Default 0 retries, initial delay 250 ms; exponential backoff capped at 30 seconds. Only `TransientError` before any native/normalized frame can retry. |
| `signal` | AbortSignal passed to the native runtime. Cancellation emits a cancelled result. Breaking iteration closes the native iterator/runtime; a partial stream has no `run_finished` or complete `RunResult`. |
| `includeRaw` | Attach raw native messages to mapped events where available; false by default. |
| `onProviderEvent` | Synchronous callback for every original native frame, including unmapped frames. Callback exceptions fail the run without retrying it. |
| `providerOptions` | Discriminated, curated native escape hatch described below. Unknown keys fail validation. |

`checkRuntime()` validates before checking SDK/runtime availability and makes no
model request. `ConfigError` and `RuntimeUnavailableError` are catchable setup
errors. Runtime failures normally become `error` events and failure results;
`ProcessTerminatedError` propagates and is never retried. `TransientError`,
`ProviderError` and `ProviderProtocolError` let custom adapters report failures.
Do not treat a non-throwing `run()` as success: check `result.status`.

`ProviderAdapter` exposes `name`, `validateRequest`, `ensureAvailable`, and
`stream(request, context)`. Adapters emit `ProviderEvent`s and call
`context.onNativeEvent` for every native frame. The Agent owns envelope ordering,
run boundaries, retries, and session persistence. Adapters must reject a stream
that ends without a native terminal result.

Requests use camelCase. JSON events/results deliberately use the Python v1
snake_case wire fields. An `EventEnvelope` has `run_id`, zero-based `sequence`,
UTC `timestamp`, and a discriminated `event`. The initial vocabulary is
`run_started`, `session_info`, `text`, `thinking`, `tool_call`, `tool_result`,
`usage`, `warning`, `error`, and `run_finished`. `text` is a completed item,
not a token delta. `run_started` records the prompt and any native Claude system
prompt. Raw callbacks may contain sensitive prompts/tool output; persistence is
the caller's choice.

`RunResult` contains status/reason, concatenated completed text, session ID,
usage/cost, duration, error, and all envelopes. `structured_output` and
`artifacts_dir` are reserved null fields for schema compatibility. `collectRun`
rejects incomplete/misordered envelopes and can await a per-envelope renderer or
trace writer. It consumes the stream once. Writing `JSON.stringify(envelope)`
per line produces input for the repository trace viewer.

## Capabilities and differences

See [Python–TypeScript API parity](PARITY.md) for behavioral differences, their
rationale, and the limits of the shared fixtures.

| Capability | Claude | Codex |
|---|---|---|
| Basic run, streaming completed items, session resume | Supported | Supported |
| Reasoning effort, cwd, raw native callback | Supported | Supported |
| Text/thinking/tool activity/usage/errors | Supported subset | Supported subset |
| Permissions, builtin tool controls | Provider-specific native options | Provider-specific sandbox/approval options; no portable builtin allowlist |
| System prompt, max turns | Provider-specific native options | Not yet implemented |
| Host-defined callable tools | Not yet implemented in wrapper; SDK has in-process MCP tools | Not yet implemented; SDK has no callback registration API |
| Structured output | Not yet implemented in wrapper; SDK has `outputFormat` | Not yet implemented in wrapper; SDK has `outputSchema` |
| External MCP registration, subagent configuration/lifecycle | Not yet implemented | Not yet implemented |
| Token deltas, compaction events, managed artifacts, timeout setting | Not yet implemented | Not yet implemented |

Passing `tools`, `mcpServers`, `outputSchema`, `subagents`, `systemPrompt`,
`maxTurns`, `builtinTools`, `permissionMode`, `timeout`, or `artifactsDir` at the
portable level throws `ConfigError`, even for an empty value. These fields are
also typed `never`. Use an AbortSignal for a deadline. Native escape hatches
cannot override wrapper-managed model/cwd/resume/partial-output fields or sneak
in the unimplemented structured-output/tool/subagent controls.

The allowed native options are intentionally finite:

```ts
new Agent({
  provider: "anthropic",
  providerOptions: {
    provider: "anthropic",
    options: {
      permissionMode: "dontAsk",
      tools: ["Read", "Glob", "Grep"],
      allowedTools: ["Read", "Glob", "Grep"],
      settingSources: [],
      systemPrompt: "Review the repository.",
      maxTurns: 4,
    },
  },
});

new Agent({
  provider: "codex",
  providerOptions: {
    provider: "openai", // canonical discriminator, even when using the codex alias
    thread: {
      sandboxMode: "read-only",
      approvalPolicy: "never",
      webSearchMode: "disabled",
    },
  },
});
```

Claude additionally allows `disallowedTools`, `env`,
`pathToClaudeCodeExecutable`, `thinking`, and `allowDangerouslySkipPermissions`.
`permissionMode: "bypassPermissions"` requires explicitly setting
`allowDangerouslySkipPermissions: true`; omitting it or setting it to false
throws `ConfigError`. Only a string `systemPrompt`, string array `tools`, and
defined string values in `env` are supported by both types and validation.
Codex `client` allows `apiKey`,
`baseUrl`, `env`, `codexPathOverride`; `thread` additionally allows
`skipGitRepoCheck`, `networkAccessEnabled`, `additionalDirectories`. Codex's
native `env` replaces environment inheritance. Claude's native env follows its
SDK semantics. Claude defaults to isolated `settingSources: []`; Codex still
reads its own native configuration and repository instructions. Existing native
configuration may enable tools beyond this wrapper's registration surface.

Permission policies are not interchangeable. Claude `allowedTools` pre-approves
calls and is not itself a hard tool filter. Codex's sandbox/approval controls do
not implement Claude tool filtering. Tool events describe native activity;
they do not establish support for host callbacks. Tool failures need not fail a
turn if the agent recovers. Codex web-search results have no exposed result body;
the normalized result omits `output`. Claude subagent messages and tool results
are excluded from portable output and remain available in raw callbacks. Other native-only event
details are lossy; there is no byte-for-byte provider equivalence promise.

The v1 event vocabulary cannot retract output already streamed to a consumer.
Claude `supersedes` or `model_refusal_fallback` notifications with retracted IDs
therefore stop the run with `provider_protocol_error`, without retrying. A failed
result can retain previously emitted text; treat it as partial, not a completed
answer. Retraction support needs a shared contract change or buffering policy.
Native `aborted_streaming`/`aborted_tools` results normalize to cancellation;
aborted assistant fragments are not emitted as completed text. Success-subtype
API errors without an HTTP response are classified as retryable upstream errors,
while explicit structural failures and refusals take priority. A retryable error
event still does not trigger an automatic replay after native progress.

Input totals include cached tokens; output totals include reasoning. Claude uses
the query's `modelUsage` totals (including auxiliary calls), folding cache into
input while keeping already-inclusive output intact. Codex's pinned exec runtime
forwards thread-cumulative counts despite its TypeScript type comment. The
adapter subtracts its last observed snapshot per session. Its output already
includes the reasoning subset, so adding reasoning again would double count.
An externally resumed thread with no baseline includes prior history and emits
a warning. A reset in native counters establishes a new baseline with a warning.
Failed/interrupted turns may not report usage; their usage can appear in a later
cumulative delta. Request count is unavailable at this granularity and is 0;
Codex dollar cost is unavailable and is null. Empty/redacted reasoning remains
visible as `thinking`, including a synthesized empty event when either provider
reports reasoning tokens without a corresponding thinking item. Claude's fallback
usage also reads `output_tokens_details.thinking_tokens` without adding it to the
already-inclusive output total.

## Tests and dependency policy

See [Validation](VALIDATION.md) for exact commands, what each check establishes,
the package smoke test, and remaining live-validation requirements.

The default tests use Node's built-in test runner, fake adapters, and mocked
native SDK streams. No API credentials or Python installation are required.
Shared JSON fixtures in `docs/fixtures/native-twin-v1.json` exercise both
implementations. Python's existing JSON Schema tooling validates them against
`docs/schemas/`; TypeScript replays the same fixtures without a validator
dependency. This tests normalized aggregation and envelope compatibility, not
native adapter mapping or complete API parity. Adapter regression tests separately
cover retractions, native cancellation, hidden reasoning, and option validation.
Exhaustive TypeScript checks require every event variant to have a shared schema
branch and replay fixture, and keep result/usage fields aligned with the schemas.

Live tests are separate and run on a Node host, leaving the existing Ubuntu
Python Docker image unchanged. They require both the explicit flag and each
provider's key. Without them the two cases are reported as skipped:

```sh
npm run test:integration
AGENT_SDK_WRAPPER_TS_RUN_INTEGRATION=1 npm run test:integration
```

Export `ANTHROPIC_API_KEY` and/or `OPENAI_API_KEY` first. Optional
`AGENT_SDK_WRAPPER_TS_ANTHROPIC_MODEL` and `AGENT_SDK_WRAPPER_TS_OPENAI_MODEL`
select models; otherwise the native default is used. Each enabled case runs
three turns in a temporary directory: stream/collect, continue, and explicit
resume through a new Agent. The test makes no claim about callable tools.

Runtime dependencies are only the two SDKs; development adds TypeScript,
`@types/node`, and Biome for lint/format. The SDKs bring their own native
binaries and peer/transitive dependencies (including MCP and Ajv through Claude).
There is no wrapper dependency on Ajv or a separate test framework.

Versions were selected from npm release timestamps on 2026-09-09 with a
2026-09-02 UTC cutoff, then installed with lifecycle scripts disabled:

| Direct dependency | Pin | Publication date / reason |
|---|---|---|
| Claude Agent SDK | 0.3.258 | 2026-09-01 |
| Codex SDK | 0.152.1 | 2026-09-01 |
| TypeScript | 7.0.2 | 2026-07-08 |
| Biome | 2.5.11 | 2026-08-27 |
| Node types | 22.20.1 | 2026-07-08; matches the minimum Node major |

For updates, query registry tags and publication times first, select a stable
release at least seven days old, inspect native declarations, regenerate the
lockfile with `npm install --before=YYYY-MM-DD --ignore-scripts`, and run the
checks above plus `npm audit` and `npm audit signatures`. A cooldown, hashes,
signatures, and clean advisory report reduce exposure; none prove absence of
malicious code. `.npmrc` keeps lifecycle scripts disabled for reproducible
installs. Registry versions are never inferred from memory.

## Next slice for Foofrix

Implement one real `record_finding`-style callback with a JSON input contract.
Claude can register it using native `tool()` plus `createSdkMcpServer()`. Codex
needs a Node-native MCP bridge: an in-process loopback MCP server can retain
host closures, while the Codex SDK passes its address through native
`mcp_servers` configuration. Bind locally, scope a token to the run, validate
arguments, preserve call IDs/errors, and close the listener on completion,
cancellation and startup failure. This is MCP between Node and the native Codex
runtime, not cross-language RPC. A stdio server backed by importable Node modules
is another option, but does not itself provide access to host closures.

Before advertising shared tools, test a real MCP initialize/list/call handshake,
callback dispatch, failure propagation and shutdown offline, then prove actual
tool invocation through both live SDKs. Add structured output with explicit
validation separately; a fenced JSON message is never a callable-tool protocol.

After that proof, Foofrix can replace the duplicated provider loops and stream
mapping in `src/commands/optimize.ts` and `src/server/worker.ts`, simplify session
retention and transient-error classification, and delete `CODEX_RECORD_PROTOCOL`
and `extractCodexFinding` for the migrated paths. Keep domain-specific finding
validation, hooks, browser orchestration and sandbox choices in Foofrix. No
Foofrix code changes are part of this slice.

Implementation references: [Claude TypeScript reference](https://code.claude.com/docs/en/agent-sdk/typescript),
[Claude custom tools](https://code.claude.com/docs/en/agent-sdk/custom-tools),
[Codex SDK](https://learn.chatgpt.com/docs/codex-sdk),
[Codex MCP](https://learn.chatgpt.com/docs/extend/mcp?surface=cli),
and the installed declarations/README files for the pinned releases. Codex usage
behavior was checked in the pinned runtime's
[JSONL mapper](https://github.com/openai/codex/blob/rust-v0.152.1/codex-rs/exec/src/event_processor_with_jsonl_output.rs)
and [Responses usage conversion](https://github.com/openai/codex/blob/rust-v0.152.1/codex-rs/codex-api/src/sse/responses.rs).
