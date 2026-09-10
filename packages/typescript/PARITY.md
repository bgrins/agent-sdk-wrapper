# Python–TypeScript API parity

The TypeScript package is a native implementation of the same conceptual
interface. This first slice shares a JSON contract subset with Python, not every
method, option, or provider behavior. Differences below are intentional unless
marked as an accounting question. No Python runtime is needed by TypeScript.

## Shared behavior

Both expose Agent run/stream, constructor defaults, provider/model resolution
(including `codex` → `openai`), validation before runtime checks, completed text
and thinking items, normalized envelopes, failed results, explicit session resume,
and opt-in native events. Envelope sequence numbers start at zero. Run-start
events contain the prompt. Input token totals include cache; output totals include
reasoning. Neither portable text stream promises individual token deltas.

The constructor session ID is immediately readable in both implementations.
TypeScript classifies Claude dropped-connection result messages as transient,
native interrupted results as cancelled, and preserves hidden reasoning. A
signal-killed process remains a `ProcessTerminatedError`, never a retry trigger.

## Deliberate differences

| Area | Python | TypeScript and rationale |
|---|---|---|
| Calls and naming | `Agent(...).run(prompt, **overrides)` / `stream(prompt, **overrides)`; snake_case request fields | `new Agent(defaults).run(promptOrRequest)` / `stream(promptOrRequest)`; camelCase requests, snake_case JSON events/results. Native JavaScript conventions; no positional Python compatibility layer. |
| Provider/model resolution | Provider fixed on Agent; explicit provider wins over inference from a bare model name; empty provider/model can mean unspecified | Per-run provider selection; explicit and inferred providers must agree; empty strings are rejected. This prevents accidental cross-provider requests. Changing provider requires replacing incompatible defaults, including model/options/sessionId. `resolveProvider()` returns provider and stripped model together. |
| Retries | Default 2, jittered backoff capped at 8 seconds; stops after normalized progress | Default 0; opt in with `maxRetries`. Configurable initial delay (250 ms), exponential cap 30 seconds, no jitter. Any native frame stops automatic retries, even an unmapped one: the runtime may already have acted. Neither retries terminal error events in place. |
| Sessions/concurrency | Updates Agent's session when `continue_session` is enabled | Records observed IDs per provider even after failures or when continuation is disabled; `continueSession` controls automatic reuse. Explicit IDs win. One active run per instance prevents races in session state. |
| Setup errors | `run()` generally records provider validation/availability failures in a failed result; constructor/request construction errors can still raise | `ConfigError` and `RuntimeUnavailableError` during setup reject before `run_started`. Callers can distinguish invalid setup from a started runtime failure. `checkRuntime()` validates first in both. |
| Cancellation/deadlines | Wall-clock `timeout`, timeout result variants; asyncio cancellation propagates after recording cancellation | `AbortSignal`, including `AbortSignal.timeout(ms)`; cancellation normally returns `status: "cancelled"`. No separate timeout status. Breaking iteration closes the runtime but leaves an incomplete stream with no terminal envelope/result. |
| Failure API | Optional `raise_on_error` / `RunFailedError`; errors expose Python-specific attributes | Inspect `result.status` and error events; no `raiseOnError`. Setup errors and signal-killed runtime errors propagate. Error classes use standard JS `cause` and `retryable`; no numeric process-signal property. |
| Callbacks | `on_event` receives normalized envelopes; `on_provider_event` receives a native-event envelope. Callback exceptions are logged and ignored | `collectRun(stream, onEvent)` awaits a normalized-envelope callback; `onProviderEvent` receives the original native frame synchronously. Exceptions fail/close consumption instead of hiding a broken consumer or trace sink. |
| Effort | Pinned Codex API accepts `none`; enum set differs | Only the pinned SDK's values are accepted, including Codex `ultra`/`persistent` but not `none`. Unsupported model-specific values can still be rejected remotely; no approximate translation. |
| Native settings | Broader provider options and `extra_options`, plus top-level env/tool/web controls | Finite discriminated `providerOptions`; whole-field replacement, no deep merge. Claude preset objects and undefined env values are rejected. Permissions, tool filtering and environment inheritance keep native semantics. |
| Results/helpers | `.ok`, serialization/tool helpers, optional event retention, sync run and context-dump helpers | Plain `RunResult` objects; use `status`, `JSON.stringify` and event filtering. Events are always retained by `collectRun`; consume `stream` directly when retention is unwanted. Helpers can be added when a consumer needs them. |
| Wire vocabulary | Full Python event/status union | Subset omits `structured_output`, `agent_updated`, `subagent_started`, `subagent_ended`, `context_compacted`, timeout status/reason and unknown reason. TS requires tool IDs, tool-call names and run-start prompt where Python permits omissions. It is not a general reader for every Python trace. |

## Features deferred to another slice

Host-defined callable tools, external MCP registration, structured output,
subagent configuration/lifecycle, compaction, managed artifacts/traces and context
dumps are not implemented. Their request keys fail validation rather than being
ignored. `structured_output` and `artifacts_dir` in results stay null for JSON
schema compatibility. System prompt and max turns are available only through
Claude native options; Codex does not expose them through this wrapper yet.

Native activity events do not prove callback support. Claude subagent content
and tool results stay outside portable output. Codex web-search items expose no
result body. Server-side tools and other native-only details may map to warnings
or only appear in `onProviderEvent`; portable permission/tool policies are not
interchangeable. See the README capability matrix and Foofrix follow-up plan.

Claude message retractions explicitly fail with `provider_protocol_error`.
The append-only v1 vocabulary cannot withdraw text or tool results already
delivered. Previously emitted text remains partial output on a failed result.
Supporting retractions needs a shared event-contract extension, or a deliberate
choice to buffer output until all retraction notices arrive; it is not silently
treated as successful output. This limitation applies to both retraction
notifications exposed by the pinned TypeScript SDK.

## Accounting differences and open verification

Claude TS uses `modelUsage`, which includes query-pipeline auxiliary calls;
Python currently uses main-loop result usage. TS reports unavailable model
request counts as 0, while Python uses turn/action counts as proxies. TS can
report a reasoning subset the older Python adapter does not expose. These fields
are not numerically equivalent across packages even for similar prompts.

Codex TS uses the pinned exec runtime's already-inclusive output count and
subtracts per-thread cumulative snapshots. Python currently adds the reasoning
count to app-server output. This is an **open accounting parity question**, not
a reason to add reasoning twice in TypeScript. Verify the pinned Python runtime's
raw accounting before changing Python; then add native-usage regression fixtures
for both. The TS runtime source references are linked from the README.
The installed Python 0.147.0 `TokenUsageBreakdown` declarations expose separate
output/reasoning fields without defining whether output includes reasoning.
The current official [app-server guide](https://learn.chatgpt.com/docs/app-server)
identifies usage notifications but does not settle that arithmetic. These checks
alone do not justify changing Python's accounting.

For external Codex resume with no baseline, both can include prior history; TS
emits a warning. TS retains baselines per thread and handles counter resets with
a warning; Python retains the latest thread and clamps negative deltas. Neither
can promise precise per-run usage after an unreported failed/interrupted turn.
Codex dollar cost remains unavailable.

## What the tests establish

Shared JSON fixtures are validated by Python against the existing schemas and
replayed in both languages. They establish normalized result aggregation and
envelope compatibility for the covered subset, not equivalence of native SDK
events, permissions, token totals, or the entire public API. Mocked native-stream
tests cover adapter mapping separately. Live smoke tests are explicitly gated
and test run/stream/resume; they do not prove callable tool support.
`test:package` additionally compiles/runs an offline consumer against the packed
public API. The [validation guide](VALIDATION.md) separates these checks from
real runtime/credential validation.
