# API differences

Both packages share run/stream, provider resolution, session resume, credential
policy, the error vocabulary and normalized JSON envelopes. TypeScript implements
a subset of Python's API.

| Area | Python | TypeScript |
|---|---|---|
| Requests | Keyword overrides; snake_case | Prompt or request object; camelCase; whole-field replacement |
| Provider selection | Fixed per Agent; explicit provider takes precedence | Per-run selection; conflicting model/provider rejected |
| Concurrency | Concurrent runs allowed with distinct `artifacts_dir`/`trace_file`; `continue_session` keeps the last reported session | One active run per Agent |
| Codex session model | Reported | Not exposed by `codex exec` |
| Codex key check | Skipped with a custom `model_provider` | Required, including with `baseUrl` |
| Claude text at a deadline or cancel | A message still receiving frames is dropped | Kept |
| Claude stop latency | Up to 5 s after a deadline or cancel (SDK close grace); an early close returns before the CLI exits | About 2 s |
| Session IDs | Automatically updated with `continue_session` | Always recorded per provider; `continueSession` reuses the latest reported session, even after a constructor `sessionId` |
| Setup failures | `ConfigError` raises at call; missing runtime or credentials give failed results | `ConfigError`, `RuntimeUnavailableError` and `ProviderError` throw |
| Deadlines | `timeout` bounds provider waits; timeout status | `AbortSignal`; cancelled status |
| Failure handling | Optional `run(raise_on_error=True)`; `RunFailedError.result` | Check status; setup errors throw |
| Callback exceptions | Logged and ignored | Propagate unclassified after closing the runtime, including rejected callback promises |
| Native callback | Envelope with `.raw` SDK object | Original SDK object, typed `unknown` |
| Native `env` | Merged over the parent environment | Replaces the parent environment |
| Codex config | `config_overrides` strings, after the wrapper's | `client.config` tables, merged over the wrapper's |
| Codex tool filters | `McpServer.enabled_tools`/`disabled_tools`; `allowed_tools`/`disallowed_tools` rejected | No MCP servers; native `config` can set `mcp_servers` |
| Tools, structured output, MCP, subagent lifecycle | Supported with provider limits | Not yet implemented |
| Traces | `trace_file` and managed artifacts | `traceFile`; no managed artifact bundle |
| Effort | Codex includes `none`; input is lowercased | Codex includes `persistent`; input is exact |
| Redacted thinking | Size from the thinking signature | Size from `redacted_thinking` blocks |
| `cli_login="require"` check | Codex account type after startup | `codex login status` before startup |

Signal-killed runtimes record a `process_terminated` error, then raise. Exit codes
129–159 and negative codes count as signal kills, including a Codex app-server killed
during startup. TypeScript reads the exit code from the error text; a signal name in the
text counts only when no exit code is given.

`cli_login` defaults to `deny`: stored logins are never used and credentials are
never persisted by the wrapper. Codex shell snapshots, which would copy the env to
`CODEX_HOME`, are disabled, `CODEX_ACCESS_TOKEN` is dropped under both policies, and
model commands see blank API keys (`shell_environment_policy.set`) while MCP servers,
tools and model providers keep them. Claude model commands can read `ANTHROPIC_API_KEY`,
and the Claude CLI stores command output in its session transcript
(`CLAUDE_CONFIG_DIR/projects/…`), so a command that prints the key writes it there.
Claude rejects `require`.

Both packages share one error classifier for message text and HTTP status, after each
adapter's structured native signals, and apply it to raised exceptions too, else
`provider_exception`; `docs/fixtures/error-classification-v1.json` holds the cases both
replay.

SDK symbols and internal client/thread handles are not re-exported in either package.
Unsupported options fail validation; native permission policies are not interchangeable.

## Error types

| `error_type` | Meaning |
|---|---|
| `transient_api_error` | 408, 409, 429, 5xx, overload, high demand or dropped connection |
| `authentication_failed`, `permission_denied` | Missing or rejected credentials; access denied |
| `invalid_request`, `model_not_found`, `context_window_exceeded` | Request rejected |
| `billing_error`, `usage_limit_exceeded`, `max_budget` | Spending or quota limits |
| `max_turns`, `refused`, `cancelled`, `timeout` | Run limits, structured refusals, cancellation, Python deadline |
| `structured_output_failed`, `execution_error` | Structured output or runtime execution failures |
| `provider_protocol_error`, `runtime_unavailable`, `process_terminated`, `provider_exception` | Wrapper-level failures |
| `api_error_<status>` | Other HTTP statuses |

## Retrying

Neither package retries a run; the runtimes already retry API errors with backoff.
`RunResult.error_type` is the first error's type. A caller that runs again after
`transient_api_error` should first check the events: a run that started a tool call
may repeat its side effects, and running a resumed session again repeats the prompt
in that session.

## Shared limits

- Text events are completed assistant messages, not token deltas.
- The packages bundle different Claude CLI builds (Python 2.1.259, TypeScript 2.1.268), so
  runtime behavior can differ; for example, after a stream fails following partial text,
  2.1.268 streams again and repeats the partial text as more Text events.
- `effort` has no effect where the model does not take it: the Claude CLI sends a fixed
  thinking budget for `claude-haiku-4-5` and `claude-sonnet-4-5`.
- Claude subagent messages are omitted with a warning; their tokens are included in totals.
- Claude background subagents are disabled, so a run has one result. The `Workflow` tool
  still runs in the background; a run that allows it can end at its first result,
  before the workflow finishes.
- Claude retractions fail with `provider_protocol_error`; prior text remains partial output
  and usage is still reported. Only `scope: "local"` notices and subagent `supersedes`
  frames are ignored. Python sees only `model_refusal_fallback` notices, because its SDK
  drops the `supersedes` and `aborted` fields.
- After a Claude model fallback, text from the original model comes first, then a warning
  (the CLI's `content`, else `Claude fell back from <original> to <fallback>`), then a
  `session_info` with the fallback model. A `model_refusal_fallback` that retracts nothing
  counts as a fallback; a `scope: "local"` fallback is ignored. A session ID that changes
  mid-run emits a new `session_info`.
- Claude built-in and defined subagents without a model inherit the run's model:
  `CLAUDE_CODE_SUBAGENT_MODEL` is blanked unless `env` sets it.
- Python emits no Claude web-search or web-fetch results: its SDK drops those blocks.
  TypeScript emits every server-tool result as a `tool_result`.
- Custom system prompts persist across resume in both runtimes; Codex ignores a new
  `system_prompt` on resume. Start a new session to change instructions.
- Configured MCP servers are trusted: Python Claude pre-approves every tool of a server,
  or only its `enabled_tools` when set (`disabled_tools` stay unavailable), and Python
  Codex defaults `default_tools_approval_mode` to `"approve"` (otherwise the default
  `auto_review` asks the model to judge each call, and `deny_all` rejects every call).
- Codex defers MCP tools behind its tool search; prompts may need to tell the model to search.
- Runtimes retry before the wrapper sees an error. By default the Claude CLI retries 429, 5xx, 529 and 401 ten times over about 3 minutes, and
  each retry becomes a warning; `CLAUDE_CODE_MAX_RETRIES` in `env` sets the count. Codex
  retries 5xx and dropped streams (`request_max_retries`, `stream_max_retries` on a custom
  `model_providers` entry), but not HTTP 429.
- Codex drops the body of an HTTP 429, so an exhausted API quota is reported as
  `transient_api_error`.
- When Codex retries a stream after a message completed, both messages are Text events.
- Claude reports `rate_limit_event` only for claude.ai logins, so API-key runs get no
  rate-limit warnings.

## TypeScript limits

- Breaking iteration leaves an incomplete run.
- Codex web-search results expose no result body.
- `structured_output` and `artifacts_dir` stay null. Compaction and Python-only event variants are omitted.
- Codex model commands keep running after a cancel, timeout or callback failure: `codex exec`
  exits on SIGTERM without stopping them. Python closes the app-server's stdin first, which
  stops them.

## Token accounting

Input totals include cache; output totals include reasoning. Both Claude adapters
prefer per-model totals, including subagents, fall back to main-loop usage, and
report thinking tokens and `num_turns` requests. Dollar cost is Claude-only.

Codex output includes reasoning; cache-write tokens are reported. Codex usage excludes
subagent threads and the model calls of the default `auto_review` approval mode.
Python counts each request of a turn, including requests that stream no item, from the
runtime's per-request usage, so resumed history and failed attempts are excluded; a turn
with no completed request reports no usage. TypeScript diffs cumulative snapshots per session and warns when a resumed
thread has no baseline; it reports zero requests.
