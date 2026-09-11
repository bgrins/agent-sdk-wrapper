# API differences

Both packages share run/stream, provider resolution, session resume and normalized
JSON envelopes. TypeScript implements a subset of Python's API.

| Area | Python | TypeScript |
|---|---|---|
| Requests | Keyword overrides; snake_case | Prompt or request object; camelCase; whole-field replacement |
| Provider selection | Fixed per Agent; explicit provider takes precedence | Per-run selection; conflicting model/provider rejected |
| Retries | Default 2; stops after normalized progress | Default 0; stops after any native or normalized frame |
| Session IDs | Automatically updated with `continue_session` | Always recorded per provider; `continueSession` controls reuse |
| Setup failures | Usually failed results; some validation raises | `ConfigError` / `RuntimeUnavailableError` propagate |
| Deadlines | `timeout` and timeout status | `AbortSignal`; cancelled status |
| Failure handling | Optional `raise_on_error` | Check status; setup and signal-killed errors propagate |
| Callback exceptions | Logged and ignored | Fail/close consumption |
| Native callback | Envelope with `.raw` SDK object | Original SDK object, typed `unknown` |
| Tools, structured output, MCP, subagent lifecycle | Supported with provider limits | Not yet implemented |
| Traces | Managed artifact files | Caller writes JSONL |
| Effort | Codex includes `none` | Codex excludes `none`, includes `persistent` |

SDK symbols and internal client/thread handles are not re-exported in either package.
Unsupported options fail validation; native permission policies are not interchangeable.

## TypeScript limits

- Text events contain completed items, not token deltas. Breaking iteration leaves an incomplete run.
- Claude subagent output is available through native callbacks; its tokens are included in totals.
- Claude retractions fail with `provider_protocol_error`; prior text remains partial output.
- Claude custom prompts persist across resume by default. Start a new session to change instructions.
- Codex web-search results expose no result body. Dollar cost is unavailable.
- `structured_output` and `artifacts_dir` stay null. Compaction and Python-only event variants are omitted.

## Token accounting

Input totals include cache; output totals include reasoning. Both Claude adapters
prefer per-model totals, including subagents, and fall back to main-loop usage.
Python request counts are proxies; TypeScript reports unavailable counts as zero.
Python Claude does not expose a separate reasoning-token count.

Codex resume accounting uses cumulative snapshots; an external session without
a baseline can include prior history. Python adds reasoning to raw output;
TypeScript treats output as inclusive. **Python's raw output accounting still
needs verification**, so numerical parity is not established.
