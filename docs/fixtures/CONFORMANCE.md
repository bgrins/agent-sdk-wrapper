# Conformance cases

`conformance-v1.json` lists cases that both packages run against the real runtimes. Offline, a
local mock API answers each model request with the next `mock` step and records the request.
Live runs (opt-in, billed) use the same case with its `live` section and skip the mock and request
checks.

`conformance-v1.schema.json` defines the format. Both packages validate the whole file against it,
including cases they skip, so an unknown or misspelled field fails in both languages.

## Case fields

| Field | Meaning |
|---|---|
| `id` | Unique `<provider>.<name>` |
| `provider` | `anthropic` or `codex` |
| `options` | Agent options by their Python name; each runner maps them to its own API |
| `run_options` | Per-call overrides and `run()` keywords (`raise_on_error`) for the case's run |
| `prompt` | The run's prompt |
| `setup` | Scratch state created before the run (below) |
| `mock` | Mock API steps, one per model request (below). Later runs continue the list |
| `expect` | Offline expectations |
| `runs` | Later runs, in order, after the case's run (below) |
| `languages` | Per-language overrides: `{"typescript": "unsupported: <reason>"}`, or an object with `options`/`expect` to replace (options stay Python-named) and a `reason`, such as `{"expect": {"config_error": true}, "reason": "..."}`. A string or `config_error` override also skips the live run |
| `live` | Optional live section: `prompt`, `options` merged over the case options, `expect` in place of the offline one, and `runs` whose fields replace those of the run at the same index |

The top-level `coverage_exemptions` maps each language to `{"<provider>:<option>": "<reason>"}`
for options no case exercises; each package's coverage test fails for an option that neither a
case nor an exemption names. Only cases the language runs count: a string override removes the
case, an override's `options` replace the case's, and a `live` section counts when the language
runs it live.

Live runs never check `requests`. Live runs use the model from the package's own variables (Python
`AGENT_SDK_WRAPPER_ANTHROPIC_MODEL` / `AGENT_SDK_WRAPPER_OPENAI_MODEL`, TypeScript
`AGENT_SDK_WRAPPER_TS_ANTHROPIC_MODEL` / `AGENT_SDK_WRAPPER_TS_OPENAI_MODEL`), else
`live.options.model`, else `claude-haiku-4-5` / `gpt-5.6-luna`; the offline model is never used
live.

### Option values

Options are JSON; runners turn these into their native values:

| Option | Value |
|---|---|
| `tools` | Names of runner-defined tools: `add(a: int, b: int) -> int` returns `a + b`; `shout(text: str) -> str` returns `text` upper-cased and raises `ValueError("nothing to shout")` for empty text |
| `output_schema` | A runner-defined model: `Weather` is `{city: str, temp_c: int}` |
| `mcp_servers` | Python `McpStdioServer` fields; `"fixture": "simple_mcp_server"` launches `packages/python/tests/fixtures/simple_mcp_server.py` (server `brief_tools`, tool `read_brief`) with the runner's interpreter in place of `command`/`args` |
| `subagents` | Name to Python `SubagentDef` fields |
| `cwd`, `trace_file`, `artifacts_dir` | Paths relative to the case's scratch directory (`cwd` is created) |
| `on_event` | `true` passes the runner's recording callback. Runners record every run's envelopes anyway: Python through `on_event`, TypeScript from the stream |
| `on_provider_event` | `true` installs a recording callback |
| `provider_options.config` | Codex: Python passes it with the mock wiring before its `config_overrides`; TypeScript takes only `sandbox_workspace_write.network_access=<bool>` entries, as `networkAccessEnabled` |
| `provider_options.codex` | `"client"`: a pre-built Codex client the runner starts with the mock wiring and `provider_options.config` |

Without `cwd`, runners use a scratch `work` directory.

`{session_id}` in any string of `options`, `run_options` or a run's `options` becomes the session
ID of the latest earlier run that returned a result (empty before one). Prompts and expectations
are never substituted.

TypeScript maps an option only where it has a native counterpart. A case that uses an option
TypeScript lacks fails there until it carries a `languages.typescript` override, which names the
native equivalent in Python terms or skips the case.

### Setup

| Field | Meaning |
|---|---|
| `files` | `{"<root>/<path>": "<content>"}` with root `cwd` (the first run's), `home`, `claude_config` (`CLAUDE_CONFIG_DIR`), `anthropic_config` (`ANTHROPIC_CONFIG_DIR`) or `codex_home` |
| `codex_login` | `"chatgpt"`: a stored ChatGPT login Codex accepts offline (account `acct_1`); `"api_key"`: a stored API-key login (`{"auth_mode": "apikey", "OPENAI_API_KEY": "sk-stored"}`) |
| `codex_provider` | `"builtin"`: the runner defines the mock provider but leaves the built-in OpenAI provider selected; the dead proxy stops its requests |

The Claude CLI uses a stored claude.ai login in `claude_config/.credentials.json`
(`{"claudeAiOauth": {"accessToken": ..., "expiresAt": <ms>}}`) offline when no API key is set.

### Runs

Each entry of `runs` has a `prompt`, optional `options`, `agent` and `expect`. With `agent`
omitted or `"same"`, `options` are per-call overrides on the same Agent. With `"new"`, a new Agent
is built from the case options with `options` merged over them (`provider_options` merged by key).
A run's `requests` expectations see only that run's requests.

## Mock steps

| Step | Meaning |
|---|---|
| `{"text": "..."}` | The model answers with this text |
| `{"thinking": "..."}` | A reasoning block before the step's other output |
| `{"tool": {"name": "...", "input": {...}}}` | The model calls a tool (Claude `tool_use`; Codex `function_call`, where `mcp__<server>__<tool>` becomes namespace `mcp__<server>` and name `<tool>`); a list makes several calls in one step |
| `{"shell": "cmd"}` | The model runs a shell command (Claude `Bash` `{command}`; Codex `exec_command` `{cmd}`) |
| `{"tool_search": "query"}` | Codex: a client `tool_search_call`, which loads deferred tools such as `spawn_agent` |
| `{"usage": [input, output]}` | Token usage reported for the step; default `[100, 10]`. Codex takes a third element, reasoning tokens |
| `{"stop_reason": "..."}` | Claude's stop reason; default `tool_use` with a tool, else `end_turn` |
| `{"status": 429, "headers": {...}, "body": {...}}` | An HTTP error response; a string `body` is sent as `text/plain` |
| `{"headers": {...}}` | Extra response headers on any step |
| `{"stream_error": {...}}` | The provider's error object inside a 200 stream: Claude `event: error` with data `{"type": "error", "error": <value>}`; Codex `response.failed` with `response.error = <value>` |
| `{"truncate": true}` | The connection drops mid-response, like a lost connection: a stream after its first event, without the chunked body's terminating chunk; a non-streaming Claude response partway through its declared `content-length` |
| `{"hang": seconds}` | No response for this long |

Steps combine where it makes sense, e.g. `{"thinking": "plan", "text": "done", "usage": [10, 2]}`,
or text followed by a `stream_error`.

Streams are chunked server-sent events. Once the steps run out, the last step answers 5 more
requests; later requests get HTTP 400 with message `conformance mock: steps exhausted` (Claude
`{"type": "error", "error": {"type": "invalid_request_error", ...}}`, Codex
`{"error": {"type": "invalid_request_error", ...}}`), so a runtime stuck in a loop fails within
seconds.

The Claude mock serves `POST /v1/messages` (`ANTHROPIC_BASE_URL` without `/v1`); the Codex mock
serves `POST /v1/responses` for a custom model provider with retries off. Offline runs set
`ANTHROPIC_API_KEY=sk-ant-mock` or `OPENAI_API_KEY=sk-mock-key`. After a failed stream
the Claude CLI may stream again (each attempt takes the next step) and then retries once without
streaming; a non-streaming request replays the step of the latest streaming one (a
`stream_error` as HTTP 529, 429 or 500 by `error.type`) instead of taking the next one. The number
of these attempts varies between CLI versions, so fault cases assert counts only where they agree.

## Expectations

An expectation is one of three kinds: `{"config_error": true}` alone, `setup_error` with at most
`files_unchanged`, or any of the other fields.

| Field | Meaning |
|---|---|
| `status`, `error_type`, `final_text`, `final_text_contains` | Result fields |
| `events.includes`, `events.excludes` | Event types that must or must not appear in the result |
| `events.count` | Number of events the result keeps |
| `tool_calls` | Names of tool calls, in order; or `{"includes": [...], "excludes": [...]}` |
| `tool_results` | `[{"contains": "...", "is_error": bool}]`, one per tool result, in order; `contains` is optional and `is_error` defaults to `false` |
| `structured_output` | Expected structured value, compared like `equals` below |
| `requests.count` | Number of model requests the mock saw, including the Claude non-streaming retry |
| `requests.match` | `{"request": n, "path": "a.b", "equals" \| "contains" \| "excludes": ...}` or `{"path": "a.b", "absent": true}` on request bodies (JSON), or `{"header": "name", ...}` (below) |
| `same_session` | `true`: the result's session ID equals that of the latest earlier run with a result; `false`: it differs |
| `on_event`, `trace_file` | `true`: the recorded envelopes, or the trace file (`trace_file`, else `artifacts_dir/trace.jsonl`), have sequences from 0 without gaps, run from `run_started` to `run_finished`, and match the result's sequences and event types when it keeps events. TypeScript does not check `on_event`: it reads envelopes from the stream, whose order `collectRun` enforces |
| `on_provider_event` | `true`: the callback received at least one native event |
| `raw` | Event types that must appear, each carrying its native `raw` payload |
| `artifacts` | Files that must exist under `artifacts_dir`; TypeScript fails a case that expects them |
| `raises` | `"RunFailedError"`: `run()` raises it; other expectations apply to its result. TypeScript fails a case that expects it |
| `files_unchanged` | Setup paths (`<root>/<path>`) whose contents after the run equal their contents before the case's first run; a missing file must stay missing |
| `setup_error` | The run fails with this error type before any model request: Python returns a failed result, TypeScript returns one or throws |
| `config_error` | The run raises `ConfigError` before any event and any model request |

`requests.match`: `path` is dot-separated with non-empty segments. On an array, a segment of
digits with an optional `-` indexes it, negative from the end; on an object, a segment names an
own key; anything else is a missing path. `request` is a 0-based index into the run's model
requests, negative from the end; an index outside them fails. `equals` is JSON value equality:
object key order is irrelevant, arrays compare in order, and types are strict (`true` is not `1`,
`"1"` is not `1`; `1` equals `1.0`). `contains` is a substring of a string value, or of the compact
JSON text (`JSON.stringify`) of any other value; `excludes` is its negation and passes for a
missing path; `absent` needs the path missing. Without `request`, `equals` and `contains` need
some request to match, so they fail when the run made none; `excludes` and `absent` need every
request to match, so they pass when it made none. Header names are lowercase.
