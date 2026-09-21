# Conformance cases

`conformance-v1.json` lists cases that both packages run against the real runtimes. Offline, a
local mock API answers each model request with the next `mock` step and records the request.
Live runs (opt-in, billed) use the same case with its `live` section and skip the mock and request
checks.

## Case fields

| Field | Meaning |
|---|---|
| `id` | Unique `<provider>.<name>` |
| `provider` | `anthropic` or `codex` |
| `options` | Agent options by their Python name; each runner maps them to its own API |
| `run_options` | Per-call overrides and `run()` keywords (`raise_on_error`) for the case's run |
| `prompt` | The run's prompt |
| `setup` | Scratch state created before the run (below) |
| `mock` | Mock API steps, one per model request; the last repeats. Later runs continue the list |
| `expect` | Offline expectations |
| `runs` | Later runs, in order, after the case's run (below) |
| `languages` | Per-language overrides: `{"typescript": "unsupported: <reason>"}`, or an object with `options`/`expect` to replace (options stay Python-named), or `{"expect": {"config_error": true}}`. A string or `config_error` override also skips the live run |
| `live` | Optional live section: `prompt`, `options` merged over the case options, `expect`, and `runs` (merged by index) |

Live runs use the model from `AGENT_SDK_WRAPPER_ANTHROPIC_MODEL` / `AGENT_SDK_WRAPPER_OPENAI_MODEL`,
else `live.options.model`, else `claude-haiku-4-5` / `gpt-5.6-luna`; the offline model is never
used live.

### Option values

Options are JSON; runners turn these into their native values:

| Option | Value |
|---|---|
| `tools` | Names of runner-defined tools: `add(a: int, b: int) -> int` returns `a + b`; `shout(text: str) -> str` returns `text` upper-cased and raises `ValueError("nothing to shout")` for empty text |
| `output_schema` | A runner-defined model: `Weather` is `{city: str, temp_c: int}` |
| `mcp_servers` | Python `McpStdioServer` fields; `"fixture": "simple_mcp_server"` launches `packages/python/tests/fixtures/simple_mcp_server.py` (server `brief_tools`, tool `read_brief`) with the runner's interpreter in place of `command`/`args` |
| `subagents` | Name to Python `SubagentDef` fields |
| `cwd`, `trace_file`, `artifacts_dir` | Paths relative to the case's scratch directory (`cwd` is created) |
| `on_event`, `on_provider_event` | `true` installs a recording callback |
| `provider_options.config` | Codex: merged after the mock wiring (`config_overrides` appended, `env` merged) |
| `provider_options.codex` | `"client"`: a pre-built Codex client the runner starts with the mock wiring |
| any string | `{session_id}` is the previous run's session ID |

Without `cwd`, runners use a scratch `work` directory.

### Setup

| Field | Meaning |
|---|---|
| `files` | `{"<root>/<path>": "<content>"}` with root `cwd`, `home`, `claude_config` (`CLAUDE_CONFIG_DIR`) or `codex_home` |
| `codex_login` | `"chatgpt"`: a stored ChatGPT login Codex accepts offline; `"api_key"`: a stored API-key login |

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
| `{"tool": {"name": "...", "input": {...}}}` | The model calls a tool (Claude `tool_use`; Codex `function_call`, where `mcp__<server>__<tool>` becomes namespace `mcp__<server>` and name `<tool>`) |
| `{"shell": "cmd"}` | The model runs a shell command (Claude `Bash` `{command}`; Codex `exec_command` `{cmd}`) |
| `{"usage": [input, output]}` | Token usage reported for the step; default `[100, 10]` |
| `{"stop_reason": "..."}` | Claude's stop reason; default `tool_use` with a tool, else `end_turn` |
| `{"status": 429, "headers": {...}, "body": {...}}` | An HTTP error response; a string `body` is sent as `text/plain` |
| `{"headers": {...}}` | Extra response headers on any step |
| `{"stream_error": {...}}` | The provider's error object inside a 200 stream: Claude `event: error` with data `{"type": "error", "error": <value>}`; Codex `response.failed` with `response.error = <value>` |
| `{"truncate": true}` | The stream closes after its first event |
| `{"hang": seconds}` | No response for this long |

Steps combine where it makes sense, e.g. `{"thinking": "plan", "text": "done", "usage": [10, 2]}`,
or text followed by a `stream_error`.

The Claude mock serves `POST /v1/messages` (`ANTHROPIC_BASE_URL` without `/v1`); the Codex mock
serves `POST /v1/responses` for a custom model provider with retries off. After a failed stream
the Claude CLI retries once without streaming; that request replays the same step (a
`stream_error` as HTTP 529, 429 or 500 by `error.type`) instead of taking the next one.

## Expectations

| Field | Meaning |
|---|---|
| `status`, `error_type`, `final_text`, `final_text_contains` | Result fields |
| `events.includes`, `events.excludes` | Event types that must or must not appear in the result |
| `events.count` | Number of events the result keeps |
| `tool_calls` | Names of tool calls, in order |
| `tool_results` | `[{"contains": "...", "is_error": bool}]`, in order; `is_error` defaults to `false` |
| `structured_output` | Expected structured value |
| `requests.count` | Number of model requests the mock saw, including the Claude non-streaming retry |
| `requests.match` | `{"request": n, "path": "a.b", "equals" \| "contains" \| "excludes": ...}` or `{"path": "a.b", "absent": true}` on request bodies (JSON), or `{"header": "name", ...}` |
| `same_session` | The result's session ID equals the previous run's |
| `on_event`, `trace_file` | `true`: the callback, or the trace file, received exactly the run's envelopes (the result's sequence numbers and event types) |
| `on_provider_event` | `true`: the callback received at least one native event |
| `raw` | Event types that must appear, each carrying its native `raw` payload |
| `artifacts` | Files that must exist under `artifacts_dir` |
| `raises` | `"RunFailedError"`: `run()` raises it; other expectations apply to its result |
| `setup_error` | The run fails with this error type before any model request (Python returns a failed result; TypeScript may throw) |
| `config_error` | The run must raise `ConfigError` before any event |

`requests.match`: `request` is a 0-based index into the run's model requests; negative counts from
the end. `path` is dot-separated; numeric segments index arrays, negative ones from the end
(`input.-1`). `equals` compares JSON values. `contains` is a substring of a string value, or of the
compact JSON text (`JSON.stringify`) of any other value; `excludes` is its negation and passes
for a missing path; `absent` needs the path missing. Without `request`, `equals` and `contains`
need some request to match, and `excludes` and `absent` need every request to match. Header
names are lowercase.
