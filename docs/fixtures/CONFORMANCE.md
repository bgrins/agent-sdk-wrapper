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
| `prompt` | The run's prompt |
| `mock` | Mock API steps, one per model request; the last repeats |
| `expect` | Offline expectations |
| `languages` | Per-language overrides: `{"typescript": "unsupported: <reason>"}`, or an object with `options`/`expect` to replace, or `{"expect": {"config_error": true}}` |
| `live` | Optional live section: `prompt`, `options` merged over the case options, and `expect` |

## Mock steps

| Step | Meaning |
|---|---|
| `{"text": "..."}` | The model answers with this text |
| `{"thinking": "..."}` | A reasoning block before the step's other output |
| `{"tool": {"name": "...", "input": {...}}}` | The model calls a tool (Claude `tool_use`; Codex `function_call`) |
| `{"shell": "cmd"}` | Codex runs a shell command |
| `{"usage": [input, output]}` | Token usage reported for the step |
| `{"status": 429, "headers": {...}, "body": {...}}` | An HTTP error response |
| `{"stream_error": {...}}` | An error event inside a 200 stream |
| `{"truncate": true}` | The stream closes after its first event |
| `{"hang": seconds}` | No response for this long |

Steps combine where it makes sense, e.g. `{"thinking": "plan", "text": "done", "usage": [10, 2]}`.

## Expectations

| Field | Meaning |
|---|---|
| `status`, `error_type`, `final_text`, `final_text_contains` | Result fields |
| `events.includes`, `events.excludes` | Event types that must or must not appear |
| `tool_calls` | Names of tool calls, in order |
| `structured_output` | Expected structured value |
| `requests.count` | Number of model requests the mock saw |
| `requests.match` | `{"request": n, "path": "a.b", "equals" \| "contains" \| "absent": ...}` on request bodies (JSON), or `{"header": "name", ...}`; `request` defaults to any |
| `config_error` | The run must raise `ConfigError` before any event |
