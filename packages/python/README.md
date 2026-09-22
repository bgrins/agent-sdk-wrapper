# Python API

Requires Python 3.12+. Run `uv sync --extra dev` from this directory.

## Run and stream

```python
import asyncio
from agent_sdk_wrapper import Agent

async def main():
    agent = Agent(provider="codex", continue_session=True)  # or "anthropic"
    result = await agent.run("Remember ALPHA42. Reply READY.")
    if result.status != "success":
        raise RuntimeError(result.error)
    print(result.final_text)

    async for event in agent.stream("Repeat the token."):
        print(event.to_json())

asyncio.run(main())
```

Claude needs `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN` or an enabled cloud-provider
flag such as `CLAUDE_CODE_USE_BEDROCK=1`; a keyless `ANTHROPIC_BASE_URL` gateway needs a
placeholder key. Codex needs `OPENAI_API_KEY` (in `env` or the process) or
`provider_options={"api_key": ...}`, unless a custom `model_provider` is selected.
With the default `cli_login="deny"`, a run never uses a runtime's stored login and
fails before starting without credentials. Codex accepts `cli_login="require"` to use
its stored ChatGPT login instead; Claude rejects it. Codex keeps API keys in memory, not
`auth.json`. A pre-built Codex client or `launch_args_override` owns its credential
store; `cli_login` then checks only the account type. Claude loads no on-disk settings
or `CLAUDE.md` unless `setting_sources` names them.

To resume in another process, save `result.session_id` and pass it as
`Agent(provider=..., session_id=...)` with the same working directory.

## Options

Constructor keywords are defaults; `run()` and `stream()` accept per-call overrides.
`RunRequest` is the resolved request passed to adapters. A `provider:` prefix in `model`
counts only for `anthropic`, `openai` or `codex`, so IDs like
`us.anthropic.claude-sonnet-4-5-20250929-v1:0` stay whole.

| Options | Purpose |
|---|---|
| `provider`, `model`, `effort`, `cwd` | Provider/model selection and execution settings; `cwd` must be an existing directory |
| `tools`, `mcp_servers`, `subagents` | Callable tools, external MCP and subagents |
| `output_schema` | Validated structured output |
| `session_id`, `continue_session` | Explicit resume or automatic continuation |
| `timeout`, `max_turns` | Provider-wait deadline and turn limit |
| `cli_login`, `setting_sources` | Stored-login policy; Claude on-disk settings (default none) |
| `provider_options`, `extra_options` | Provider-specific settings; unsupported combinations fail |

`Agent()` raises `ConfigError` for unknown `provider_options` keys; `run()` and `stream()`
raise it for invalid settings before any event, including duplicate MCP server names or the
reserved `agent_sdk_wrapper_tools`, a bare string where a list is expected, and output
paths of the wrong type. Other failures produce failed results; check `result.status`, or
call `run(..., raise_on_error=True)` for `RunFailedError`, whose `.result` is the failed
`RunResult`. Signal-killed runtimes are recorded, then raise `ProcessTerminatedError`;
from `run()`, its `.result` is the failed `RunResult`. Runs are never retried; the runtimes retry API
errors themselves ([limits](../typescript/PARITY.md#shared-limits)). `timeout` is a deadline from
the start of the run: consumer code is never cancelled, but the next provider wait
after it fails with `timeout`. Concurrent runs on one Agent are allowed but need distinct
`artifacts_dir` and `trace_file`; with `continue_session` the Agent keeps the last session
a run reported, so a new run may resume a session another run is still using. `check_runtime()`
validates settings, runtime and credentials (for Codex, not under `require` or with a
custom `model_provider`). `run_sync()` works outside an event loop.

`result.final_text` is the last assistant message. `result.error_type` is the first
error's type, from the [shared vocabulary](../typescript/PARITY.md#error-types).
`transient_api_error` marks a failure worth running again; see
[retrying](../typescript/PARITY.md#retrying) for what a re-run repeats.

| Capability | Claude | Codex |
|---|---|---|
| Callable tools | In-process MCP; pydantic-validated arguments; sync tools run in threads | Stdio MCP the model may need to find with Codex tool search; functions must be importable or self-contained; 600 s default timeout |
| Structured output | Supported | Strict JSON schema: object root, no free-form dicts or `Any`; optional fields are sent as nullable |
| External MCP, resume | Supported | Supported |
| Subagents | Native definitions; background tasks disabled | Native multi-agent config; per-subagent tools/turn limits rejected |
| `max_turns` | Native turn limit | Unsupported; Codex has no turn limit |
| Built-in tool filtering | Native controls | Unsupported; `web_tools` sets the `web_search` mode |

Tool names must be unique and match `[A-Za-z0-9_-]{1,64}`. Both providers validate
arguments with the same schema, pass a `**kwargs` tool the arguments it doesn't name and
return a raised exception to the model as `Error: <message>` (the type name when the
message is empty); positional-only parameters are rejected.

The Claude child env sets `CLAUDE_CODE_EFFORT_LEVEL` to `effort` (blank without it; the
CLI ranks it above `--effort`), sets `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1` unless
`env` sets it, blanks `CLAUDE_CODE_SUBAGENT_MODEL` unless `env` sets it (so subagents
inherit the run's model), and blanks claude.ai login tokens; a login token in `env` is
rejected. MCP servers without `enabled_tools` have all their tools pre-approved.
Claude `extra_options` must be Claude Agent SDK options, and cannot set a key a
first-class option sets, nor `env` or `cli_path` (use `provider_options`); `tools` counts as set whenever `builtin_tools`,
`web_tools=True` or `subagents` is. Thinking defaults to adaptive with summarized display
unless `extra_options` sets `thinking` or `max_thinking_tokens`. Codex drops
`CODEX_ACCESS_TOKEN` from the runtime env, since it would replace the stored login. Model
commands get blank `OPENAI_API_KEY` and `CODEX_API_KEY` through
`shell_environment_policy.set`, while MCP servers, tools and model providers keep them.
Shell snapshots, which would copy the env to `CODEX_HOME`, are disabled. Codex
`env_passthrough` names are read from the run env. Codex `sandbox` applies per thread;
`thread_options`/`turn_options` override first-class values, start-only thread options
are dropped on resume, and native option values are checked against the SDK's params
before launch. Caller `config_overrides` take precedence, except login-store keys, config
that replaces those `shell_environment_policy.set` entries and, with `web_tools`,
web-search keys, which fail. A stopped Codex run closes the app-server's stdin and gives it
up to 2 s to exit, which stops its commands.

See [examples](examples/) and [API differences](../typescript/PARITY.md).

## Events and traces

`EventEnvelope` contains `run_id`, `sequence`, `timestamp` and `event`.
`RunResult` contains status, text, usage/cost, session ID and retained events.

- `on_event`: normalized envelopes.
- `on_provider_event`: native envelopes; `.raw` is the SDK object, `.message` is serialized.
- `trace_file`: normalized JSONL.
- `artifacts_dir`: trace, result, manifest and native-event files. The manifest records
  status, `ended_reason`, `error_type` and the reported model; its file paths are relative
  to `artifacts_dir`, or absolute for files outside it. Native-event lines carry `run_id`;
  each run replaces the file.

From the repository root, run `npm run trace-viewer -- packages/python/results`.
Open the printed URL and select a trace.
Claude token totals include subagents via `model_usage`; `requests` remains a
main-loop turn-count proxy. Codex usage is per turn; its cost is unavailable.
`SessionInfo.model` reports the model the runtime actually used; after a Claude model
fallback, the original model's text, then a warning, precede a new `SessionInfo`, and a
changed session ID also emits one.
[Accounting limits](../typescript/PARITY.md).

CLI: `uv run agent-sdk-wrapper run --provider codex --prompt "Say hello" --output jsonl`.
`--stream` requires `--output text`; `--cli-login` and repeatable `--setting-source`
set those options. `--config` reads a TOML or JSON file whose keys are `Agent` keywords
(except tools, `output_schema`, callbacks and `continue_session`) plus `prompt`,
`prompt_file`, `output` and `stream`; `mcp_servers` entries are `McpStdioServer` or, with a
`url`, `McpHttpServer` fields. Values must match the keyword's type exactly: `"false"` is
not a boolean and a string is not a list. Relative `cwd`, `trace_file`, `artifacts_dir`,
`prompt_file` and `mcp_servers[].cwd` paths resolve against the file. Flags replace its
values, except that `--env`, `--provider-option` and `--extra-option` merge by key.
Exit codes: 1 failed run, 2 invalid settings, 128+N killed runtime (`--output json` or
`text` prints the result first).
Run tests with `uv run pytest`; see [validation](../typescript/VALIDATION.md) for Compose and live tests.
