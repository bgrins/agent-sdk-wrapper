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
| `allowed_tools`, `disallowed_tools` | Native permission rules | Unsupported; pass only the callable tools you want and filter MCP tools with `enabled_tools`/`disabled_tools` |

Tool names must be unique and match `[A-Za-z0-9_-]{1,64}`. Both providers validate
arguments with the same schema, pass a `**kwargs` tool the arguments it doesn't name and
return a raised exception to the model as `Error: <message>` (the type name when the
message is empty); positional-only parameters are rejected.

Claude pins the child effort, disables background tasks and blanks subagent model and
claude.ai login tokens by default; an explicit login token in `env` is rejected.
MCP servers without `enabled_tools` have all tools pre-approved. Claude
`extra_options` accepts SDK options except keys controlled by first-class options,
`env` and `cli_path`. Thinking defaults to adaptive with summarized display.

Codex drops `CODEX_ACCESS_TOKEN`, blanks API keys in model commands (not MCP servers,
tools or model providers) and disables shell snapshots. Its sandbox applies per thread;
native thread/turn options are validated, and start-only options are dropped on resume.
Caller `config_overrides` take precedence except login-store keys,
`features.shell_snapshot`, replacements for `shell_environment_policy.set`, and
web-search keys when `web_tools` is set. Stopping a run closes the app-server's stdin
to stop its commands.

For a run without built-in tools, Claude takes `extra_options={"tools": []}`. Codex always offers
`apply_patch` and `request_user_input`. Its feature flags remove
the other built-in tools, and a read-only sandbox with `deny_all` approvals refuses every
file write:

```python
Agent(
    provider="openai",
    web_tools=False,
    provider_options={
        "sandbox": "read-only",
        "approval_mode": "deny_all",
        "config": {"config_overrides": [
            "features.shell_tool=false",
            "features.view_image=false",
            "features.goals=false",
            "features.multi_agent=false",
        ]},
    },
)
```

`features.multi_agent=false` also removes the tool search Codex uses to find MCP tools, and
turns off `subagents`. Under the SDK's default `auto_review` approvals, a reviewer model can
approve a write the sandbox blocks. The flags belong to the pinned Codex release;
`codex features list` shows them.

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

Run a single prompt with `uv run python examples/run_basic.py`, or a staged
plan–draft–review flow with `uv run python examples/agent_flow.py`. Set `PROVIDER=codex`
and optionally `MODEL` to choose Codex. The staged flow uses a separate, schema-free
exploration turn before Codex's structured extraction turn; each turn has its own
`artifacts_dir`.
Run tests with `uv run pytest`; see [validation](../typescript/VALIDATION.md) for Compose and live tests.
