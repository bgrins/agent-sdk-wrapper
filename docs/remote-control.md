# Remote Control Protocol (Spike)

Status: experimental spike on the `remote-control-spike` branch.

Instead of wrapping each agent SDK behind a one-shot `run()`/`stream()` call,
this spike exposes long-lived agent **sessions** behind a small HTTP + SSE
protocol, plus a dependency-free web UI to drive them. The goal is to remote
control either agent (start / stop / steer / resume / answer permission
prompts) while reusing each SDK's native session primitives — similar in
spirit to Claude Code's "remote control" feature, but self-hosted and
provider-agnostic.

## Native remote-control surfaces

### Claude (claude-agent-sdk)

The Python SDK drives the bundled Claude Code CLI over a JSON-lines stdio
control protocol (`control_request` / `control_response`). The SDK surface is
official; the wire protocol underneath is internal and version-coupled, so we
build on `ClaudeSDKClient`, never raw stdio.

| Capability | API | Notes |
|---|---|---|
| Start session | `ClaudeSDKClient.connect()` (streaming-input mode) | |
| Send / steer | `client.query(text)` | Mid-turn sends are queued/steered natively |
| Interrupt | `client.interrupt()` | Cancels turn, keeps session |
| Permission proxy | `can_use_tool` callback → `PermissionResultAllow/Deny` | Async callback; ideal for remote UIs. Requires streaming-input mode |
| Runtime mode/model switch | `set_permission_mode()`, `set_model()` | |
| Resume after process exit | `ClaudeAgentOptions(resume=session_id)` | Transcripts in `~/.claude/projects/<key>/`; pluggable `SessionStore` exists |
| Fork | `fork_session=True` | |
| Events | `receive_messages()` (Assistant/User/System/Result messages) | `include_partial_messages=True` for token deltas |
| Hooks | `hooks={...}` (PreToolUse, PostToolUse, Stop, …) | Extra interception/audit surface |
| Usage/cost | `ResultMessage.usage`, `total_cost_usd`, `get_context_usage()` | |

Claude Code's own "Remote Control" (`claude remote-control`) is a closed,
relay-based research preview (outbound-only HTTPS to Anthropic, claude.ai
frontend). Not reusable for a self-hosted UI — the SDK control protocol is the
reusable surface.

### Codex (openai-codex)

The Python SDK spawns `codex app-server` (JSON-RPC 2.0 over stdio) — the same
protocol the TUI, IDE extensions, and mobile remote use. The protocol is
documented and schema-generated but explicitly evolving; the SDK pins a CLI
binary. The legacy `codex proto` submission/event protocol is effectively
deprecated; `codex exec --json` is fire-and-forget only (no steer/approvals).

| Capability | API (SDK) | Underlying RPC |
|---|---|---|
| Start session | `AsyncCodex.thread_start()` | `thread/start` |
| Send | `thread.turn(input)` → `AsyncTurnHandle` | `turn/start` |
| Steer mid-turn | `turn.steer(input)` | `turn/steer` |
| Interrupt | `turn.interrupt()` | `turn/interrupt` |
| Approval proxy | `approval_handler(method, params)` → `{"decision": ...}` | `item/commandExecution/requestApproval`, `item/fileChange/requestApproval` |
| Resume after process exit | `thread_resume(thread_id)` | `thread/resume`; rollout JSONL in `~/.codex/sessions/` |
| Fork / rollback | `thread_fork()` / (`thread/rollback`) | |
| Events | `turn.stream()` typed notifications | `item/*` deltas + `item/started`/`completed`, `turn/completed`, `thread/tokenUsage/updated` |
| List threads | `thread_list()` | `thread/list` |

## Unified wire protocol (this spike)

HTTP + Server-Sent Events, served by `agent-sdk-wrapper serve`
(`src/agent_sdk_wrapper/remote/`). One `session_id` names a live wrapper
session; the provider-native id (Claude session id / Codex thread id) is
surfaced for resume.

| Endpoint | Meaning | Claude mapping | Codex mapping |
|---|---|---|---|
| `POST /api/sessions` `{provider, model?, cwd?, resume?, permission_mode?}` | Start (or resume) a session | `ClaudeSDKClient.connect()`; `resume=` option | `thread_start()` / `thread_resume()` |
| `GET /api/sessions` | List live sessions | — (wrapper state) | — (wrapper state) |
| `GET /api/sessions/{id}` | Session status | wrapper state | wrapper state |
| `GET /api/sessions/{id}/events?since=N` | SSE event stream (replayable) | normalized `receive_messages()` | normalized `turn.stream()` |
| `POST /api/sessions/{id}/messages` `{text}` | Send if idle, steer if running | `client.query()` (native steering) | `thread.turn()` if idle, `turn.steer()` if running |
| `POST /api/sessions/{id}/interrupt` | Stop current turn, keep session | `client.interrupt()` | `turn.interrupt()` |
| `POST /api/sessions/{id}/permissions/{request_id}` `{behavior: allow\|deny, message?}` | Answer a pending permission/approval | resolve `can_use_tool` future | resolve `approval_handler` future |
| `DELETE /api/sessions/{id}` | Close session (resumable later via native id) | `client.disconnect()` | `codex.close()` |

SSE events are JSON envelopes `{seq, timestamp, event}`. Normalized agent
events reuse the existing `events.py` vocabulary (`text`, `thinking`,
`tool_call`, `tool_result`, `session_info`, `usage`, `error`). The remote layer
adds control events: `state_changed` (`starting|idle|running|awaiting_permission|closed`),
`permission_request` (`request_id`, `tool`, `input`), `permission_resolved`,
`user_message` (echo for multi-client sync), `turn_started` / `turn_finished`.

The web UI (`GET /`) is a single dependency-free HTML page: create/resume
sessions, live event log, send/steer input, interrupt, and permission prompt
cards with allow/deny.

## Gaps and challenges

1. **Codex approval callbacks run on the SDK's reader thread**
   (`CodexClient._reader_loop` → `_handle_server_request`). Blocking there for
   a human answer stalls *all* notifications and RPC responses — including the
   `turn/interrupt` response — so a naive human-in-the-loop handler deadlocks
   interrupt-while-approval-pending. The spike works around this by resolving
   any pending approval as `decline` before sending an interrupt. A proper fix
   needs the SDK to answer server requests asynchronously.
2. **`AsyncCodexClient` doesn't expose `approval_handler`** — only the sync
   `CodexClient` constructor takes it. The spike reaches into
   `AsyncCodex._client._sync._approval_handler` (private API).
3. **Codex high-level `ApprovalMode` can't express human-in-the-loop**: only
   `auto_review` (agent answers its own approvals) and `deny_all`. Getting
   `approval_policy=on-request` *without* the auto-reviewer requires building
   generated `ThreadStartParams` directly.
4. **Steering semantics differ**: Claude's `query()` mid-turn is a real
   queued user message; Codex `turn/steer` injects input into the *current*
   turn (and fails if the turn just completed — race). The wrapper retries as
   a new turn on that race.
5. **Permission model mismatch**: Claude prompts per *tool call* with rich
   allow/deny + input-rewrite semantics (`updated_input`, `updated_permissions`);
   Codex prompts per *command/patch escalation* with
   `accept/acceptForSession/decline/cancel`. The unified `allow|deny` maps to
   the lossy common subset.
6. **Turn/idle state is inferred**: Claude signals end-of-turn via
   `ResultMessage`; Codex via `turn/completed`. Neither SDK exposes a direct
   "is busy" query, so `state_changed` is wrapper bookkeeping and can be
   momentarily stale.
7. **Resume asymmetry**: both resume across process exits from local disk
   (`~/.claude/projects/`, `~/.codex/sessions/`), but listing resumable
   sessions is asymmetric: Codex has `thread/list`; Claude's SDK offers
   offline `list_sessions()` utilities keyed by project directory.
8. **Protocol stability**: both native surfaces are official-but-evolving.
   Claude's control protocol is internal to the SDK (pin `claude-agent-sdk`);
   Codex app-server schemas drift per CLI version (pin `openai-codex-cli-bin`,
   regenerate types when bumping).
9. **No auth story in the spike**: the server binds localhost with no
   authentication; anything beyond localhost needs TLS + tokens before it is
   real "remote" control.
10. **Token-level streaming**: enabled for neither provider in the spike
    (Claude `include_partial_messages` and Codex `item/*/delta` both support
    it) — completed items only, to keep normalization identical.

## State and persistence

Both providers persist sessions as append-only JSONL that resume-by-id
replays:

- Claude: `$CLAUDE_CONFIG_DIR/projects/<cwd-dashified>/<session-id>.jsonl`
  (default `~/.claude`). The project key derives from the *resolved* cwd, so
  restores must land on the same workspace path.
- Codex: `$CODEX_HOME/sessions/YYYY/MM/DD/rollout-<ts>-<thread-id>.jsonl`
  (default `~/.codex`), plus a versioned SQLite index used only for
  `thread/list`. `auth.json` lives in the same directory.

`agent-sdk-wrapper serve --state-dir DIR` relocates both: `DIR/claude`
becomes `CLAUDE_CONFIG_DIR` and `DIR/codex` becomes `CODEX_HOME` for spawned
runtimes (both SDKs merge these over the inherited environment). This makes
session state a single syncable directory — to survive a VM restart, snapshot
the state dir plus the workspaces and resume by native id. Wrapper-level
session metadata (session id → provider/native id/cwd) is in-memory only;
persisting that manifest is future work, as is backing Claude with the SDK's
pluggable `SessionStore` instead of directory relocation.
