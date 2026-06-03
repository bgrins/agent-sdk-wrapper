# Agent Notes

Repo-specific invariants for editing `agent-sdk-wrapper`.

- Keep provider SDK imports inside `src/agent_sdk_wrapper/providers/`.
- Keep Docker Ubuntu-based. Do not switch to Alpine/musl.
- Do not install Node/npm, Claude Code, or a standalone Codex CLI in Docker;
  runtimes come from the Python SDK packages.
- Keep `.env`, `.claude/`, `results/`, caches, and generated artifacts out of
  git.
- Provider/model resolution belongs in the wrapper. `codex` is an alias for the
  `openai` adapter.
- Unsupported provider combinations should raise `ConfigError`, not silently
  degrade.
- `Agent.check_runtime()` validates the request before checking provider
  availability.
- Default tests must stay offline. Live tests run through Docker Compose with
  provider credentials.
