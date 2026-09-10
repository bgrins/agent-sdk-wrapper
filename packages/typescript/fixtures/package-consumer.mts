import assert from "node:assert/strict";
import {
  Agent,
  collectRun,
  normalizeProvider,
  type ProviderAdapter,
  type RunResult,
} from "agent-sdk-wrapper";

// Compiled against the extracted tarball's public declarations, then run offline.
for (const selected of ["anthropic", "codex"] as const) {
  const provider = normalizeProvider(selected);
  const sessions = new Map<string, number>();
  const adapter: ProviderAdapter = {
    name: provider,
    validateRequest(req) {
      assert.equal(req.provider, provider);
    },
    async ensureAvailable() {},
    async *stream(req, context) {
      const id = req.sessionId ?? `${provider}-session`;
      const turn = (sessions.get(id) ?? 0) + 1;
      sessions.set(id, turn);
      context.onNativeEvent({ turn });
      yield { type: "session_info", id };
      yield { type: "text", text: turn === 1 ? "READY" : "NATIVE_TWIN_42" };
    },
  };
  const defaults = { provider: selected, continueSession: true };
  const agent = new Agent(defaults, { [provider]: adapter });
  const seen: string[] = [];
  const first: RunResult = await collectRun(
    agent.stream("Remember NATIVE_TWIN_42"),
    (env) => {
      seen.push(env.event.type);
    },
  );
  assert.equal(first.status, "success");
  assert.equal(seen[0], "run_started");
  assert.equal(seen.at(-1), "run_finished");
  assert.ok(first.session_id);
  const saved: { provider: typeof provider; sessionId: string } = JSON.parse(
    JSON.stringify({ provider, sessionId: first.session_id }),
  );
  const second = await agent.run("Repeat the token");
  assert.equal(second.status, "success");
  assert.equal(second.final_text, "NATIVE_TWIN_42");
  const resumed = new Agent({ ...defaults, ...saved }, { [provider]: adapter });
  assert.equal(resumed.sessionId, saved.sessionId);
  const third = await resumed.run("Repeat the token again");
  assert.equal(third.status, "success");
  assert.equal(third.session_id, saved.sessionId);
  assert.equal(third.final_text, "NATIVE_TWIN_42");
  assert.equal(sessions.get(saved.sessionId), 3);
}
console.log(
  "Packed public API: Claude and Codex stream/collect/continue/resume passed offline.",
);
