import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type TestContext, test } from "node:test";
import { setTimeout as delay } from "node:timers/promises";
import type { ThreadEvent } from "@openai/codex-sdk";
import { Agent, type AgentDefaults } from "../src/index.js";

// Runs the real Codex SDK exec path against a fake runtime that prints JSONL.
type After = "exit" | "exit1" | "hang" | "close-stdout" | "sigkill";
const script = `#!${process.execPath}
const fs = require("node:fs");
const plan = JSON.parse(fs.readFileSync(process.env.FAKE_CODEX_PLAN, "utf8"));
fs.writeFileSync(plan.pidFile, String(process.pid));
fs.readFileSync(0);
for (const event of plan.events) fs.writeSync(1, JSON.stringify(event) + "\\n");
if (plan.after === "exit") process.exit(0);
if (plan.after === "exit1") process.exit(1);
if (plan.after === "sigkill") process.kill(process.pid, "SIGKILL");
if (plan.after === "close-stdout") fs.closeSync(1);
setInterval(() => {}, 1000);
`;
async function fakeCodex(
  t: TestContext,
  events: ThreadEvent[],
  after: After,
  defaults: AgentDefaults = {},
) {
  const root = await mkdtemp(join(tmpdir(), "agent-sdk-wrapper-codex-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const executable = join(root, "codex.cjs");
  const plan = join(root, "plan.json");
  const pidFile = join(root, "pid");
  await writeFile(executable, script, { mode: 0o755 });
  await writeFile(plan, JSON.stringify({ events, after, pidFile }));
  const agent = new Agent({
    provider: "codex",
    providerOptions: {
      provider: "openai",
      client: {
        codexPathOverride: executable,
        env: { FAKE_CODEX_PLAN: plan },
      },
    },
    ...defaults,
  });
  return {
    agent,
    // Resolves once the fake runtime is gone, failing if it outlives the run.
    async exited(): Promise<void> {
      const pid = Number(await readFile(pidFile, "utf8"));
      for (let waited = 0; waited < 5000; waited += 20) {
        try {
          process.kill(pid, 0);
        } catch {
          return;
        }
        await delay(20);
      }
      assert.fail(`fake Codex runtime ${pid} is still running`);
    },
  };
}
const started: ThreadEvent[] = [
  { type: "thread.started", thread_id: "thread-1" },
  { type: "turn.started" },
];
const answer: ThreadEvent = {
  type: "item.completed",
  item: { type: "agent_message", id: "text", text: "answer" },
};

test("closing a real Codex stream early exits cleanly and kills the runtime", async (t) => {
  const { agent, exited } = await fakeCodex(t, [...started, answer], "hang");
  // Aborting the SDK's signal after its cleanup crashes with an uncaught AbortError.
  for await (const env of agent.stream("close"))
    if (env.event.type === "text") break;
  await exited();
});

test("a throwing provider-event callback fails the run and kills the runtime", async (t) => {
  const { agent, exited } = await fakeCodex(t, [...started, answer], "hang", {
    onProviderEvent: (event) => {
      if ((event as ThreadEvent).type === "turn.started")
        throw new Error("callback boom");
    },
  });
  const run = await agent.run("callback");
  assert.equal(run.status, "failure");
  assert.equal(run.error, "callback boom");
  await exited();
});
