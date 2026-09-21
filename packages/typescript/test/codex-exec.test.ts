import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type TestContext, test } from "node:test";
import { setTimeout as delay } from "node:timers/promises";
import type { ThreadEvent } from "@openai/codex-sdk";
import {
  Agent,
  type AgentDefaults,
  collectRun,
  type EventEnvelope,
  ProcessTerminatedError,
} from "../src/index.js";

// Adapters refuse to launch without API credentials; these tests fake the runtime.
process.env.OPENAI_API_KEY ||= "test-key";

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
        apiKey: "test-key",
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
const completed: ThreadEvent = {
  type: "turn.completed",
  usage: {
    input_tokens: 10,
    cached_input_tokens: 0,
    cache_write_input_tokens: 0,
    output_tokens: 2,
    reasoning_output_tokens: 0,
  },
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

test("an abort after turn.completed keeps the completed Codex run", async (t) => {
  const controller = new AbortController();
  const { agent, exited } = await fakeCodex(
    t,
    [...started, answer, completed],
    "hang",
    { signal: controller.signal },
  );
  const run = await collectRun(agent.stream("late abort"), (env) => {
    if (env.event.type === "usage") controller.abort();
  });
  assert.equal(run.status, "success");
  assert.equal(run.final_text, "answer");
  await exited();
});

test("Codex reconnect notices are warnings and the recovered turn succeeds", async (t) => {
  const { agent } = await fakeCodex(
    t,
    [
      ...started,
      {
        type: "error",
        message:
          "Reconnecting... 1/5 (stream disconnected before completion: stream closed before response.completed)",
      },
      answer,
      completed,
    ],
    "exit",
  );
  const run = await agent.run("recover");
  assert.equal(run.status, "success");
  assert.equal(run.final_text, "answer");
  assert.equal(run.error, null);
  const warnings = run.events.flatMap((env) =>
    env.event.type === "warning" ? [env.event.message] : [],
  );
  assert.equal(warnings.length, 1);
  assert.match(warnings[0] ?? "", /^Reconnecting\.\.\. 1\/5/);
});

test("a fatal Codex failure yields one classified error despite the exit code", async (t) => {
  const message =
    "unexpected status 401 Unauthorized: bad key, url: http://127.0.0.1/v1/responses";
  const { agent } = await fakeCodex(
    t,
    [
      ...started,
      { type: "error", message },
      { type: "turn.failed", error: { message } },
    ],
    "exit1",
  );
  const run = await agent.run("fail");
  const errors = run.events.flatMap((env) =>
    env.event.type === "error" ? [env.event] : [],
  );
  assert.deepEqual(errors, [
    {
      type: "error",
      message,
      error_type: "authentication_failed",
    },
  ]);
  assert.equal(run.status, "failure");
});

test("a Codex stream without turn.completed or turn.failed is a protocol error", async (t) => {
  const { agent } = await fakeCodex(
    t,
    [...started, { type: "error", message: "Reconnecting... 5/5" }],
    "exit",
  );
  const run = await agent.run("truncated");
  const errors = run.events.flatMap((env) =>
    env.event.type === "error" ? [env.event] : [],
  );
  assert.equal(errors.length, 1);
  assert.equal(errors[0]?.error_type, "provider_protocol_error");
});

test("a cancel between Codex stdout EOF and exit reports cancelled", async (t) => {
  const controller = new AbortController();
  const { agent, exited } = await fakeCodex(t, started, "close-stdout", {
    signal: controller.signal,
  });
  const run = await collectRun(agent.stream("cancel"), (env) => {
    if (env.event.type === "session_info")
      setTimeout(() => controller.abort(), 200);
  });
  assert.equal(run.status, "cancelled");
  await exited();
});

test("a signal-killed Codex runtime records the failure, then throws", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "agent-sdk-wrapper-trace-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const trace = join(root, "trace.jsonl");
  const { agent } = await fakeCodex(t, started, "sigkill", {
    traceFile: trace,
  });
  await assert.rejects(agent.run("killed"), ProcessTerminatedError);
  const events = (await readFile(trace, "utf8"))
    .trim()
    .split("\n")
    .map((line) => (JSON.parse(line) as EventEnvelope).event);
  assert.deepEqual(
    events.map((event) => event.type),
    ["run_started", "session_info", "error", "run_finished"],
  );
  assert.equal(
    events[2]?.type === "error" && events[2].error_type,
    "process_terminated",
  );
});
