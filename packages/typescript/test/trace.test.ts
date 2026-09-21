import assert from "node:assert/strict";
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type TestContext, test } from "node:test";
import {
  Agent,
  type AgentDefaults,
  ConfigError,
  type EventEnvelope,
  ProcessTerminatedError,
  type ProviderAdapter,
  RuntimeUnavailableError,
  TraceWriteError,
} from "../src/index.js";

function directory(t: TestContext): string {
  const root = mkdtempSync(join(tmpdir(), "agent-trace-"));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  return root;
}

function readTrace(path: string): EventEnvelope[] {
  return readFileSync(path, "utf8")
    .trim()
    .split("\n")
    .map((line) => JSON.parse(line));
}

function provider(stream: ProviderAdapter["stream"]): ProviderAdapter {
  return {
    name: "openai",
    validateRequest() {},
    async ensureAvailable() {},
    stream,
  };
}

test("trace files are visible during streaming and retain resumed calls in separate paths", async (t) => {
  const root = directory(t);
  const firstPath = join(root, "first", "trace.jsonl");
  const secondPath = join(root, "second", "trace.jsonl");
  const seen: (string | undefined)[] = [];
  const agent = new Agent(
    { provider: "codex", traceFile: firstPath, continueSession: true },
    {
      openai: provider(async function* (req) {
        seen.push(req.sessionId);
        yield { type: "session_info", id: "session-1" };
        yield { type: "text", text: req.prompt };
      }),
    },
  );
  const events: EventEnvelope[] = [];
  for await (const envelope of agent.stream("first")) {
    events.push(envelope);
    assert.deepEqual(readTrace(firstPath), events);
  }
  const second = await agent.run({ prompt: "second", traceFile: secondPath });
  assert.deepEqual(readTrace(firstPath), events);
  assert.deepEqual(readTrace(secondPath), second.events);
  assert.deepEqual(seen, [undefined, "session-1"]);
  const third = await agent.run("replacement");
  assert.deepEqual(readTrace(firstPath), third.events);
  assert.notEqual(third.run_id, events[0]?.run_id);
});

test("trace records normalized failure and cancellation", async (t) => {
  const root = directory(t);
  const agent = new Agent(
    { provider: "openai" },
    {
      openai: provider(async function* () {
        yield {
          type: "error",
          message: "refused",
          error_type: "refused",
        };
      }),
    },
  );
  const failed = await agent.run({
    prompt: "test",
    traceFile: join(root, "failure.jsonl"),
  });
  assert.equal(failed.status, "failure");
  assert.deepEqual(readTrace(join(root, "failure.jsonl")), failed.events);
  assert.deepEqual(
    failed.events.map((env) => env.event.type),
    ["run_started", "error", "run_finished"],
  );
  const abort = new AbortController();
  abort.abort();
  const cancelled = await agent.run({
    prompt: "cancel",
    signal: abort.signal,
    traceFile: join(root, "cancel.jsonl"),
  });
  assert.equal(cancelled.status, "cancelled");
  assert.deepEqual(readTrace(join(root, "cancel.jsonl")), cancelled.events);
});

test("breaking iteration and killed runtimes preserve partial traces and release the Agent", async (t) => {
  const path = join(directory(t), "trace.jsonl");
  let closed = 0;
  const agent = new Agent(
    { provider: "openai", traceFile: path },
    {
      openai: provider(async function* () {
        try {
          yield { type: "text", text: "partial" };
          throw new ProcessTerminatedError("killed");
        } finally {
          closed++;
        }
      }),
    },
  );
  for await (const envelope of agent.stream("break")) {
    if (envelope.event.type === "text") break;
  }
  assert.equal(closed, 1);
  assert.deepEqual(
    readTrace(path).map((env) => env.event.type),
    ["run_started", "text"],
  );
  await assert.rejects(agent.run("killed"), ProcessTerminatedError);
  assert.equal(closed, 2);
  const killed = readTrace(path).map((env) => env.event);
  assert.deepEqual(
    killed.map((event) => event.type),
    ["run_started", "text", "error", "run_finished"],
  );
  assert.deepEqual(killed[2], {
    type: "error",
    message: "killed",
    error_type: "process_terminated",
  });
  assert.equal(
    killed[3]?.type === "run_finished" && killed[3].status,
    "failure",
  );
});

test("trace validation and runtime checks leave files untouched", async (t) => {
  const path = join(directory(t), "nested", "trace.jsonl");
  const adapter = provider(async function* () {
    yield { type: "text", text: "ok" };
  });
  const agent = new Agent(
    { provider: "openai", traceFile: path },
    { openai: adapter },
  );
  for (const traceFile of ["", " ", "bad\0path", 42, null]) {
    await assert.rejects(
      agent.checkRuntime({ traceFile } as AgentDefaults),
      ConfigError,
    );
  }
  await agent.checkRuntime();
  assert.equal(existsSync(path), false);
  adapter.ensureAvailable = async () => {
    throw new RuntimeUnavailableError("missing");
  };
  await assert.rejects(agent.run("unavailable"), RuntimeUnavailableError);
  assert.equal(existsSync(path), false);
});

test("trace I/O and serialization failures propagate", async (t) => {
  const root = directory(t);
  const path = join(root, "trace.jsonl");
  let calls = 0;
  let closed = 0;
  const circular: Record<string, unknown> = {};
  circular.self = circular;
  const agent = new Agent(
    { provider: "openai", traceFile: path },
    {
      openai: provider(async function* () {
        calls++;
        try {
          yield { type: "text", text: "raw", raw: circular };
        } finally {
          closed++;
        }
      }),
    },
  );
  mkdirSync(path);
  await assert.rejects(agent.run("bad path"), TraceWriteError);
  assert.equal(calls, 0);
  rmSync(path, { recursive: true });
  await assert.rejects(agent.run("bad raw"), TraceWriteError);
  assert.equal(calls, 1);
  assert.equal(closed, 1);
  assert.deepEqual(
    readTrace(path).map((env) => env.event.type),
    ["run_started"],
  );
  const untraced = await agent.run({
    prompt: "disable tracing",
    traceFile: undefined,
  });
  assert.equal(untraced.status, "success");
  assert.equal(calls, 2);
});
