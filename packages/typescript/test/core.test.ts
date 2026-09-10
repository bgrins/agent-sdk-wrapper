import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import {
  Agent,
  ConfigError,
  ProcessTerminatedError,
  ProviderProtocolError,
  RuntimeUnavailableError,
  TransientError,
  collectRun,
  resolveProvider,
} from "../src/index.js";
import type {
  AgentDefaults,
  EventEnvelope,
  ProviderAdapter,
  ProviderContext,
  ProviderEvent,
  ResolvedRequest,
  RunResult,
} from "../src/index.js";
import { emptyUsage } from "../src/events.js";

function fake(
  events: (
    req: ResolvedRequest,
    context: ProviderContext,
  ) => AsyncIterable<ProviderEvent>,
): ProviderAdapter {
  return {
    name: "openai",
    validateRequest() {},
    async ensureAvailable() {},
    stream: events,
  };
}
const successful = () =>
  fake(async function* () {
    yield { type: "text", text: "ok" };
  });
test("provider aliases, inference, model prefixes and conflicting selections", () => {
  for (const [provider, model, expected] of [
    ["codex", "gpt-test", "openai"],
    [undefined, "codex:gpt-test", "openai"],
    [undefined, "claude-test", "anthropic"],
    [" ANTHROPIC ", "custom-model", "anthropic"],
  ])
    assert.equal(resolveProvider(provider, model).provider, expected);
  assert.deepEqual(resolveProvider(undefined, "codex:gpt-test"), {
    provider: "openai",
    model: "gpt-test",
  });
  for (const [provider, model] of [
    [undefined, undefined],
    ["bad", undefined],
    [undefined, "unknown"],
    ["anthropic", "codex:gpt-test"],
    ["openai", "claude-test"],
    ["openai", "codex:"],
  ])
    assert.throws(() => resolveProvider(provider, model), ConfigError);
});
test("unknown, reserved, malformed and cross-provider options fail before availability", async () => {
  let checked = 0;
  const provider = successful();
  provider.ensureAvailable = async () => {
    checked++;
    throw new RuntimeUnavailableError("missing");
  };
  for (const options of [
    { tools: [] },
    { mcpServers: [] },
    { outputSchema: {} },
    { systemPrompt: "system" },
    { maxTurns: 1 },
    { artifactsDir: "out" },
    { subagents: {} },
    { builtinTools: [] },
    { maxRetries: -1 },
    { maxRetries: Number.NaN },
    { retryDelayMs: 0.5 },
    { includeRaw: "yes" },
    { continueSession: 1 },
    { effort: "none" },
    { cwd: 1 },
    { sessionId: "" },
    { providerOptions: { provider: "anthropic" } },
    { unexpected: true },
  ])
    assert.throws(
      () =>
        new Agent({ provider: "openai", ...options } as AgentDefaults, {
          openai: provider,
        }),
      ConfigError,
    );
  const agent = new Agent({ provider: "openai" }, { openai: provider });
  await assert.rejects(
    agent.checkRuntime({ effort: "invalid" } as unknown as AgentDefaults),
    ConfigError,
  );
  assert.equal(checked, 0);
  await assert.rejects(agent.checkRuntime(), RuntimeUnavailableError);
});
test("constructor defaults, envelopes, streaming collection and usage aggregation", async () => {
  const seen: ResolvedRequest[] = [];
  const agent = new Agent(
    { model: "codex:gpt-default", cwd: "/tmp", effort: "low" },
    {
      openai: fake(async function* (req) {
        seen.push(req);
        yield { type: "session_info", id: "session-1" };
        yield { type: "text", text: "one " };
        yield { type: "text", text: "two" };
        yield {
          type: "usage",
          usage: {
            ...emptyUsage(),
            input_tokens: 4,
            output_tokens: 2,
            total_tokens: 6,
          },
          cost_usd: 0.1,
        };
        yield {
          type: "usage",
          usage: {
            ...emptyUsage(),
            input_tokens: 3,
            output_tokens: 1,
            total_tokens: 4,
          },
          cost_usd: 0.2,
        };
      }),
    },
  );
  const rendered: EventEnvelope[] = [];
  const result = await collectRun(
    agent.stream({ prompt: "question", model: "gpt-override" }),
    (env) => {
      rendered.push(env);
    },
  );
  assert.equal(seen[0]?.cwd, "/tmp");
  assert.equal(seen[0]?.effort, "low");
  assert.equal(seen[0]?.model, "gpt-override");
  assert.equal(result.final_text, "one two");
  assert.equal(result.usage?.total_tokens, 10);
  assert.ok(Math.abs((result.cost_usd ?? 0) - 0.3) < 1e-10);
  assert.equal(result.session_id, "session-1");
  assert.equal(result.status, "success");
  assert.deepEqual(result.events, rendered);
  result.events.forEach((env, index) => {
    assert.equal(env.sequence, index);
    assert.equal(env.run_id, result.run_id);
    assert.ok(Number.isFinite(Date.parse(env.timestamp)));
  });
  assert.deepEqual(result.events[0]?.event, {
    type: "run_started",
    provider: "openai",
    model: "gpt-override",
    prompt: "question",
    cwd: "/tmp",
  });
  assert.equal(result.events.at(-1)?.event.type, "run_finished");
});
test("sessions persist after stream/run/failure, explicit resume wins, providers stay separate", async () => {
  const seen: ResolvedRequest[] = [];
  const provider = fake(async function* (req) {
    seen.push(req);
    yield { type: "session_info", id: req.sessionId ?? "saved" };
    yield {
      type: "error",
      message: "terminal",
      error_type: "turn_failed",
      retryable: false,
    };
  });
  const agent = new Agent(
    { provider: "codex", continueSession: true },
    { openai: provider, anthropic: { ...provider, name: "anthropic" } },
  );
  await collectRun(agent.stream("first"));
  assert.equal(agent.sessionId, "saved");
  await agent.run("second");
  await agent.run({ prompt: "third", sessionId: "explicit" });
  await agent.run({ prompt: "fourth", continueSession: false });
  await agent.run({ prompt: "other provider", provider: "anthropic" });
  assert.deepEqual(
    seen.map((req) => req.sessionId),
    [undefined, "saved", "explicit", undefined, undefined],
  );
});
test("constructor session ID is available before the first run", async () => {
  const agent = new Agent(
    { provider: "codex", sessionId: "saved", continueSession: true },
    {
      openai: fake(async function* (req) {
        assert.equal(req.sessionId, "saved");
        yield { type: "session_info", id: "saved" };
      }),
    },
  );
  assert.equal(agent.sessionId, "saved");
  assert.equal((await agent.run("resume")).session_id, "saved");
});
test("retries are opt-in by default", async () => {
  let calls = 0;
  const agent = new Agent(
    { provider: "codex" },
    {
      // biome-ignore lint/correctness/useYield: model a transient startup failure
      openai: fake(async function* () {
        calls++;
        throw new TransientError("unavailable");
      }),
    },
  );
  assert.equal((await agent.run("no automatic retry")).status, "failure");
  assert.equal(calls, 1);
});
test("retries transient startup failures with ordered warnings then succeeds", async () => {
  let calls = 0;
  const agent = new Agent(
    { provider: "openai", maxRetries: 2, retryDelayMs: 0 },
    {
      openai: fake(async function* () {
        if (++calls < 3) throw new TransientError("overloaded");
        yield { type: "text", text: "ok" };
      }),
    },
  );
  const result = await agent.run("retry");
  assert.equal(calls, 3);
  assert.deepEqual(
    result.events.map((env) => env.event.type),
    ["run_started", "warning", "warning", "text", "run_finished"],
  );
});
test("retry exhaustion emits a typed retryable failure", async () => {
  let calls = 0;
  const agent = new Agent(
    { provider: "openai", maxRetries: 1, retryDelayMs: 0 },
    {
      // biome-ignore lint/correctness/useYield: model an async stream failing before its first frame
      openai: fake(async function* () {
        calls++;
        throw new TransientError("overloaded");
      }),
    },
  );
  const result = await agent.run("retry");
  assert.equal(calls, 2);
  assert.equal(result.status, "failure");
  assert.deepEqual(result.events.at(-2)?.event, {
    type: "error",
    message: "overloaded",
    error_type: "transient_api_error",
    retryable: true,
  });
});
for (const progress of ["normalized", "native", "terminal"] as const)
  test(`does not retry after ${progress} progress`, async () => {
    let calls = 0;
    const agent = new Agent(
      { provider: "openai", maxRetries: 5, retryDelayMs: 0 },
      {
        openai: fake(async function* (_req, context) {
          calls++;
          if (progress === "normalized")
            yield { type: "text", text: "partial" };
          if (progress === "native")
            context.onNativeEvent({ type: "turn.started" });
          if (progress === "terminal")
            yield {
              type: "error",
              message: "real failure",
              error_type: "max_turns",
              retryable: false,
            };
          throw new TransientError("cleanup failure");
        }),
      },
    );
    const result = await agent.run("retry");
    assert.equal(calls, 1);
    assert.equal(result.status, "failure");
    if (progress === "terminal") {
      assert.equal(result.error, "real failure");
      assert.equal(result.ended_reason, "max_turns");
    }
  });
test("signal-killed runtimes throw and never retry", async () => {
  let calls = 0;
  const agent = new Agent(
    { provider: "openai", maxRetries: 5 },
    {
      // biome-ignore lint/correctness/useYield: model a runtime killed before its first frame
      openai: fake(async function* () {
        calls++;
        throw new ProcessTerminatedError("SIGTERM");
      }),
    },
  );
  await assert.rejects(agent.run("stop"), ProcessTerminatedError);
  assert.equal(calls, 1);
});
test("aborted runs finish cancelled, including cancellation during backoff", async () => {
  const controller = new AbortController();
  const agent = new Agent(
    { provider: "openai", signal: controller.signal, maxRetries: 2 },
    {
      // biome-ignore lint/correctness/useYield: model an async stream failing before its first frame
      openai: fake(async function* () {
        throw new TransientError("retry");
      }),
    },
  );
  const result = await collectRun(agent.stream("abort"), (envelope) => {
    if (envelope.event.type === "warning") controller.abort();
  });
  assert.equal(result.status, "cancelled");
  assert.equal(result.ended_reason, "cancelled");
});
test("cancellation also works when the native iterator returns without throwing", async () => {
  const controller = new AbortController();
  const agent = new Agent(
    { provider: "openai", signal: controller.signal },
    {
      openai: fake(async function* () {
        yield { type: "session_info", id: "cancelled-session" };
        controller.abort();
      }),
    },
  );
  const result = await agent.run("abort");
  assert.equal(result.status, "cancelled");
  assert.equal(result.ended_reason, "cancelled");
  assert.equal(agent.sessionId, "cancelled-session");
  assert.equal(
    result.events.filter((env) => env.event.type === "error").length,
    1,
  );
});
test("early iterator closure cleans up and releases the Agent concurrency guard", async () => {
  let closed = 0;
  const agent = new Agent(
    { provider: "openai" },
    {
      openai: fake(async function* () {
        try {
          yield { type: "text", text: "partial" };
          yield { type: "text", text: "later" };
        } finally {
          closed++;
        }
      }),
    },
  );
  for await (const env of agent.stream("close")) {
    if (env.event.type === "text") {
      await assert.rejects(agent.run("concurrent"), ConfigError);
      break;
    }
  }
  assert.equal(closed, 1);
  assert.equal((await agent.run("again")).status, "success");
});
test("collectRun rejects truncated and out-of-order streams", async () => {
  const result = await new Agent(
    { provider: "openai" },
    { openai: successful() },
  ).run("ok");
  const first = result.events[0];
  assert.ok(first);
  for (const events of [
    result.events.slice(0, -1),
    result.events.slice(1),
    [...result.events, first],
  ]) {
    await assert.rejects(
      collectRun(
        (async function* () {
          yield* events;
        })(),
      ),
      ProviderProtocolError,
    );
  }
});
test("shared v1 fixtures produce the same result in Python and TypeScript", async () => {
  const fixtures: RunResult[] = JSON.parse(
    readFileSync(
      new URL("../../../../docs/fixtures/native-twin-v1.json", import.meta.url),
      "utf8",
    ),
  );
  for (const expected of fixtures) {
    const started = expected.events[0]?.event;
    assert.ok(started);
    assert.equal(started.type, "run_started");
    if (started.type !== "run_started") throw new Error("fixture");
    const adapter = fake(async function* () {
      for (const env of expected.events.slice(1, -1))
        yield env.event as ProviderEvent;
    });
    const actual = await new Agent(
      { provider: expected.provider, model: expected.model ?? undefined },
      { openai: adapter },
    ).run(started.prompt);
    const stable = ({
      run_id: _id,
      duration_ms: _duration,
      events,
      ...rest
    }: RunResult) => ({
      ...rest,
      events: events.map((env) =>
        env.event.type === "run_finished"
          ? { ...env.event, duration_ms: 0 }
          : env.event,
      ),
    });
    assert.deepEqual(stable(actual), stable(expected));
    assert.deepEqual(
      await collectRun(
        (async function* () {
          yield* expected.events;
        })(),
      ),
      expected,
    );
  }
});
