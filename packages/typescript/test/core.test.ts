import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
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
import { emptyUsage, type ErrorEvent } from "../src/events.js";
import { nativeError } from "../src/providers/common.js";
import { classify } from "../src/providers/common.js";

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
test("model IDs with colons stay whole unless the prefix is a provider", () => {
  for (const [provider, model, expected] of [
    [
      "anthropic",
      "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
      "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    ],
    [
      "anthropic",
      "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc",
      "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc",
    ],
    [
      undefined,
      "anthropic:us.anthropic.claude-sonnet-4-5-20250929-v1:0",
      "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    ],
    ["codex", "ft:gpt-4o:org:custom:id", "ft:gpt-4o:org:custom:id"],
    ["openai", "qwen2.5-coder:7b", "qwen2.5-coder:7b"],
  ] as const)
    assert.equal(resolveProvider(provider, model).model, expected);
  assert.throws(
    () => resolveProvider(undefined, "anthropic.claude-3-5-sonnet-v2:0"),
    ConfigError,
  );
});
test("a missing or non-directory cwd fails request resolution", async () => {
  let checked = 0;
  const provider = successful();
  provider.ensureAvailable = async () => {
    checked++;
  };
  const file = fileURLToPath(import.meta.url);
  for (const cwd of ["/definitely/missing/dir", file]) {
    assert.throws(
      () => new Agent({ provider: "openai", cwd }, { openai: provider }),
      ConfigError,
    );
    const agent = new Agent({ provider: "openai" }, { openai: provider });
    await assert.rejects(agent.checkRuntime({ cwd }), ConfigError);
    await assert.rejects(agent.run({ prompt: "x", cwd }), ConfigError);
  }
  assert.equal(checked, 0);
});
test("message classification matches the shared cases", () => {
  const { cases } = JSON.parse(
    readFileSync(
      new URL(
        "../../../../docs/fixtures/error-classification-v1.json",
        import.meta.url,
      ),
      "utf8",
    ),
  ) as {
    cases: { message: string; status?: number; error_type: string | null }[];
  };
  for (const { message, status, error_type } of cases)
    assert.equal(
      classify(message, "fallback", status).error_type,
      error_type ?? "fallback",
      message,
    );
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
        yield { type: "text", text: "draft" };
        yield { type: "text", text: "one two" };
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
test("a transient failure ends the run with its type", async () => {
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
  const result = await agent.run("no retry");
  assert.equal(calls, 1);
  assert.equal(result.status, "failure");
  assert.deepEqual(
    [result.error, result.error_type],
    ["unavailable", "transient_api_error"],
  );
});
test("a provider error wins over a later exception", async () => {
  const agent = new Agent(
    { provider: "openai" },
    {
      openai: fake(async function* () {
        yield { type: "text", text: "partial" };
        yield {
          type: "error",
          message: "real failure",
          error_type: "max_turns",
        };
        throw new TransientError("cleanup failure");
      }),
    },
  );
  const result = await agent.run("limit");
  assert.equal(result.final_text, "partial");
  assert.deepEqual(
    [result.error, result.error_type, result.ended_reason],
    ["real failure", "max_turns", "max_turns"],
  );
  assert.equal(
    result.events.filter((env) => env.event.type === "error").length,
    1,
  );
});
test("signal-killed runtimes record the failure, then throw", async () => {
  let calls = 0;
  const agent = new Agent(
    { provider: "openai" },
    {
      // biome-ignore lint/correctness/useYield: model a runtime killed before its first frame
      openai: fake(async function* () {
        calls++;
        throw new ProcessTerminatedError("SIGTERM");
      }),
    },
  );
  const seen: EventEnvelope[] = [];
  await assert.rejects(
    collectRun(agent.stream("stop"), (env) => {
      seen.push(env);
    }),
    ProcessTerminatedError,
  );
  assert.equal(calls, 1);
  assert.deepEqual(
    seen.map((env) => env.event.type),
    ["run_started", "error", "run_finished"],
  );
  assert.deepEqual(seen[1]?.event, {
    type: "error",
    message: "SIGTERM",
    error_type: "process_terminated",
  });
  const finished = seen[2]?.event;
  assert.equal(finished?.type === "run_finished" && finished.status, "failure");
});
test("a runtime killed after the caller's abort is cancelled", async () => {
  const controller = new AbortController();
  const agent = new Agent(
    { provider: "openai", signal: controller.signal },
    {
      // biome-ignore lint/correctness/useYield: model a kill racing the caller's abort
      openai: fake(async function* () {
        controller.abort();
        throw new ProcessTerminatedError(
          "Codex Exec exited with signal SIGTERM",
        );
      }),
    },
  );
  const result = await agent.run("abort");
  assert.equal(result.status, "cancelled");
});
test("a native failure after the caller's abort is cancelled once", async () => {
  const controller = new AbortController();
  const agent = new Agent(
    { provider: "openai", signal: controller.signal },
    {
      openai: fake(async function* () {
        yield { type: "session_info", id: "cancelled-session" };
        controller.abort();
        throw new ProviderProtocolError("stream ended without a result");
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
test("an abort after the terminal frame leaves a completed run successful", async () => {
  const controller = new AbortController();
  const agent = new Agent(
    { provider: "openai", signal: controller.signal },
    {
      openai: fake(async function* () {
        yield { type: "text", text: "complete answer" };
        yield { type: "usage", usage: emptyUsage() };
      }),
    },
  );
  const result = await collectRun(agent.stream("late"), (env) => {
    if (env.event.type === "usage") controller.abort();
  });
  assert.equal(result.status, "success");
  assert.equal(result.final_text, "complete answer");
  assert.equal(result.error, null);
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
      { provider: expected.provider, model: started.model },
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
test("only real signal names mark a runtime as terminated", () => {
  for (const message of [
    "Codex Exec exited with code 1: 401 Unauthorized. Please log out and sign in again.",
    "Claude Code process exited: The request signature we calculated does not match",
  ])
    assert.ok(
      !(nativeError(new Error(message)) instanceof ProcessTerminatedError),
    );
  assert.ok(
    nativeError(
      new Error("Claude Code process terminated by signal SIGKILL"),
    ) instanceof ProcessTerminatedError,
  );
});
test("the first error sets the result error; later errors are kept", async () => {
  const agent = new Agent(
    { provider: "openai" },
    {
      openai: fake(async function* () {
        yield {
          type: "error",
          message: "busy",
          error_type: "transient_api_error",
        };
        yield {
          type: "error",
          message: "denied",
          error_type: "permission_denied",
        };
      }),
    },
  );
  const run = await agent.run("errors");
  assert.deepEqual(
    [run.error, run.error_type],
    ["busy", "transient_api_error"],
  );
  assert.deepEqual(
    run.events
      .map((env) => env.event)
      .filter((event): event is ErrorEvent => event.type === "error")
      .map((event) => event.error_type),
    ["transient_api_error", "permission_denied"],
  );
});
test("any upper-case signal name marks a runtime as terminated", () => {
  assert.ok(
    nativeError(
      new Error("Codex Exec exited with signal SIGXFSZ: stream disconnected"),
    ) instanceof ProcessTerminatedError,
  );
});
test("high demand, 408 and 409 are transient; exit codes 130/137/143 are kills", () => {
  for (const [message, status] of [
    [
      "We're currently experiencing high demand, which may cause temporary errors.",
      undefined,
    ],
    ["request failed", 408],
    ["request failed", 409],
  ] as const)
    assert.equal(
      classify(message, "provider_exception", status).error_type,
      "transient_api_error",
    );
  assert.ok(
    nativeError(new Error("Codex Exec exited with code 137")) instanceof
      ProcessTerminatedError,
  );
});
test("a kill after a provider error records both", async () => {
  const agent = new Agent(
    { provider: "openai" },
    {
      openai: fake(async function* () {
        yield {
          type: "error",
          message: "overloaded",
          error_type: "transient_api_error",
        };
        throw new ProcessTerminatedError("killed by SIGKILL");
      }),
    },
  );
  const errors: string[] = [];
  await assert.rejects(
    collectRun(agent.stream("kill"), (env) => {
      if (env.event.type === "error") errors.push(env.event.error_type);
    }),
    ProcessTerminatedError,
  );
  assert.deepEqual(errors, ["transient_api_error", "process_terminated"]);
});
test("a mid-stream ConfigError is recorded and finishes the run before it throws", async () => {
  const agent = new Agent(
    { provider: "openai" },
    {
      openai: fake(async function* () {
        yield { type: "session_info", id: "s" };
        throw new ConfigError("late invalid setting");
      }),
    },
  );
  const types: string[] = [];
  await assert.rejects(
    collectRun(agent.stream("late"), (env) => {
      types.push(env.event.type);
    }),
    ConfigError,
  );
  assert.deepEqual(types.slice(-2), ["error", "run_finished"]);
});
