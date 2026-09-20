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
test("message classification uses the canonical vocabulary", () => {
  for (const [message, expected] of [
    [
      "API Error: Connection refused — a firewall or proxy may be blocking it (ConnectionRefused)",
      "transient_api_error",
    ],
    ["connection reset by peer", "transient_api_error"],
    ["Request timed out.", "transient_api_error"],
    [
      "stream disconnected before completion: stream closed before response.completed",
      "transient_api_error",
    ],
    [
      "Selected model is at capacity. Please try a different model.",
      "transient_api_error",
    ],
    ["server busy, try later", "transient_api_error"],
    ['{"type":"rate_limit_error"}', "transient_api_error"],
    ["exceeded retry limit, last status: 529", "transient_api_error"],
    ["unexpected status 503 Service Unavailable", "transient_api_error"],
    ["unexpected status 401 Unauthorized: bad key", "authentication_failed"],
    ["Not logged in · Please run /login", "authentication_failed"],
    ["unexpected status 403 Forbidden: denied", "permission_denied"],
    [
      "unexpected status 404 Not Found: The model 'gpt-nope' does not exist or you do not have access to it.",
      "model_not_found",
    ],
    ["unexpected status 400 Bad Request: malformed", "invalid_request"],
    ["unexpected status 418 I'm a teapot", "api_error_418"],
    [
      "Codex ran out of room in the model's context window. Start a new thread or clear earlier history before retrying.",
      "context_window_exceeded",
    ],
    [
      '{"error":{"message":"Your input exceeds the context window of this model.","type":"invalid_request_error","code":"context_length_exceeded"}}',
      "context_window_exceeded",
    ],
    [
      "Quota exceeded. Check your plan and billing details.",
      "usage_limit_exceeded",
    ],
    [
      "You've hit your usage limit. Upgrade to Plus to continue.",
      "usage_limit_exceeded",
    ],
    ["Your credit balance is too low", "billing_error"],
    // Refusals need a structured signal and bare numbers are not statuses.
    ["The model refused the request", "fallback"],
    ["Processed 503 files before failing", "fallback"],
  ]) {
    const error = classify(message ?? "", "fallback");
    assert.equal(error.error_type, expected, message);
    assert.equal(error.retryable, expected === "transient_api_error", message);
  }
  assert.equal(
    classify("overloaded", "fallback", 529).error_type,
    "transient_api_error",
  );
  assert.equal(classify("gone", "fallback", 410).error_type, "api_error_410");
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
for (const progress of ["normalized", "terminal"] as const)
  test(`does not retry after ${progress} progress`, async () => {
    let calls = 0;
    const agent = new Agent(
      { provider: "openai", maxRetries: 5, retryDelayMs: 0 },
      {
        openai: fake(async function* () {
          calls++;
          if (progress === "normalized")
            yield { type: "text", text: "partial" };
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
test("retryable error events retry after non-progress frames of a new session", async () => {
  const seen: (string | undefined)[] = [];
  const agent = new Agent(
    { provider: "openai", maxRetries: 1, retryDelayMs: 0 },
    {
      openai: fake(async function* (req, context) {
        seen.push(req.sessionId);
        context.onNativeEvent({ type: "thread.started" });
        if (seen.length === 1) {
          yield { type: "session_info", id: "failed-attempt" };
          yield { type: "warning", message: "reconnecting" };
          yield {
            type: "usage",
            usage: { ...emptyUsage(), input_tokens: 5, total_tokens: 5 },
          };
          yield {
            type: "error",
            message: "overloaded",
            error_type: "transient_api_error",
            retryable: true,
          };
          return;
        }
        yield { type: "text", text: "ok" };
      }),
    },
  );
  const result = await agent.run("retry");
  assert.deepEqual(seen, [undefined, undefined]);
  assert.equal(result.status, "success");
  assert.equal(result.usage?.input_tokens, 5);
  assert.deepEqual(
    result.events.map((env) => env.event.type),
    [
      "run_started",
      "session_info",
      "warning",
      "usage",
      "warning",
      "text",
      "run_finished",
    ],
  );
  assert.match(
    result.events[4]?.event.type === "warning"
      ? result.events[4].event.message
      : "",
    /retry 1\/1 .*overloaded/,
  );
});
test("retryable error events are emitted when retries are exhausted or progress occurred", async () => {
  for (const progress of [false, true]) {
    let calls = 0;
    const agent = new Agent(
      { provider: "openai", maxRetries: progress ? 3 : 1, retryDelayMs: 0 },
      {
        openai: fake(async function* () {
          calls++;
          if (progress) yield { type: "thinking", text: "plan" };
          yield {
            type: "error",
            message: "overloaded",
            error_type: "transient_api_error",
            retryable: true,
          };
        }),
      },
    );
    const result = await agent.run("retry");
    assert.equal(calls, progress ? 1 : 2);
    assert.equal(result.status, "failure");
    assert.deepEqual(result.events.at(-2)?.event, {
      type: "error",
      message: "overloaded",
      error_type: "transient_api_error",
      retryable: true,
    });
    assert.equal(
      result.events.filter((env) => env.event.type === "error").length,
      1,
    );
  }
});
test("signal-killed runtimes record the failure, then throw without retrying", async () => {
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
    retryable: false,
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
test("a later error shows a held retryable error and prevents a retry", async () => {
  let calls = 0;
  const agent = new Agent(
    { provider: "openai", maxRetries: 2, retryDelayMs: 0 },
    {
      openai: fake(async function* () {
        calls++;
        yield {
          type: "error",
          message: "busy",
          error_type: "transient_api_error",
          retryable: true,
        };
        yield {
          type: "error",
          message: "denied",
          error_type: "permission_denied",
          retryable: false,
        };
      }),
    },
  );
  const run = await agent.run("held");
  assert.equal(calls, 1);
  assert.deepEqual(
    run.events
      .map((env) => env.event)
      .filter((event): event is ErrorEvent => event.type === "error")
      .map((event) => event.error_type),
    ["transient_api_error", "permission_denied"],
  );
});
test("a resumed session is not retried once it has started", async () => {
  let calls = 0;
  const agent = new Agent(
    { provider: "openai", maxRetries: 2, retryDelayMs: 0, sessionId: "orig" },
    {
      openai: fake(async function* () {
        calls++;
        yield { type: "session_info", id: "orig" };
        yield {
          type: "error",
          message: "overloaded",
          error_type: "transient_api_error",
          retryable: true,
        };
      }),
    },
  );
  const run = await agent.run("resume");
  assert.equal(calls, 1);
  assert.equal(run.error, "overloaded");
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
test("a kill after a held retryable error keeps the held error", async () => {
  const agent = new Agent(
    { provider: "openai", maxRetries: 2, retryDelayMs: 0 },
    {
      openai: fake(async function* () {
        yield {
          type: "error",
          message: "overloaded",
          error_type: "transient_api_error",
          retryable: true,
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
test("an abort while a retryable error is held is a cancelled run", async () => {
  const controller = new AbortController();
  const agent = new Agent(
    {
      provider: "openai",
      maxRetries: 1,
      retryDelayMs: 0,
      signal: controller.signal,
    },
    {
      openai: fake(async function* () {
        yield {
          type: "error",
          message: "overloaded",
          error_type: "transient_api_error",
          retryable: true,
        };
        controller.abort();
      }),
    },
  );
  const run = await agent.run("abort");
  assert.equal(run.status, "cancelled");
  assert.ok(run.events.some((env) => env.event.type === "warning"));
});
