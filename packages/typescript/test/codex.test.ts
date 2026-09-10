import assert from "node:assert/strict";
import { test } from "node:test";
import type {
  CodexOptions,
  ThreadEvent,
  ThreadOptions,
  TurnOptions,
} from "@openai/codex-sdk";
import {
  Agent,
  ConfigError,
  ProcessTerminatedError,
  RuntimeUnavailableError,
} from "../src/index.js";
import type { AgentDefaults } from "../src/index.js";
import { CodexAdapter } from "../src/providers/codex.js";

const completed: ThreadEvent = {
  type: "turn.completed",
  usage: {
    input_tokens: 20,
    cached_input_tokens: 5,
    cache_write_input_tokens: 2,
    output_tokens: 7,
    reasoning_output_tokens: 3,
  },
};
function harness(
  messages: ThreadEvent[],
  defaults: AgentDefaults = {},
  failure?: Error,
) {
  const clients: CodexOptions[] = [];
  const threads: { id?: string; options: ThreadOptions }[] = [];
  const turns: TurnOptions[] = [];
  let closed = 0;
  const provider = new CodexAdapter((clientOptions) => {
    clients.push(clientOptions);
    const thread = {
      async runStreamed(_prompt: string, options: TurnOptions) {
        turns.push(options);
        return {
          events: (async function* () {
            try {
              yield* messages;
              if (failure) throw failure;
            } finally {
              closed++;
            }
          })(),
        };
      },
    };
    return {
      startThread(options) {
        threads.push({ options });
        return thread;
      },
      resumeThread(id, options) {
        threads.push({ id, options });
        return thread;
      },
    };
  });
  return {
    agent: new Agent({ provider: "codex", ...defaults }, { openai: provider }),
    clients,
    threads,
    turns,
    closed: () => closed,
  };
}
test("Codex maps final items once, tools and inclusive token totals", async () => {
  const item = {
    type: "command_execution",
    id: "cmd",
    command: "pwd",
    aggregated_output: "",
    status: "in_progress",
  } as const;
  const text = { type: "agent_message", id: "text", text: "answer" } as const;
  const messages: ThreadEvent[] = [
    { type: "thread.started", thread_id: "thread-1" },
    { type: "turn.started" },
    { type: "item.started", item },
    { type: "item.updated", item: { ...item, aggregated_output: "partial" } },
    {
      type: "item.completed",
      item: {
        ...item,
        aggregated_output: "/tmp",
        status: "completed",
        exit_code: 0,
      },
    },
    {
      type: "item.completed",
      item: { type: "reasoning", id: "think", text: "" },
    },
    { type: "item.updated", item: { ...text, text: "ans" } },
    { type: "item.completed", item: text },
    { type: "item.completed", item: text },
    completed,
  ];
  const raw: unknown[] = [];
  const { agent, clients, threads } = harness(messages, {
    model: "codex:gpt-test",
    cwd: "/tmp",
    effort: "high",
    includeRaw: true,
    onProviderEvent: (event) => {
      raw.push(event);
    },
  });
  const run = await agent.run("prompt");
  assert.equal(run.final_text, "answer");
  assert.equal(run.session_id, "thread-1");
  assert.equal(
    run.events.filter((env) => env.event.type === "tool_call").length,
    1,
  );
  assert.equal(
    run.events.filter((env) => env.event.type === "tool_result").length,
    1,
  );
  assert.equal(
    run.events.filter((env) => env.event.type === "thinking").length,
    1,
  );
  assert.deepEqual(run.usage, {
    input_tokens: 20,
    output_tokens: 7,
    total_tokens: 27,
    cache_read_tokens: 5,
    cache_write_tokens: 2,
    reasoning_output_tokens: 3,
    requests: 0,
  });
  assert.equal(run.cost_usd, null);
  assert.equal(raw.length, messages.length);
  assert.equal(threads[0]?.options.model, "gpt-test");
  assert.equal(threads[0]?.options.modelReasoningEffort, "high");
  assert.equal(threads[0]?.options.workingDirectory, "/tmp");
  assert.deepEqual(clients[0]?.config, { model_reasoning_summary: "auto" });
});
test("Codex completion-only file/MCP/search items get paired calls and results", async () => {
  const { agent } = harness([
    {
      type: "item.completed",
      item: {
        type: "file_change",
        id: "file",
        changes: [{ path: "file", kind: "update" }],
        status: "failed",
      },
    },
    {
      type: "item.completed",
      item: {
        type: "mcp_tool_call",
        id: "mcp",
        server: "external",
        tool: "read",
        arguments: { path: "f" },
        status: "failed",
        error: { message: "denied" },
      },
    },
    {
      type: "item.completed",
      item: { type: "web_search", id: "web", query: "query" },
    },
    completed,
  ]);
  const run = await agent.run("tools");
  const calls = run.events.filter((env) => env.event.type === "tool_call");
  const results = run.events.filter((env) => env.event.type === "tool_result");
  assert.equal(calls.length, 3);
  assert.deepEqual(
    results.map(
      (env) => env.event.type === "tool_result" && env.event.is_error,
    ),
    [true, true, false],
  );
  assert.equal(run.status, "success"); // Tool failures are agent-visible; the turn may still succeed.
});
test("Codex subtracts cumulative thread usage on resume", async () => {
  const messages: ThreadEvent[] = [
    { type: "thread.started", thread_id: "saved" },
    completed,
  ];
  const { agent, threads } = harness(messages, { continueSession: true });
  const first = await agent.run("remember");
  messages[1] = {
    type: "turn.completed",
    usage: {
      input_tokens: 40,
      cached_input_tokens: 10,
      cache_write_input_tokens: 4,
      output_tokens: 14,
      reasoning_output_tokens: 6,
    },
  };
  const second = await agent.run("recall");
  assert.equal(threads[0]?.id, undefined);
  assert.equal(threads[1]?.id, "saved");
  assert.deepEqual(first.usage, second.usage);
  assert.equal(second.session_id, "saved");
});
for (const type of ["turn.failed", "error"] as const)
  test(`Codex ${type} is a terminal typed failure`, async () => {
    const message = "401 Unauthorized";
    const event: ThreadEvent =
      type === "turn.failed" ? { type, error: { message } } : { type, message };
    const { agent } = harness([event], {}, new Error("secondary exit failure"));
    const run = await agent.run("bad credentials");
    assert.equal(run.status, "failure");
    assert.equal(run.error, message);
    assert.equal(
      run.events.filter((env) => env.event.type === "error").length,
      1,
    );
    assert.ok(
      run.events.some(
        (env) =>
          env.event.type === "error" &&
          env.event.error_type === "authentication_failed" &&
          !env.event.retryable,
      ),
    );
  });
test("Codex transient error events are classified and item errors remain warnings", async () => {
  const failure = await harness([
    { type: "turn.failed", error: { message: "429 rate limit" } },
  ]).agent.run("fail");
  assert.ok(
    failure.events.some(
      (env) =>
        env.event.type === "error" &&
        env.event.error_type === "transient_api_error" &&
        env.event.retryable,
    ),
  );
  const success = await harness([
    {
      type: "item.completed",
      item: { id: "warning", type: "error", message: "tool recovered" },
    },
    completed,
  ]).agent.run("recover");
  assert.equal(success.status, "success");
  assert.ok(success.events.some((env) => env.event.type === "warning"));
});
test("Codex truncated streams fail; signal exits throw without retrying", async () => {
  const truncated = await harness([{ type: "turn.started" }]).agent.run(
    "truncated",
  );
  assert.equal(truncated.status, "failure");
  assert.match(truncated.error ?? "", /without turn.completed/);
  const killed = harness(
    [],
    { maxRetries: 5 },
    new Error("Codex Exec exited with signal SIGTERM:"),
  );
  await assert.rejects(killed.agent.run("killed"), ProcessTerminatedError);
  assert.equal(killed.clients.length, 1);
});
test("closing a Codex stream aborts its native signal and closes its iterator", async () => {
  const { agent, turns, closed } = harness([
    {
      type: "item.completed",
      item: { type: "agent_message", id: "text", text: "partial" },
    },
    completed,
  ]);
  for await (const env of agent.stream("close"))
    if (env.event.type === "text") break;
  assert.equal(closed(), 1);
  assert.equal(turns[0]?.signal?.aborted, true);
});
test("Codex rejects native config that could bypass wrapper guarantees", () => {
  for (const native of [
    { client: { config: { mcp_servers: {} } } },
    { thread: { model: "override" } },
    { thread: { outputSchema: {} } },
    { thread: { sandboxMode: "bad" } },
    { thread: { skipGitRepoCheck: "yes" } },
    { client: { env: { X: 1 } } },
  ]) {
    assert.throws(
      () =>
        harness([], {
          providerOptions: { provider: "openai", ...native },
        } as AgentDefaults),
      ConfigError,
    );
  }
});
test("Codex runtime check validates the native executable override without API calls", async () => {
  const agent = new Agent({
    provider: "codex",
    providerOptions: {
      provider: "openai",
      client: { codexPathOverride: "/definitely/missing/codex" },
    },
  });
  await assert.rejects(agent.checkRuntime(), RuntimeUnavailableError);
});

test("Codex external resume warns about history and preserves invisible reasoning", async () => {
  const { agent } = harness([completed], { sessionId: "external" });
  const run = await agent.run("resume");
  assert.ok(
    run.events.some(
      (env) =>
        env.event.type === "warning" &&
        env.event.message.includes("prior history"),
    ),
  );
  assert.ok(
    run.events.some(
      (env) => env.event.type === "thinking" && env.event.text === "",
    ),
  );
  assert.equal(run.usage?.output_tokens, 7);
});

test("Codex counter resets warn and start a new accounting baseline", async () => {
  const messages: ThreadEvent[] = [
    { type: "thread.started", thread_id: "saved" },
    completed,
  ];
  const { agent } = harness(messages, { continueSession: true });
  await agent.run("first");
  messages[1] = {
    type: "turn.completed",
    usage: {
      input_tokens: 2,
      cached_input_tokens: 0,
      cache_write_input_tokens: 0,
      output_tokens: 1,
      reasoning_output_tokens: 0,
    },
  };
  const reset = await agent.run("reset");
  assert.equal(reset.usage?.total_tokens, 3);
  assert.ok(
    reset.events.some(
      (env) =>
        env.event.type === "warning" && env.event.message.includes("reset"),
    ),
  );
});

test("explicit Codex env does not inherit the host API key", async () => {
  const { agent, clients } = harness([completed], {
    providerOptions: { provider: "openai", client: { env: {} } },
  });
  await agent.run("isolated");
  assert.equal(clients[0]?.apiKey, undefined);
  assert.deepEqual(clients[0]?.env, {});
});
