import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
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
  ProviderError,
  RuntimeUnavailableError,
} from "../src/index.js";
import type { AgentDefaults } from "../src/index.js";
import { CodexAdapter } from "../src/providers/codex.js";

// Adapters refuse to launch without API credentials; these tests fake the runtime.
process.env.OPENAI_API_KEY ||= "test-key";

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
  assert.deepEqual(
    run.events
      .filter((env) => env.event.type === "tool_call")
      .map((env) => env.event),
    [
      {
        type: "tool_call",
        id: "cmd",
        name: "command",
        input: { command: "pwd" },
        raw: messages[2],
      },
    ],
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
  assert.deepEqual(clients[0]?.config, {
    model_reasoning_summary: "auto",
    cli_auth_credentials_store: "ephemeral",
  });
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
test("Codex turn.failed is the one terminal typed failure, not also a warning", async () => {
  const message = "401 Unauthorized";
  const { agent } = harness(
    [
      { type: "error", message },
      { type: "turn.failed", error: { message } },
    ],
    {},
    new Error("secondary exit failure"),
  );
  const run = await agent.run("bad credentials");
  assert.equal(run.status, "failure");
  assert.equal(run.error, message);
  assert.deepEqual(
    run.events
      .filter((env) => env.event.type === "error")
      .map((env) => env.event),
    [
      {
        type: "error",
        message,
        error_type: "authentication_failed",
      },
    ],
  );
  assert.deepEqual(
    run.events
      .filter((env) => env.event.type === "warning")
      .map((env) => env.event),
    [],
  );
});
test("Codex stops reading at its terminal event", async () => {
  const { agent, closed } = harness([
    completed,
    {
      type: "item.completed",
      item: { type: "agent_message", id: "late", text: "late" },
    },
    completed,
  ]);
  const run = await agent.run("once");
  assert.equal(run.status, "success");
  assert.equal(run.final_text, "");
  assert.equal(
    run.events.filter((env) => env.event.type === "usage").length,
    1,
  );
  assert.equal(closed(), 1);
});
test("Codex todo lists map to thinking instead of unmapped warnings", async () => {
  const run = await harness([
    {
      type: "item.completed",
      item: {
        type: "todo_list",
        id: "plan",
        items: [
          { text: "inspect", completed: true },
          { text: "fix", completed: false },
        ],
      },
    },
    completed,
  ]).agent.run("plan");
  assert.equal(
    run.events.some((env) => env.event.type === "warning"),
    false,
  );
  assert.deepEqual(run.events[1]?.event, {
    type: "thinking",
    text: "- [x] inspect\n- [ ] fix",
  });
});
test("Codex transient error events are classified and item errors remain warnings", async () => {
  const failure = await harness([
    { type: "turn.failed", error: { message: "429 rate limit" } },
  ]).agent.run("fail");
  assert.ok(
    failure.events.some(
      (env) =>
        env.event.type === "error" &&
        env.event.error_type === "transient_api_error",
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
test("Codex truncated streams fail; signal exits throw", async () => {
  const truncated = await harness([{ type: "turn.started" }]).agent.run(
    "truncated",
  );
  assert.equal(truncated.status, "failure");
  assert.match(truncated.error ?? "", /without turn.completed/);
  const killed = harness(
    [],
    {},
    new Error("Codex Exec exited with signal SIGTERM:"),
  );
  await assert.rejects(killed.agent.run("killed"), ProcessTerminatedError);
  assert.equal(killed.clients.length, 1);
});
test("closing a Codex stream closes its iterator without aborting the native signal", async () => {
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
  // The SDK's cleanup kills the child; a later abort would raise an uncaught AbortError.
  assert.equal(turns[0]?.signal?.aborted, false);
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
  await assert.rejects(
    agent.run("isolated"),
    (error) =>
      error instanceof ProviderError &&
      error.errorType === "authentication_failed",
  );
  assert.equal(clients.length, 0);
});
test("Codex cliLogin require keeps API keys out of the child and skips the ephemeral store", async () => {
  assert.throws(
    () =>
      harness([], {
        cliLogin: "require",
        providerOptions: { provider: "openai", client: { apiKey: "k" } },
      }),
    ConfigError,
  );
  const { agent, clients } = harness([completed], {
    cliLogin: "require",
    providerOptions: {
      provider: "openai",
      client: { env: { OPENAI_API_KEY: "k", CODEX_API_KEY: "k", HOME: "/h" } },
    },
  });
  await agent.run("login");
  assert.equal(clients[0]?.apiKey, undefined);
  assert.deepEqual(clients[0]?.env, { HOME: "/h" });
  assert.deepEqual(clients[0]?.config, { model_reasoning_summary: "auto" });
});
test("Codex cliLogin require accepts only a stored ChatGPT login", async (t) => {
  const dir = mkdtempSync(join(tmpdir(), "codex-login-"));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  const script = join(dir, "codex");
  writeFileSync(
    script,
    '#!/bin/sh\n[ "$1 $2" = "login status" ] && echo "$FAKE_LOGIN" >&2\n',
    { mode: 0o755 },
  );
  const agent = (status: string) =>
    new Agent({
      provider: "openai",
      cliLogin: "require",
      providerOptions: {
        provider: "openai",
        client: { codexPathOverride: script, env: { FAKE_LOGIN: status } },
      },
    });
  await agent("Logged in using ChatGPT").checkRuntime();
  for (const status of ["Logged in using an API key - sk-***", "Not logged in"])
    await assert.rejects(
      agent(status).checkRuntime(),
      (error) =>
        error instanceof ProviderError &&
        error.errorType === "authentication_failed",
    );
});
test("Codex cliLogin deny keeps an access-token login out of the child", async () => {
  const { agent, clients } = harness([completed], {
    providerOptions: {
      provider: "openai",
      client: { apiKey: "k", env: { CODEX_ACCESS_TOKEN: "token", HOME: "/h" } },
    },
  });
  await agent.run("deny");
  assert.deepEqual(clients[0]?.env, { HOME: "/h" });
});
test("Codex web search calls wait for their query", async () => {
  const { agent } = harness([
    { type: "thread.started", thread_id: "t" },
    { type: "item.started", item: { type: "web_search", id: "w", query: "" } },
    {
      type: "item.completed",
      item: { type: "web_search", id: "w", query: "codex sdk" },
    },
    completed,
  ]);
  const run = await agent.run("search");
  const calls = run.events
    .map((env) => env.event)
    .filter((event) => event.type === "tool_call");
  assert.deepEqual(
    calls.map((event) => event.type === "tool_call" && event.input),
    [{ query: "codex sdk" }],
  );
});
