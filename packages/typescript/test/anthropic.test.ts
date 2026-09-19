import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { test } from "node:test";
import type {
  SDKAssistantMessage,
  SDKMessage,
  SDKResultSuccess,
  Options,
} from "@anthropic-ai/claude-agent-sdk";
import { Agent, ConfigError, RuntimeUnavailableError } from "../src/index.js";
import type { AgentDefaults, AnthropicNativeOptions } from "../src/index.js";
import { AnthropicAdapter } from "../src/providers/anthropic.js";

const usage: SDKResultSuccess["usage"] = {
  input_tokens: 3,
  output_tokens: 4,
  cache_read_input_tokens: 5,
  cache_creation_input_tokens: 2,
  cache_creation: {
    ephemeral_1h_input_tokens: 0,
    ephemeral_5m_input_tokens: 2,
  },
  fallback_credit: { status: { type: "not_applied", reason: "not_enabled" } },
  inference_geo: "us",
  iterations: [],
  output_tokens_details: { thinking_tokens: 0 },
  server_tool_use: { web_fetch_requests: 0, web_search_requests: 0 },
  service_tier: "standard",
  speed: "standard",
};
function assistant(
  content: SDKAssistantMessage["message"]["content"],
  overrides: Partial<SDKAssistantMessage> = {},
  message: Partial<SDKAssistantMessage["message"]> = {},
): SDKAssistantMessage {
  return {
    type: "assistant",
    uuid: randomUUID(),
    session_id: "claude-session",
    parent_tool_use_id: null,
    message: {
      id: "native-message",
      type: "message",
      role: "assistant",
      model: "claude-test",
      content,
      stop_reason: null,
      stop_sequence: null,
      usage,
      container: null,
      context_management: null,
      diagnostics: null,
      stop_details: null,
      ...message,
    },
    ...overrides,
  };
}
function init(model: string): SDKMessage {
  return {
    type: "system",
    subtype: "init",
    apiKeySource: "ANTHROPIC_API_KEY",
    claude_code_version: "test",
    cwd: "/tmp",
    tools: [],
    mcp_servers: [],
    model,
    permissionMode: "default",
    slash_commands: [],
    output_style: "default",
    skills: [],
    plugins: [],
    uuid: randomUUID(),
    session_id: "claude-session",
  };
}
const textBlock = (value: string) =>
  ({ type: "text", text: value, citations: null }) as const;
function result(overrides: Partial<SDKResultSuccess> = {}): SDKResultSuccess {
  return {
    type: "result",
    subtype: "success",
    uuid: randomUUID(),
    session_id: "claude-session",
    is_error: false,
    duration_ms: 10,
    duration_api_ms: 5,
    num_turns: 1,
    result: "answer",
    stop_reason: "end_turn",
    total_cost_usd: 0.03,
    usage,
    modelUsage: {},
    permission_denials: [],
    ...overrides,
  };
}
function harness(
  messages: SDKMessage[],
  defaults: AgentDefaults = {},
  failure?: Error,
) {
  const captured: Options[] = [];
  let closed = 0;
  const provider = new AnthropicAdapter(({ options }) => {
    captured.push(options);
    return {
      async *[Symbol.asyncIterator]() {
        yield* messages;
        if (failure) throw failure;
      },
      close() {
        closed++;
      },
    };
  });
  return {
    agent: new Agent(
      { provider: "anthropic", ...defaults },
      { anthropic: provider },
    ),
    captured,
    closed: () => closed,
  };
}
test("Claude maps completed blocks, hidden reasoning, tool results, final usage and raw events", async () => {
  const text = assistant([{ type: "text", text: "answer", citations: null }]);
  const messages: SDKMessage[] = [
    assistant([
      { type: "thinking", thinking: "plan", signature: "s" },
      { type: "redacted_thinking", data: "abc" },
    ]),
    assistant([
      {
        type: "tool_use",
        id: "call",
        name: "Read",
        input: { file_path: "file" },
      },
    ]),
    {
      type: "user",
      session_id: "claude-session",
      parent_tool_use_id: null,
      message: {
        role: "user",
        content: [
          { type: "tool_result", tool_use_id: "call", content: "contents" },
        ],
      },
    },
    text,
    text,
    result(),
  ];
  const raw: unknown[] = [];
  const { agent, captured, closed } = harness(messages, {
    model: "claude-test",
    effort: "high",
    cwd: "/tmp",
    sessionId: "resume-me",
    includeRaw: true,
    onProviderEvent: (event) => {
      raw.push(event);
    },
  });
  const run = await agent.run("question");
  assert.equal(run.final_text, "answer");
  assert.deepEqual(
    run.events
      .filter((env) => env.event.type === "thinking")
      .map((env) => env.event.type === "thinking" && env.event.text),
    ["plan", ""],
  );
  assert.equal(
    run.events.find((env) => env.event.type === "tool_result")?.event.type,
    "tool_result",
  );
  assert.deepEqual(run.usage, {
    input_tokens: 10,
    output_tokens: 4,
    total_tokens: 14,
    cache_read_tokens: 5,
    cache_write_tokens: 2,
    reasoning_output_tokens: 0,
    requests: 0,
  });
  assert.equal(run.cost_usd, 0.03);
  assert.equal(run.session_id, "claude-session");
  assert.equal(raw.length, messages.length);
  assert.equal(captured[0]?.includePartialMessages, false);
  assert.equal(captured[0]?.resume, "resume-me");
  assert.equal(captured[0]?.effort, "high");
  assert.equal(captured[0]?.cwd, "/tmp");
  assert.deepEqual(captured[0]?.thinking, {
    type: "adaptive",
    display: "summarized",
  });
  assert.equal(closed(), 1);
  const normalizedText = run.events.find(
    (env) => env.event.type === "text",
  )?.event;
  assert.equal(
    normalizedText && "raw" in normalizedText && normalizedText.raw,
    text,
  );
});
test("Claude sums main and subagent modelUsage without adding main-loop usage or thinking twice", async () => {
  const { agent } = harness([
    result({
      modelUsage: {
        "claude-test": {
          inputTokens: 10,
          outputTokens: 7,
          thinkingTokens: 3,
          cacheReadInputTokens: 20,
          cacheCreationInputTokens: 4,
          webSearchRequests: 0,
          costUSD: 0.03,
          contextWindow: 1000,
          maxOutputTokens: 100,
        },
        "claude-subagent": {
          inputTokens: 5,
          outputTokens: 3,
          thinkingTokens: 1,
          cacheReadInputTokens: 6,
          cacheCreationInputTokens: 2,
          webSearchRequests: 0,
          costUSD: 0.01,
          contextWindow: 1000,
          maxOutputTokens: 100,
        },
      },
      total_cost_usd: 0.04,
    }),
  ]);
  const run = await agent.run("usage");
  assert.deepEqual(run.usage, {
    input_tokens: 47,
    output_tokens: 10,
    total_tokens: 57,
    cache_read_tokens: 26,
    cache_write_tokens: 6,
    reasoning_output_tokens: 4,
    requests: 0,
  });
  assert.equal(run.cost_usd, 0.04);
  assert.equal(run.final_text, "answer");
  assert.deepEqual(
    run.events
      .filter((env) => env.event.type === "thinking")
      .map((env) => env.event),
    [{ type: "thinking", text: "" }],
  );
  assert.ok(run.events.every((env) => !("raw" in env.event)));
});
test("Claude fallback usage preserves hidden thinking without double counting or duplicate events", async () => {
  for (const thinking of [
    [],
    [assistant([{ type: "thinking", thinking: "plan", signature: "s" }])],
  ]) {
    const run = await harness([
      ...thinking,
      result({
        usage: { ...usage, output_tokens_details: { thinking_tokens: 3 } },
      }),
    ]).agent.run("usage");
    assert.equal(run.usage?.reasoning_output_tokens, 3);
    assert.equal(run.usage?.output_tokens, 4);
    assert.equal(run.usage?.total_tokens, 14);
    assert.equal(
      run.events.filter((env) => env.event.type === "thinking").length,
      1,
    );
  }
});
for (const [overrides, expectedType, retryable] of [
  [
    { is_error: true, api_error_status: 429, result: "overloaded" },
    "transient_api_error",
    true,
  ],
  [
    { is_error: true, api_error_status: 401, result: "bad credentials" },
    "authentication_failed",
    false,
  ],
  [
    { is_error: true, api_error_status: null, result: "connection dropped" },
    "transient_api_error",
    true,
  ],
  [
    {
      is_error: true,
      terminal_reason: "api_error",
      result: "connection dropped",
    },
    "transient_api_error",
    true,
  ],
  [
    {
      is_error: true,
      terminal_reason: "budget_exhausted",
      result: "budget exhausted",
    },
    "max_budget",
    false,
  ],
  [{ stop_reason: "refusal", result: "declined" }, "refused", false],
] as const)
  test(`Claude terminal ${expectedType} fails without needing an exception`, async () => {
    const { agent } = harness([result(overrides)], {}, new Error("cleanup"));
    const run = await agent.run("fail");
    assert.equal(run.status, "failure");
    const errors = run.events.filter((env) => env.event.type === "error");
    assert.equal(errors.length, 1);
    assert.equal(
      errors[0]?.event.type === "error" && errors[0].event.error_type,
      expectedType,
    );
    assert.equal(
      errors[0]?.event.type === "error" && errors[0].event.retryable,
      retryable,
    );
  });
for (const reason of ["aborted_streaming", "aborted_tools"] as const)
  test(`Claude ${reason} is cancelled even without a thrown abort`, async () => {
    const { agent, closed } = harness(
      [
        result({
          terminal_reason: reason,
          is_error: true,
          result: "interrupted",
        }),
      ],
      {},
      new Error("cleanup"),
    );
    const run = await agent.run("cancel");
    assert.equal(run.status, "cancelled");
    assert.equal(run.ended_reason, "cancelled");
    assert.equal(run.final_text, "");
    assert.ok(run.usage);
    assert.deepEqual(run.events.at(-2)?.event, {
      type: "error",
      message: "Run cancelled",
      error_type: "cancelled",
      retryable: false,
    });
    assert.equal(closed(), 1);
  });
test("Claude interrupted assistant content is not emitted as a completed item", async () => {
  const partial = assistant([
    { type: "text", text: "truncat", citations: null },
  ]);
  partial.aborted = true;
  const run = await harness([partial, result()]).agent.run("cancel");
  assert.equal(run.status, "cancelled");
  assert.equal(run.final_text, "");
});
test("Claude retractions fail explicitly through either native notification", async () => {
  const original = assistant([
    { type: "text", text: "withdrawn", citations: null },
  ]);
  const replacement = assistant([
    { type: "text", text: "replacement", citations: null },
  ]);
  const notice: SDKMessage = {
    type: "system",
    subtype: "model_refusal_fallback",
    trigger: "refusal",
    direction: "retry",
    original_model: "claude-test",
    fallback_model: "claude-fallback",
    request_id: null,
    retracted_message_uuids: [original.uuid],
    content: "retracted",
    uuid: randomUUID(),
    session_id: "claude-session",
  };
  for (const messages of [
    [original, { ...replacement, supersedes: [original.uuid] }, result()],
    [original, notice, replacement, result()],
  ]) {
    const raw: unknown[] = [];
    const { agent, captured, closed } = harness(messages, {
      maxRetries: 2,
      onProviderEvent: (event) => {
        raw.push(event);
      },
    });
    const run = await agent.run("retract");
    assert.equal(run.status, "failure");
    assert.equal(run.ended_reason, "error");
    assert.match(run.error ?? "", /retractions are not implemented/);
    const error = run.events.at(-2)?.event;
    assert.equal(
      error?.type === "error" && error.error_type,
      "provider_protocol_error",
    );
    assert.ok(raw.length >= 2);
    assert.equal(captured.length, 1);
    assert.equal(closed(), 1);
  }
});
test("Claude excludes subagent tool results with their omitted calls", async () => {
  const run = await harness([
    {
      type: "user",
      session_id: "claude-session",
      parent_tool_use_id: "subagent",
      message: {
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id: "child-call",
            content: "child output",
          },
        ],
      },
    },
    result(),
  ]).agent.run("top level");
  assert.equal(
    run.events.some((env) => env.event.type === "tool_result"),
    false,
  );
});
test("Claude max turns maps to max_turns and truncated streams fail", async () => {
  const { result: _text, api_error_status: _status, ...base } = result();
  const message: SDKMessage = {
    ...base,
    subtype: "error_max_turns" as const,
    errors: ["turn limit"],
    is_error: true,
  };
  assert.equal(
    (await harness([message]).agent.run("limit")).ended_reason,
    "max_turns",
  );
  const truncated = harness([
    assistant([{ type: "text", text: "partial", citations: null }]),
  ]);
  assert.equal((await truncated.agent.run("truncated")).status, "failure");
  assert.equal(truncated.closed(), 1);
});
test("Claude continuation uses latest session and closes on consumer break", async () => {
  const { agent, captured, closed } = harness(
    [assistant([{ type: "text", text: "ok", citations: null }]), result()],
    { continueSession: true },
  );
  await agent.run("first");
  for await (const env of agent.stream("second"))
    if (env.event.type === "text") break;
  assert.equal(captured[1]?.resume, "claude-session");
  assert.equal(closed(), 2);
});
test("Claude rejects unimplemented native overrides and malformed option values", () => {
  for (const native of [
    { includePartialMessages: true },
    { mcpServers: {} },
    { agents: {} },
    { outputFormat: {} },
    { model: "override" },
    { permissionMode: "bad" },
    { permissionMode: "bypassPermissions" },
    {
      permissionMode: "bypassPermissions",
      allowDangerouslySkipPermissions: false,
    },
    { allowDangerouslySkipPermissions: "yes" },
    { maxTurns: 0 },
    { env: { A: 1 } },
    { thinking: { type: "enabled", budgetTokens: -1 } },
  ]) {
    assert.throws(
      () =>
        harness([], {
          providerOptions: { provider: "anthropic", options: native },
        } as AgentDefaults),
      ConfigError,
    );
  }
});
test("Claude forwards the explicit permission bypass prerequisite", async () => {
  const { agent, captured } = harness([result()], {
    providerOptions: {
      provider: "anthropic",
      options: {
        permissionMode: "bypassPermissions",
        allowDangerouslySkipPermissions: true,
      },
    },
  });
  await agent.run("fake runtime only");
  assert.equal(captured[0]?.permissionMode, "bypassPermissions");
  assert.equal(captured[0]?.allowDangerouslySkipPermissions, true);
});
test("Claude native option types and runtime agree on unsupported shapes", () => {
  const invalid: AnthropicNativeOptions[] = [
    // @ts-expect-error SDK presets are unsupported.
    { tools: { type: "preset", preset: "claude_code" } },
    // @ts-expect-error SDK presets are unsupported.
    { systemPrompt: { type: "preset", preset: "claude_code" } },
    // @ts-expect-error Native env values must be strings.
    { env: { EXAMPLE: undefined } },
  ];
  for (const options of invalid)
    assert.throws(
      () =>
        harness([], { providerOptions: { provider: "anthropic", options } }),
      ConfigError,
    );
});
test("Claude checkRuntime verifies a missing override without a model request", async () => {
  const agent = new Agent({
    provider: "anthropic",
    providerOptions: {
      provider: "anthropic",
      options: { pathToClaudeCodeExecutable: "/definitely/missing/claude" },
    },
  });
  await assert.rejects(agent.checkRuntime(), RuntimeUnavailableError);
});
test("Claude stops at its first result, ignoring background-task turns", async () => {
  const { agent, closed } = harness([
    assistant([textBlock("launched")]),
    result({ result: "launched" }),
    assistant([textBlock("background answer")], {}, { id: "second" }),
    result({ result: "background answer" }),
  ]);
  const run = await agent.run("subagent");
  assert.equal(run.status, "success");
  assert.equal(run.final_text, "launched");
  assert.equal(closed(), 1);
});
// Frames captured from the Claude CLI against a local mock API.
for (const [error, status, reason, message, expected] of [
  [
    "authentication_failed",
    null,
    "api_error",
    "Not logged in · Please run /login",
    "authentication_failed",
  ],
  [
    "authentication_failed",
    403,
    "api_error",
    "Failed to authenticate. API Error: 403 forbidden",
    "permission_denied",
  ],
  [
    "invalid_request",
    400,
    "prompt_too_long",
    "Prompt is too long",
    "context_window_exceeded",
  ],
  [
    "billing_error",
    400,
    "api_error",
    "Credit balance is too low",
    "billing_error",
  ],
  [
    "model_not_found",
    404,
    "api_error",
    "There's an issue with the selected model (claude-test). It may not exist or you may not have access to it.",
    "model_not_found",
  ],
  [
    "server_error",
    null,
    "api_error",
    "API Error: Connection refused — a firewall or proxy may be blocking it (ConnectionRefused)",
    "transient_api_error",
  ],
] as const)
  test(`Claude synthetic ${error} (${status}) is a ${expected} error, not text`, async () => {
    const { agent } = harness([
      assistant(
        [textBlock(message)],
        { error },
        { model: "<synthetic>", stop_reason: "stop_sequence" },
      ),
      result({
        is_error: true,
        api_error_status: status,
        terminal_reason: reason,
        stop_reason: "stop_sequence",
        result: message,
      }),
    ]);
    const run = await agent.run("fail");
    assert.equal(run.final_text, "");
    assert.equal(
      run.events.some((env) => env.event.type === "text"),
      false,
    );
    assert.deepEqual(
      run.events
        .filter((env) => env.event.type === "error")
        .map((env) => env.event),
      [
        {
          type: "error",
          message,
          error_type: expected,
          retryable: expected === "transient_api_error",
        },
      ],
    );
  });
test("Claude retries transient failures reported through synthetic messages", async () => {
  const message = "API Error: 529 overloaded";
  const { agent, captured } = harness(
    [
      init("claude-test"),
      assistant(
        [textBlock(message)],
        { error: "server_error" },
        { model: "<synthetic>" },
      ),
      result({ is_error: true, api_error_status: 529, result: message }),
    ],
    { maxRetries: 1, retryDelayMs: 0 },
  );
  const run = await agent.run("retry");
  assert.equal(captured.length, 2);
  assert.equal(run.error, message);
  assert.equal(
    run.events.filter((env) => env.event.type === "error").length,
    1,
  );
});
test("Claude child env disables background tasks and pins effort without replacing env semantics", async () => {
  const inherited = harness([result()], { effort: "low" });
  await inherited.agent.run("inherit");
  const env = inherited.captured[0]?.env;
  assert.equal(env?.CLAUDE_CODE_DISABLE_BACKGROUND_TASKS, "1");
  assert.equal(env?.CLAUDE_CODE_EFFORT_LEVEL, "low");
  assert.equal(env?.PATH, process.env.PATH);
  const options = (env: Record<string, string>): AgentDefaults => ({
    providerOptions: { provider: "anthropic", options: { env } },
  });
  const explicit = harness(
    [result()],
    options({ ONLY: "1", CLAUDE_CODE_DISABLE_BACKGROUND_TASKS: "0" }),
  );
  await explicit.agent.run("explicit");
  assert.deepEqual(explicit.captured[0]?.env, {
    ONLY: "1",
    CLAUDE_CODE_DISABLE_BACKGROUND_TASKS: "0",
  });
  const same = harness([result()], {
    effort: "high",
    ...options({ CLAUDE_CODE_EFFORT_LEVEL: "high" }),
  });
  await same.agent.run("same");
  assert.deepEqual(same.captured[0]?.env, {
    CLAUDE_CODE_EFFORT_LEVEL: "high",
    CLAUDE_CODE_DISABLE_BACKGROUND_TASKS: "1",
  });
  assert.throws(
    () =>
      harness([], {
        effort: "high",
        ...options({ CLAUDE_CODE_EFFORT_LEVEL: "low" }),
      }),
    ConfigError,
  );
});
test("Claude session_info reports the runtime model from init", async () => {
  const run = await harness(
    [init("claude-resolved"), assistant([textBlock("ok")]), result()],
    { model: "claude-test" },
  ).agent.run("model");
  assert.deepEqual(
    run.events
      .filter((env) => env.event.type === "session_info")
      .map((env) => env.event),
    [{ type: "session_info", id: "claude-session", model: "claude-resolved" }],
  );
  assert.equal(run.model, "claude-resolved");
});
test("Claude rate limit events become warnings", async () => {
  const run = await harness([
    {
      type: "rate_limit_event",
      rate_limit_info: {
        status: "allowed_warning",
        rateLimitType: "five_hour",
        utilization: 0.9,
        resetsAt: 1700000000,
      },
      uuid: randomUUID(),
      session_id: "claude-session",
    },
    result(),
  ]).agent.run("limits");
  assert.deepEqual(
    run.events
      .filter((env) => env.event.type === "warning")
      .map((env) => env.event),
    [
      {
        type: "warning",
        message:
          "Claude rate limit status: allowed_warning, type=five_hour, utilization=0.9, resets_at=1700000000",
      },
    ],
  );
});
test("Claude list tool results join their text blocks", async () => {
  const run = await harness([
    assistant([{ type: "tool_use", id: "call", name: "Agent", input: {} }]),
    {
      type: "user",
      session_id: "claude-session",
      parent_tool_use_id: null,
      message: {
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id: "call",
            content: [
              { type: "text", text: "RESULT_42" },
              { type: "text", text: "\nagentId: a1" },
            ],
          },
        ],
      },
    },
    result(),
  ]).agent.run("tool");
  const output = run.events.find(
    (env) => env.event.type === "tool_result",
  )?.event;
  assert.equal(
    output?.type === "tool_result" && output.output,
    "RESULT_42\nagentId: a1",
  );
});
