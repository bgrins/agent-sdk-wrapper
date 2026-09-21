import { createServer, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";

/** One scripted model response; see docs/fixtures/CONFORMANCE.md. */
export interface Step {
  text?: string;
  thinking?: string;
  tool?: ToolCall | ToolCall[];
  shell?: string;
  tool_search?: string;
  /** Input and output tokens; Codex also takes reasoning tokens. */
  usage?: [number, number, number?];
  stop_reason?: string;
  status?: number;
  headers?: Record<string, string>;
  body?: unknown;
  stream_error?: Record<string, unknown>;
  truncate?: boolean;
  hang?: number;
}
interface ToolCall {
  name: string;
  input?: Record<string, unknown>;
}
export interface Recorded {
  path: string;
  headers: Record<string, string | string[] | undefined>;
  body: unknown;
  step: number;
}
export type MockProvider = "anthropic" | "codex";
export interface Mock {
  /** Claude's ANTHROPIC_BASE_URL, or Codex's provider base_url. */
  url: string;
  /** Model requests, in arrival order. */
  requests: Recorded[];
  close(): Promise<void>;
}

type Event = [name: string, data: Record<string, unknown>];
const sse = (events: Event[]) =>
  events
    .map(([name, data]) => `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`)
    .join("");
const json = "application/json";

/** Serve Claude Messages or Codex Responses requests from `steps`; the last step repeats. */
export async function startMock(
  provider: MockProvider,
  steps: Step[],
): Promise<Mock> {
  const requests: Recorded[] = [];
  const route = provider === "anthropic" ? "/v1/messages" : "/v1/responses";
  let used = 0;
  const server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (chunk: Buffer) => chunks.push(chunk));
    req.on("end", () => {
      const path = req.url ?? "";
      if (req.method !== "POST" || path.split("?")[0] !== route) {
        const count = path.includes("/count_tokens");
        res
          .writeHead(count ? 200 : 404, { "content-type": json })
          .end(count ? '{"input_tokens":10}' : "{}");
        return;
      }
      const text = Buffer.concat(chunks).toString("utf8");
      let body: unknown = text;
      try {
        body = JSON.parse(text);
      } catch {}
      const fields = (body ?? {}) as { stream?: unknown; model?: string };
      const streaming = provider === "codex" || fields.stream === true;
      // The Claude CLI retries a failed stream once without streaming; that
      // request replays the failed step instead of taking the next one.
      const index = !streaming && used > 0 ? used - 1 : used++;
      const step = steps[Math.min(index, steps.length - 1)] ?? {};
      requests.push({ path, headers: req.headers, body, step: index });
      respond(provider, res, step, index, streaming, fields.model);
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const origin = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  return {
    url: provider === "anthropic" ? origin : `${origin}/v1`,
    requests,
    close: () =>
      new Promise((resolve) => {
        server.closeAllConnections();
        server.close(() => resolve());
      }),
  };
}

function respond(
  provider: MockProvider,
  res: ServerResponse,
  step: Step,
  index: number,
  streaming: boolean,
  model = "mock-model",
): void {
  if (step.hang !== undefined) {
    const timer = setTimeout(() => res.destroy(), step.hang * 1000);
    res.on("close", () => clearTimeout(timer));
    return;
  }
  const send = (status: number, body: unknown, type = json) =>
    res
      .writeHead(status, { "content-type": type, ...step.headers })
      .end(typeof body === "string" ? body : JSON.stringify(body));
  if (step.status !== undefined) {
    const text = typeof step.body === "string";
    send(step.status, step.body ?? {}, text ? "text/plain" : json);
    return;
  }
  if (!streaming) {
    if (step.stream_error) {
      const kind = step.stream_error.type;
      const status =
        kind === "overloaded_error"
          ? 529
          : kind === "rate_limit_error"
            ? 429
            : 500;
      send(status, { type: "error", error: step.stream_error });
    } else if (step.truncate) {
      res.writeHead(200, { "content-type": json, "content-length": "1000" });
      res.write('{"id": "msg_', () => res.destroy());
    } else send(200, claudeMessage(step, index, model, false));
    return;
  }
  const events =
    provider === "anthropic"
      ? claudeEvents(step, index, model)
      : codexEvents(step, index);
  res.writeHead(200, { "content-type": "text/event-stream", ...step.headers });
  if (step.truncate) res.write(sse(events.slice(0, 1)), () => res.destroy());
  else res.end(sse(events));
}

const toolCalls = (step: Step, shell: (command: string) => ToolCall) => [
  ...[step.tool ?? []].flat(),
  ...(step.shell !== undefined ? [shell(step.shell)] : []),
];

function claudeBlocks(step: Step, index: number): Record<string, unknown>[] {
  const blocks: Record<string, unknown>[] = [];
  if (step.thinking !== undefined)
    blocks.push({
      type: "thinking",
      thinking: step.thinking,
      signature: "sig",
    });
  if (step.text !== undefined) blocks.push({ type: "text", text: step.text });
  const calls = toolCalls(step, (command) => ({
    name: "Bash",
    input: { command },
  }));
  calls.forEach((call, n) => {
    blocks.push({
      type: "tool_use",
      id: `toolu_${index}_${n}`,
      name: call.name,
      input: call.input ?? {},
    });
  });
  return blocks;
}

function claudeMessage(
  step: Step,
  index: number,
  model: string,
  started: boolean,
): Record<string, unknown> {
  const blocks = claudeBlocks(step, index);
  const tool = blocks.some((block) => block.type === "tool_use");
  return {
    id: `msg_mock_${index}`,
    type: "message",
    role: "assistant",
    model,
    content: started ? [] : blocks,
    stop_reason: started
      ? null
      : (step.stop_reason ?? (tool ? "tool_use" : "end_turn")),
    stop_sequence: null,
    usage: {
      input_tokens: step.usage?.[0] ?? 100,
      output_tokens: started ? 1 : (step.usage?.[1] ?? 10),
      cache_creation_input_tokens: 0,
      cache_read_input_tokens: 0,
    },
  };
}

function claudeEvents(step: Step, index: number, model: string): Event[] {
  const message = claudeMessage(step, index, model, true);
  const events: Event[] = [
    ["message_start", { type: "message_start", message }],
  ];
  claudeBlocks(step, index).forEach((block, n) => {
    const delta = (delta: Record<string, unknown>): Event => [
      "content_block_delta",
      { type: "content_block_delta", index: n, delta },
    ];
    const empty =
      block.type === "thinking"
        ? { ...block, thinking: "", signature: "" }
        : block.type === "text"
          ? { ...block, text: "" }
          : { ...block, input: {} };
    events.push([
      "content_block_start",
      { type: "content_block_start", index: n, content_block: empty },
    ]);
    if (block.type === "thinking")
      events.push(
        delta({ type: "thinking_delta", thinking: block.thinking }),
        delta({ type: "signature_delta", signature: block.signature }),
      );
    else if (block.type === "text")
      events.push(delta({ type: "text_delta", text: block.text }));
    else
      events.push(
        delta({
          type: "input_json_delta",
          partial_json: JSON.stringify(block.input),
        }),
      );
    // A stream error arrives inside the open block.
    if (!step.stream_error)
      events.push([
        "content_block_stop",
        { type: "content_block_stop", index: n },
      ]);
  });
  if (step.stream_error)
    return [...events, ["error", { type: "error", error: step.stream_error }]];
  const final = claudeMessage(step, index, model, false);
  return [
    ...events,
    [
      "message_delta",
      {
        type: "message_delta",
        delta: { stop_reason: final.stop_reason, stop_sequence: null },
        usage: { output_tokens: step.usage?.[1] ?? 10 },
      },
    ],
    ["message_stop", { type: "message_stop" }],
  ];
}

function codexEvents(step: Step, index: number): Event[] {
  const id = `resp_${index}`;
  const items: Record<string, unknown>[] = [];
  if (step.thinking !== undefined)
    items.push({
      type: "reasoning",
      id: `rs_${index}`,
      summary: [{ type: "summary_text", text: step.thinking }],
    });
  if (step.tool_search !== undefined)
    items.push({
      type: "tool_search_call",
      id: `ts_${index}`,
      call_id: `ts_${index}`,
      execution: "client",
      status: "completed",
      arguments: { query: step.tool_search },
    });
  const calls = toolCalls(step, (cmd) => ({
    name: "exec_command",
    input: { cmd },
  }));
  calls.forEach(({ name, input }, n) => {
    // Codex addresses MCP tools as a namespace plus the tool's own name.
    const mcp = /^(mcp__.+?)__(.+)$/.exec(name);
    items.push({
      type: "function_call",
      id: `fc_${index}_${n}`,
      call_id: `call_${index}_${n}`,
      name: mcp ? mcp[2] : name,
      ...(mcp ? { namespace: mcp[1] } : {}),
      arguments: JSON.stringify(input ?? {}),
    });
  });
  if (step.text !== undefined)
    items.push({
      type: "message",
      role: "assistant",
      id: `msg_${index}`,
      content: [{ type: "output_text", text: step.text }],
    });
  const [input, output, reasoning = 0] = step.usage ?? [100, 10];
  const done: Event = step.stream_error
    ? [
        "response.failed",
        {
          type: "response.failed",
          response: { id, status: "failed", error: step.stream_error },
        },
      ]
    : [
        "response.completed",
        {
          type: "response.completed",
          response: {
            id,
            usage: {
              input_tokens: input,
              input_tokens_details: { cached_tokens: 0 },
              output_tokens: output,
              output_tokens_details: { reasoning_tokens: reasoning },
              total_tokens: input + output,
            },
          },
        },
      ];
  return [
    ["response.created", { type: "response.created", response: { id } }],
    ...items.map(
      (item): Event => [
        "response.output_item.done",
        { type: "response.output_item.done", item },
      ],
    ),
    done,
  ];
}
