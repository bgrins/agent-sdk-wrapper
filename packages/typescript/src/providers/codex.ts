import type {
  CodexOptions,
  ThreadEvent,
  ThreadItem,
  ThreadOptions,
  TurnOptions,
} from "@openai/codex-sdk";
import {
  ConfigError,
  ProviderProtocolError,
  RuntimeUnavailableError,
} from "../errors.js";
import { emptyUsage, type ProviderEvent, type TokenUsage } from "../events.js";
import type { ResolvedRequest } from "../request.js";
import type { ProviderAdapter, ProviderContext } from "./base.js";
import {
  classify,
  enumOption,
  envOption,
  executable,
  nativeError,
  object,
  options,
  stringOption,
  stringsOption,
} from "./common.js";

interface NativeThread {
  runStreamed(
    prompt: string,
    options: TurnOptions,
  ): Promise<{ events: AsyncIterable<ThreadEvent> }>;
}
interface NativeCodex {
  startThread(options: ThreadOptions): NativeThread;
  resumeThread(id: string, options: ThreadOptions): NativeThread;
}
type CodexFactory = (options: CodexOptions) => NativeCodex;
export class CodexAdapter implements ProviderAdapter {
  readonly name = "openai";
  private readonly usageBaselines = new Map<string, TokenUsage>();
  constructor(private readonly factory?: CodexFactory) {}
  validateRequest(req: ResolvedRequest): void {
    const native = req.providerOptions;
    if (native?.provider !== "openai" && native !== undefined)
      throw new ConfigError("Expected openai providerOptions");
    options(native, ["provider", "client", "thread"], "providerOptions");
    options(
      native?.client,
      ["apiKey", "baseUrl", "env", "codexPathOverride"],
      "openai.client",
    );
    options(
      native?.thread,
      [
        "sandboxMode",
        "skipGitRepoCheck",
        "networkAccessEnabled",
        "webSearchMode",
        "approvalPolicy",
        "additionalDirectories",
      ],
      "openai.thread",
    );
    const thread = native?.thread;
    enumOption(
      thread?.sandboxMode,
      ["read-only", "workspace-write", "danger-full-access"],
      "sandboxMode",
    );
    enumOption(
      thread?.webSearchMode,
      ["disabled", "cached", "live"],
      "webSearchMode",
    );
    enumOption(
      thread?.approvalPolicy,
      ["never", "on-request", "on-failure", "untrusted"],
      "approvalPolicy",
    );
    for (const key of ["skipGitRepoCheck", "networkAccessEnabled"] as const)
      if (thread?.[key] !== undefined && typeof thread[key] !== "boolean")
        throw new ConfigError(`${key} must be boolean`);
    stringsOption(thread?.additionalDirectories, "additionalDirectories");
    for (const key of ["apiKey", "baseUrl", "codexPathOverride"] as const)
      stringOption(native?.client?.[key], key);
    envOption(native?.client?.env);
  }
  private async client(req: ResolvedRequest): Promise<NativeCodex> {
    const native =
      req.providerOptions?.provider === "openai"
        ? req.providerOptions.client
        : undefined;
    const opts: CodexOptions = {
      apiKey: native?.env
        ? native.env.OPENAI_API_KEY
        : process.env.OPENAI_API_KEY,
      ...native,
      config: { model_reasoning_summary: "auto" },
    };
    return this.factory
      ? this.factory(opts)
      : new (await import("@openai/codex-sdk")).Codex(opts);
  }
  async ensureAvailable(req: ResolvedRequest): Promise<void> {
    if (this.factory) return;
    try {
      const override =
        req.providerOptions?.provider === "openai"
          ? req.providerOptions.client?.codexPathOverride
          : undefined;
      if (override) await executable(override);
      await this.client(req); // The SDK constructor resolves its bundled runtime without spawning it.
    } catch (cause) {
      throw new RuntimeUnavailableError(
        "Codex SDK/runtime unavailable; install its platform optional dependency or supply codexPathOverride",
        { cause },
      );
    }
  }
  async *stream(
    req: ResolvedRequest,
    context: ProviderContext,
  ): AsyncGenerator<ProviderEvent> {
    const abort = new AbortController();
    const onAbort = () => abort.abort();
    req.signal?.addEventListener("abort", onAbort, { once: true });
    if (req.signal?.aborted) abort.abort();
    let terminal = false;
    let session = req.sessionId;
    let sawReasoning = false;
    const started = new Set<string>();
    const completed = new Set<string>();
    try {
      const client = await this.client(req);
      const native =
        req.providerOptions?.provider === "openai"
          ? req.providerOptions.thread
          : undefined;
      const opts: ThreadOptions = {
        ...native,
        model: req.model,
        modelReasoningEffort: req.effort,
        workingDirectory: req.cwd,
      };
      const thread = req.sessionId
        ? client.resumeThread(req.sessionId, opts)
        : client.startThread(opts);
      const { events } = await thread.runStreamed(req.prompt, {
        signal: abort.signal,
      });
      for await (const event of events) {
        context.onNativeEvent(event);
        const raw = req.includeRaw
          ? { raw: event as unknown as Record<string, unknown> }
          : {};
        if (event.type === "thread.started") {
          session = event.thread_id;
          yield { type: "session_info", id: session };
        } else if (event.type === "turn.completed") {
          if (terminal)
            throw new ProviderProtocolError(
              "Codex emitted more than one terminal result",
            );
          terminal = true;
          const nativeUsage = event.usage;
          const input = nativeUsage.input_tokens;
          // Pinned exec forwards ThreadTokenUsage.total, not per-turn usage. The
          // Responses output_tokens total already includes its reasoning subset.
          const output = nativeUsage.output_tokens;
          const cumulative: TokenUsage = {
            input_tokens: input,
            output_tokens: output,
            total_tokens: input + output,
            cache_read_tokens: nativeUsage.cached_input_tokens,
            cache_write_tokens: nativeUsage.cache_write_input_tokens,
            reasoning_output_tokens: nativeUsage.reasoning_output_tokens,
            requests: 0,
          };
          const baseline = session
            ? this.usageBaselines.get(session)
            : undefined;
          const usage = emptyUsage();
          const reset =
            baseline &&
            Object.keys(cumulative).some(
              (key) =>
                cumulative[key as keyof TokenUsage] <
                baseline[key as keyof TokenUsage],
            );
          for (const key of Object.keys(usage) as (keyof TokenUsage)[])
            usage[key] =
              cumulative[key] - (!reset ? (baseline?.[key] ?? 0) : 0);
          if (session) this.usageBaselines.set(session, cumulative);
          if (req.sessionId && !baseline)
            yield {
              type: "warning",
              message:
                "Codex usage includes prior history: this adapter has no baseline for the resumed thread",
            };
          if (reset)
            yield {
              type: "warning",
              message:
                "Codex usage counters reset; treating the new snapshot as the run's usage",
            };
          if (usage.reasoning_output_tokens > 0 && !sawReasoning)
            yield { type: "thinking", text: "" };
          yield { type: "usage", usage, ...raw };
        } else if (event.type === "turn.failed" || event.type === "error") {
          terminal = true;
          yield {
            ...classify(
              event.type === "turn.failed"
                ? event.error.message
                : event.message,
              event.type === "turn.failed" ? "turn_failed" : "stream_error",
            ),
            ...raw,
          };
        } else if (
          event.type === "item.started" ||
          event.type === "item.updated" ||
          event.type === "item.completed"
        ) {
          const item = event.item;
          const tool = toolInfo(item);
          if (tool && !started.has(item.id)) {
            started.add(item.id);
            yield { type: "tool_call", id: item.id, ...tool, ...raw };
          }
          if (event.type !== "item.completed" || completed.has(item.id))
            continue;
          completed.add(item.id);
          if (item.type === "agent_message")
            yield { type: "text", text: item.text, ...raw };
          else if (item.type === "reasoning") {
            sawReasoning = true;
            yield { type: "thinking", text: item.text, ...raw };
          } else if (tool)
            yield {
              type: "tool_result",
              id: item.id,
              name: tool.name,
              ...toolResult(item),
              ...raw,
            };
          else if (item.type === "error")
            yield { type: "warning", message: item.message, ...raw };
          else
            yield {
              type: "warning",
              message: `Unmapped Codex item: ${item.type}`,
              ...raw,
            };
        }
      }
      if (!terminal)
        throw new ProviderProtocolError(
          "Codex stream ended without turn.completed or a terminal error",
        );
    } catch (cause) {
      throw nativeError(cause);
    } finally {
      abort.abort();
      req.signal?.removeEventListener("abort", onAbort);
    }
  }
}
function toolInfo(
  item: ThreadItem,
): { name: string; input?: Record<string, unknown> } | undefined {
  switch (item.type) {
    case "command_execution":
      return { name: "command_execution", input: { command: item.command } };
    case "file_change":
      return { name: "file_change", input: { changes: item.changes } };
    case "mcp_tool_call":
      return {
        name: `${item.server}.${item.tool}`,
        input: object(item.arguments),
      };
    case "web_search":
      return { name: "web_search", input: { query: item.query } };
    default:
      return undefined;
  }
}
function toolResult(item: ThreadItem): { output?: string; is_error: boolean } {
  if (item.type === "command_execution")
    return {
      output: item.aggregated_output,
      is_error:
        item.status !== "completed" ||
        (item.exit_code !== undefined && item.exit_code !== 0),
    };
  if (item.type === "file_change")
    return {
      output: JSON.stringify(item.changes),
      is_error: item.status !== "completed",
    };
  if (item.type === "mcp_tool_call")
    return {
      output: JSON.stringify(item.error ?? item.result ?? null),
      is_error: item.status !== "completed",
    };
  return { is_error: false }; // The native web-search item exposes no result body.
}
