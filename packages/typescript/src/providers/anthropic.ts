import { createRequire } from "node:module";
import type {
  Options,
  SDKMessage,
  SDKResultMessage,
} from "@anthropic-ai/claude-agent-sdk";
import {
  ConfigError,
  ProviderProtocolError,
  RuntimeUnavailableError,
} from "../errors.js";
import { emptyUsage, type ProviderEvent } from "../events.js";
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

type NativeQuery = AsyncIterable<SDKMessage> & { close(): void };
type QueryFn = (params: { prompt: string; options: Options }) => NativeQuery;
export class AnthropicAdapter implements ProviderAdapter {
  readonly name = "anthropic";
  constructor(private readonly queryFn?: QueryFn) {}
  validateRequest(req: ResolvedRequest): void {
    const native = req.providerOptions;
    if (native?.provider !== "anthropic" && native !== undefined)
      throw new ConfigError("Expected anthropic providerOptions");
    options(native, ["provider", "options"], "providerOptions");
    const opts = native?.options;
    options(
      opts,
      [
        "permissionMode",
        "allowDangerouslySkipPermissions",
        "allowedTools",
        "disallowedTools",
        "tools",
        "settingSources",
        "env",
        "pathToClaudeCodeExecutable",
        "systemPrompt",
        "maxTurns",
        "thinking",
      ],
      "anthropic.options",
    );
    enumOption(
      opts?.permissionMode,
      [
        "default",
        "acceptEdits",
        "bypassPermissions",
        "plan",
        "dontAsk",
        "auto",
      ],
      "permissionMode",
    );
    if (
      opts?.allowDangerouslySkipPermissions !== undefined &&
      typeof opts.allowDangerouslySkipPermissions !== "boolean"
    )
      throw new ConfigError("allowDangerouslySkipPermissions must be boolean");
    if (
      opts?.permissionMode === "bypassPermissions" &&
      opts.allowDangerouslySkipPermissions !== true
    )
      throw new ConfigError(
        "bypassPermissions requires allowDangerouslySkipPermissions: true",
      );
    for (const key of [
      "allowedTools",
      "disallowedTools",
      "settingSources",
    ] as const)
      stringsOption(opts?.[key], key);
    if (
      opts?.settingSources?.some(
        (value) => !["user", "project", "local"].includes(value),
      )
    )
      throw new ConfigError("Invalid settingSources");
    if (opts?.tools !== undefined && !Array.isArray(opts.tools))
      throw new ConfigError(
        "This slice accepts only an explicit native tools array",
      );
    stringsOption(opts?.tools, "tools");
    stringOption(
      opts?.pathToClaudeCodeExecutable,
      "pathToClaudeCodeExecutable",
    );
    if (
      opts?.systemPrompt !== undefined &&
      typeof opts.systemPrompt !== "string"
    )
      throw new ConfigError(
        "This slice accepts only a string native systemPrompt",
      );
    if (
      opts?.maxTurns !== undefined &&
      (!Number.isSafeInteger(opts.maxTurns) || opts.maxTurns < 1)
    )
      throw new ConfigError("maxTurns must be a positive integer");
    options(opts?.thinking, ["type", "display", "budgetTokens"], "thinking");
    if (opts?.thinking) {
      if (!opts.thinking.type)
        throw new ConfigError("thinking.type is required");
      enumOption(
        opts.thinking.type,
        ["adaptive", "enabled", "disabled"],
        "thinking.type",
      );
      const thinking = object(opts.thinking);
      enumOption(
        thinking?.display,
        ["summarized", "omitted"],
        "thinking.display",
      );
      if (thinking?.type === "disabled" && thinking.display !== undefined)
        throw new ConfigError(
          "thinking.display is not supported for disabled thinking",
        );
      if (
        thinking?.budgetTokens !== undefined &&
        (thinking.type !== "enabled" ||
          !Number.isSafeInteger(thinking.budgetTokens) ||
          Number(thinking.budgetTokens) < 1)
      )
        throw new ConfigError(
          "thinking.budgetTokens requires enabled thinking and a positive integer",
        );
    }
    envOption(opts?.env);
  }
  async ensureAvailable(req: ResolvedRequest): Promise<void> {
    if (this.queryFn) return;
    try {
      await import("@anthropic-ai/claude-agent-sdk");
      const override =
        req.providerOptions?.provider === "anthropic"
          ? req.providerOptions.options?.pathToClaudeCodeExecutable
          : undefined;
      const require = createRequire(import.meta.url);
      const suffix = process.platform === "win32" ? "claude.exe" : "claude";
      await executable(
        override ??
          require.resolve(
            `@anthropic-ai/claude-agent-sdk-${process.platform}-${process.arch}/${suffix}`,
          ),
      );
    } catch (cause) {
      throw new RuntimeUnavailableError(
        "Claude SDK/runtime unavailable; install its platform optional dependency or supply pathToClaudeCodeExecutable",
        { cause },
      );
    }
  }
  async *stream(
    req: ResolvedRequest,
    context: ProviderContext,
  ): AsyncGenerator<ProviderEvent> {
    const native =
      req.providerOptions?.provider === "anthropic"
        ? req.providerOptions.options
        : undefined;
    const abort = new AbortController();
    const onAbort = () => abort.abort();
    req.signal?.addEventListener("abort", onAbort, { once: true });
    if (req.signal?.aborted) abort.abort();
    let query: NativeQuery | undefined;
    let terminal = false;
    let seenText = false;
    let seenThinking = false;
    let interrupted = false;
    let session: string | undefined;
    const names = new Map<string, string>();
    const seen = new Set<string>();
    try {
      const queryFn =
        this.queryFn ?? (await import("@anthropic-ai/claude-agent-sdk")).query;
      query = queryFn({
        prompt: req.prompt,
        options: {
          settingSources: [],
          thinking: { type: "adaptive", display: "summarized" },
          ...native,
          model: req.model,
          cwd: req.cwd,
          effort: req.effort as Options["effort"],
          resume: req.sessionId,
          includePartialMessages: false,
          abortController: abort,
        },
      });
      for await (const message of query) {
        context.onNativeEvent(message);
        const raw = req.includeRaw
          ? { raw: message as unknown as Record<string, unknown> }
          : {};
        // v1 text/tool events are append-only: we cannot retract output the
        // consumer has already received without extending the shared contract.
        if (
          (message.type === "assistant" && message.supersedes?.length) ||
          (message.type === "system" &&
            message.subtype === "model_refusal_fallback" &&
            message.retracted_message_uuids?.length)
        )
          throw new ProviderProtocolError(
            "Claude message retractions are not implemented in the v1 event contract; partial output must not be treated as a completed answer",
          );
        if (
          "session_id" in message &&
          message.session_id &&
          message.session_id !== session &&
          !("parent_tool_use_id" in message && message.parent_tool_use_id)
        ) {
          session = message.session_id;
          yield { type: "session_info", id: session };
        }
        if (message.type === "assistant") {
          if (seen.has(message.uuid)) continue;
          seen.add(message.uuid);
          if (message.parent_tool_use_id) {
            yield {
              type: "warning",
              message:
                "Subagent message omitted from portable output; inspect onProviderEvent",
              ...raw,
            };
            continue;
          }
          if (message.aborted) {
            interrupted = true;
            continue; // Truncated content is not a completed text/thinking item.
          }
          for (const block of message.message.content) {
            if (block.type === "text") {
              seenText = true;
              yield { type: "text", text: block.text, ...raw };
            } else if (block.type === "thinking") {
              seenThinking = true;
              yield { type: "thinking", text: block.thinking, ...raw };
            } else if (block.type === "redacted_thinking") {
              seenThinking = true;
              yield {
                type: "thinking",
                text: "",
                redacted_bytes: Buffer.byteLength(block.data),
                ...raw,
              };
            } else if (
              block.type === "tool_use" ||
              block.type === "server_tool_use"
            ) {
              names.set(block.id, block.name);
              yield {
                type: "tool_call",
                id: block.id,
                name: block.name,
                input: object(block.input),
                ...raw,
              };
            } else
              yield {
                type: "warning",
                message: `Unmapped Claude content block: ${block.type}`,
                ...raw,
              };
          }
        } else if (
          message.type === "user" &&
          !message.parent_tool_use_id &&
          Array.isArray(message.message.content)
        ) {
          for (const block of message.message.content) {
            if (block.type === "tool_result")
              yield {
                type: "tool_result",
                id: block.tool_use_id,
                name: names.get(block.tool_use_id),
                output:
                  typeof block.content === "string"
                    ? block.content
                    : JSON.stringify(block.content ?? null),
                is_error: block.is_error ?? false,
                ...raw,
              };
          }
        } else if (message.type === "result") {
          if (terminal)
            throw new ProviderProtocolError(
              "Claude emitted more than one terminal result",
            );
          terminal = true;
          const usage = usageEvent(message, raw);
          if (usage.usage.reasoning_output_tokens > 0 && !seenThinking)
            yield { type: "thinking", text: "", ...raw };
          yield usage;
          if (
            interrupted ||
            message.terminal_reason === "aborted_streaming" ||
            message.terminal_reason === "aborted_tools"
          ) {
            yield {
              type: "error",
              message: "Run cancelled",
              error_type: "cancelled",
              retryable: false,
              ...raw,
            };
            continue;
          }
          const status =
            message.subtype === "success"
              ? message.api_error_status
              : undefined;
          if (
            message.is_error ||
            message.subtype !== "success" ||
            message.stop_reason === "refusal" ||
            (message.terminal_reason && message.terminal_reason !== "completed")
          ) {
            const text =
              message.subtype === "success"
                ? message.result
                : message.errors.join("\n");
            const error = classify(
              text || message.subtype,
              message.terminal_reason ??
                (message.subtype === "success"
                  ? "result_error"
                  : message.subtype),
              status ?? undefined,
            );
            // A success-subtype error with no HTTP response is the SDK's
            // dropped-connection shape. Structural reasons still take priority.
            if (
              message.subtype === "success" &&
              message.is_error &&
              status == null &&
              (!message.terminal_reason ||
                message.terminal_reason === "api_error" ||
                message.terminal_reason === "completed") &&
              (error.error_type === "result_error" ||
                error.error_type === "api_error" ||
                error.error_type === "completed")
            ) {
              error.error_type = "transient_api_error";
              error.retryable = true;
            }
            if (
              message.subtype === "error_max_turns" ||
              message.terminal_reason === "max_turns"
            ) {
              error.error_type = "max_turns";
              error.retryable = false;
            }
            if (message.stop_reason === "refusal") {
              error.error_type = "refused";
              error.retryable = false;
            }
            yield { ...error, ...raw };
          } else if (!seenText && message.result)
            yield { type: "text", text: message.result, ...raw };
        } else if (message.type === "stream_event")
          throw new ProviderProtocolError(
            "Unexpected partial Claude frames with includePartialMessages disabled",
          );
      }
      if (!terminal)
        throw new ProviderProtocolError("Claude stream ended without a result");
    } catch (cause) {
      throw nativeError(cause);
    } finally {
      abort.abort();
      query?.close();
      req.signal?.removeEventListener("abort", onAbort);
    }
  }
}
function usageEvent(
  message: SDKResultMessage,
  raw: { raw?: Record<string, unknown> },
): Extract<ProviderEvent, { type: "usage" }> {
  const usage = emptyUsage();
  const models = Object.values(message.modelUsage);
  if (models.length) {
    for (const model of models) {
      usage.input_tokens +=
        model.inputTokens +
        model.cacheReadInputTokens +
        model.cacheCreationInputTokens;
      usage.output_tokens += model.outputTokens;
      usage.cache_read_tokens += model.cacheReadInputTokens;
      usage.cache_write_tokens += model.cacheCreationInputTokens;
      usage.reasoning_output_tokens += model.thinkingTokens ?? 0;
    }
  } else {
    usage.cache_read_tokens = message.usage.cache_read_input_tokens ?? 0;
    usage.cache_write_tokens = message.usage.cache_creation_input_tokens ?? 0;
    usage.input_tokens =
      message.usage.input_tokens +
      usage.cache_read_tokens +
      usage.cache_write_tokens;
    usage.output_tokens = message.usage.output_tokens;
    usage.reasoning_output_tokens =
      message.usage.output_tokens_details?.thinking_tokens ?? 0;
  }
  usage.total_tokens = usage.input_tokens + usage.output_tokens;
  return { type: "usage", usage, cost_usd: message.total_cost_usd, ...raw };
}
