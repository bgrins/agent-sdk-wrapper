import { constants } from "node:fs";
import { access } from "node:fs/promises";
import { findPackageJSON } from "node:module";
import { dirname, join } from "node:path";
import type {
  Options,
  SDKAssistantMessageError,
  SDKMessage,
  SDKResultMessage,
} from "@anthropic-ai/claude-agent-sdk";
import {
  ConfigError,
  ProviderProtocolError,
  RuntimeUnavailableError,
} from "../errors.js";
import { type ErrorEvent, emptyUsage, type ProviderEvent } from "../events.js";
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
      throw new ConfigError("tools must be an array");
    stringsOption(opts?.tools, "tools");
    stringOption(
      opts?.pathToClaudeCodeExecutable,
      "pathToClaudeCodeExecutable",
    );
    if (
      opts?.systemPrompt !== undefined &&
      typeof opts.systemPrompt !== "string"
    )
      throw new ConfigError("systemPrompt must be a string");
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
    // The CLI ranks CLAUDE_CODE_EFFORT_LEVEL above --effort.
    const effortEnv = opts?.env?.CLAUDE_CODE_EFFORT_LEVEL;
    if (req.effort && effortEnv !== undefined && effortEnv !== req.effort)
      throw new ConfigError(
        "anthropic env CLAUDE_CODE_EFFORT_LEVEL conflicts with effort",
      );
  }
  async ensureAvailable(req: ResolvedRequest): Promise<void> {
    if (this.queryFn) return;
    try {
      await import("@anthropic-ai/claude-agent-sdk");
      const override =
        req.providerOptions?.provider === "anthropic"
          ? req.providerOptions.options?.pathToClaudeCodeExecutable
          : undefined;
      if (override) {
        // The SDK launches these scripts through an interpreter; require readability.
        if (
          [".js", ".mjs", ".tsx", ".ts", ".jsx"].some((ext) =>
            override.endsWith(ext),
          )
        )
          await access(override, constants.R_OK);
        else await executable(override);
      } else {
        // Resolve in the SDK dependency scope to support non-hoisted installs.
        const manifest = findPackageJSON(
          `@anthropic-ai/claude-agent-sdk-${process.platform}-${process.arch}`,
          import.meta.resolve("@anthropic-ai/claude-agent-sdk"),
        );
        if (!manifest) throw new Error("Claude platform package not found");
        const suffix = process.platform === "win32" ? "claude.exe" : "claude";
        await executable(join(dirname(manifest), suffix));
      }
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
    const additions: Record<string, string> = {};
    if (req.effort) additions.CLAUDE_CODE_EFFORT_LEVEL = req.effort;
    // Background subagents make the CLI emit an extra turn and a second result.
    if (native?.env?.CLAUDE_CODE_DISABLE_BACKGROUND_TASKS === undefined)
      additions.CLAUDE_CODE_DISABLE_BACKGROUND_TASKS = "1";
    const abort = new AbortController();
    const onAbort = () => abort.abort();
    req.signal?.addEventListener("abort", onAbort, { once: true });
    if (req.signal?.aborted) abort.abort();
    let query: NativeQuery | undefined;
    let seenText = false;
    let seenThinking = false;
    let interrupted = false;
    let assistantError: SDKAssistantMessageError | undefined;
    let session: string | undefined;
    let sessionModel: string | undefined;
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
          // The native env option replaces process.env.
          env: { ...(native?.env ?? process.env), ...additions },
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
        // The v1 contract cannot retract emitted text or tool events.
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
          !("parent_tool_use_id" in message && message.parent_tool_use_id)
        ) {
          const model =
            message.type === "system" && message.subtype === "init"
              ? message.model
              : sessionModel;
          if (message.session_id !== session || model !== sessionModel) {
            session = message.session_id;
            sessionModel = model;
            yield {
              type: "session_info",
              id: session,
              ...(model ? { model } : {}),
            };
          }
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
          // The CLI reports API failures as synthetic assistant text.
          if (message.error || message.message.model === "<synthetic>") {
            assistantError = message.error;
            continue;
          }
          assistantError = undefined;
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
          const usage = usageEvent(message, raw);
          if (usage.usage.reasoning_output_tokens > 0 && !seenThinking)
            yield { type: "thinking", text: "", ...raw };
          yield usage;
          const error = interrupted
            ? cancelled()
            : resultError(message, assistantError);
          if (error) yield { ...error, ...raw };
          else if (!seenText && message.subtype === "success" && message.result)
            yield { type: "text", text: message.result, ...raw };
          return;
        } else if (message.type === "stream_event")
          throw new ProviderProtocolError(
            "Unexpected partial Claude frames with includePartialMessages disabled",
          );
      }
      throw new ProviderProtocolError("Claude stream ended without a result");
    } catch (cause) {
      throw nativeError(cause);
    } finally {
      req.signal?.removeEventListener("abort", onAbort);
      abort.abort();
      query?.close();
    }
  }
}
const cancelled = (): ErrorEvent => ({
  type: "error",
  message: "Run cancelled",
  error_type: "cancelled",
  retryable: false,
});
const assistantErrorTypes: Partial<Record<SDKAssistantMessageError, string>> = {
  authentication_failed: "authentication_failed",
  verification_required: "authentication_failed",
  cloud_credential_error: "authentication_failed",
  oauth_org_not_allowed: "permission_denied",
  account_on_hold: "permission_denied",
  billing_error: "billing_error",
  rate_limit: "transient_api_error",
  overloaded: "transient_api_error",
  server_error: "transient_api_error",
  invalid_request: "invalid_request",
  model_not_found: "model_not_found",
};
/** Prefer subtype, terminal_reason and the assistant error over HTTP status and text. */
function resultError(
  message: SDKResultMessage,
  assistantError: SDKAssistantMessageError | undefined,
): ErrorEvent | undefined {
  const reason = message.terminal_reason;
  const text =
    (message.subtype === "success"
      ? message.result
      : message.errors.join("\n")) || message.subtype;
  const error = (error_type: string): ErrorEvent => ({
    type: "error",
    message: text,
    error_type,
    retryable: error_type === "transient_api_error",
  });
  if (reason === "aborted_streaming" || reason === "aborted_tools")
    return cancelled();
  if (message.subtype === "error_max_turns" || reason === "max_turns")
    return error("max_turns");
  if (
    message.subtype === "error_max_budget_usd" ||
    reason === "budget_exhausted"
  )
    return error("max_budget");
  if (
    message.subtype === "error_max_structured_output_retries" ||
    reason === "structured_output_retry_exhausted"
  )
    return error("structured_output_failed");
  if (message.stop_reason === "refusal") return error("refused");
  if (message.subtype === "error_during_execution")
    return error("execution_error");
  if (!message.is_error && (!reason || reason === "completed")) return;
  // The CLI groups these as context limits.
  if (
    reason === "prompt_too_long" ||
    reason === "blocking_limit" ||
    reason === "rapid_refill_breaker"
  )
    return error("context_window_exceeded");
  const status =
    message.subtype === "success"
      ? (message.api_error_status ?? undefined)
      : undefined;
  const structured = assistantError && assistantErrorTypes[assistantError];
  if (structured)
    return error(
      structured === "authentication_failed" && status === 403
        ? "permission_denied"
        : structured,
    );
  if (!message.is_error) return error("execution_error");
  const classified = classify(text, "execution_error", status);
  // An API error without an HTTP status or a recognizable message is a dropped connection.
  return status === undefined &&
    classified.error_type === "execution_error" &&
    (!reason || reason === "api_error" || reason === "completed")
    ? error("transient_api_error")
    : classified;
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
