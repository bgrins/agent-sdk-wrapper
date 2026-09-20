import { randomUUID } from "node:crypto";
import { setTimeout as delay } from "node:timers/promises";
import {
  ConfigError,
  ProcessTerminatedError,
  ProviderError,
  ProviderProtocolError,
  RuntimeUnavailableError,
  TraceWriteError,
  TransientError,
} from "./errors.js";
import {
  type AgentEvent,
  type ErrorEvent,
  type EventEnvelope,
  emptyUsage,
  type Provider,
  type RunEndedReason,
  type RunResult,
} from "./events.js";
import type { ProviderAdapter } from "./providers/base.js";
import { buildProvider } from "./providers/index.js";
import {
  type AgentDefaults,
  type ResolvedRequest,
  type RunRequest,
  resolveRequest,
} from "./request.js";
import { TraceWriter } from "./trace.js";

const progressEvents = new Set<AgentEvent["type"]>([
  "text",
  "thinking",
  "tool_call",
  "tool_result",
]);
const cancelledError = (): ErrorEvent => ({
  type: "error",
  message: "Run cancelled",
  error_type: "cancelled",
  retryable: false,
});
function errorEvent(cause: unknown): ErrorEvent {
  return {
    type: "error",
    message: cause instanceof Error ? cause.message : String(cause),
    error_type:
      cause instanceof TransientError
        ? "transient_api_error"
        : cause instanceof ProviderError
          ? cause.errorType
          : cause instanceof ProviderProtocolError
            ? "provider_protocol_error"
            : cause instanceof RuntimeUnavailableError
              ? "runtime_unavailable"
              : "provider_exception",
    retryable: cause instanceof TransientError,
  };
}

export class Agent {
  private readonly defaults: AgentDefaults;
  private readonly adapters: Partial<Record<Provider, ProviderAdapter>>;
  private readonly sessions = new Map<Provider, string>();
  private active = false;
  private latestSession: string | undefined;
  constructor(
    defaults: AgentDefaults,
    adapters: Partial<Record<Provider, ProviderAdapter>> = {},
  ) {
    this.defaults = { ...defaults };
    this.adapters = { ...adapters };
    const [req] = this.request({ prompt: "" });
    this.latestSession = req.sessionId;
    if (req.sessionId) this.sessions.set(req.provider, req.sessionId);
  }
  get sessionId(): string | undefined {
    return this.latestSession;
  }
  private request(input: RunRequest): [ResolvedRequest, ProviderAdapter] {
    const req = resolveRequest({ ...this.defaults, ...input });
    if (!req.sessionId && req.continueSession)
      req.sessionId = this.sessions.get(req.provider);
    const adapter = this.adapters[req.provider] ?? buildProvider(req.provider);
    this.adapters[req.provider] = adapter;
    if (adapter.name !== req.provider)
      throw new ConfigError(
        "Adapter name does not match the resolved provider",
      );
    adapter.validateRequest(req);
    return [req, adapter];
  }
  async checkRuntime(overrides: AgentDefaults = {}): Promise<void> {
    const [req, adapter] = this.request({ ...overrides, prompt: "" });
    await adapter.ensureAvailable(req);
  }
  run(input: string | RunRequest): Promise<RunResult> {
    return collectRun(this.stream(input));
  }
  async *stream(input: string | RunRequest): AsyncGenerator<EventEnvelope> {
    if (this.active)
      throw new ConfigError(
        "An Agent supports one active run; use separate instances for concurrent runs",
      );
    const [req, adapter] = this.request(
      typeof input === "string" ? { prompt: input } : input,
    );
    this.active = true;
    let writer: TraceWriter | undefined;
    try {
      await adapter.ensureAvailable(req);
      if (req.traceFile !== undefined) writer = new TraceWriter(req.traceFile);
      const start = performance.now();
      const runId = randomUUID();
      let sequence = 0;
      const frame = (event: AgentEvent): EventEnvelope => {
        const envelope = {
          run_id: runId,
          sequence: sequence++,
          timestamp: new Date().toISOString(),
          event,
        };
        writer?.write(envelope);
        return envelope;
      };
      const systemPrompt =
        req.providerOptions?.provider === "anthropic"
          ? req.providerOptions.options?.systemPrompt
          : undefined;
      yield frame({
        type: "run_started",
        provider: req.provider,
        prompt: req.prompt,
        ...(req.model ? { model: req.model } : {}),
        ...(req.cwd ? { cwd: req.cwd } : {}),
        ...(typeof systemPrompt === "string"
          ? { system_prompt: systemPrompt }
          : {}),
      });
      const finished = (failure: ErrorEvent | undefined) => {
        const reason: RunEndedReason = !failure
          ? "success"
          : failure.error_type === "cancelled"
            ? "cancelled"
            : failure.error_type === "max_turns"
              ? "max_turns"
              : failure.error_type === "refused"
                ? "refused"
                : "error";
        return frame({
          type: "run_finished",
          status:
            reason === "success"
              ? "success"
              : reason === "cancelled"
                ? "cancelled"
                : "failure",
          ended_reason: reason,
          duration_ms: Math.max(0, Math.round(performance.now() - start)),
        });
      };
      let failure: ErrorEvent | undefined;
      for (let attempt = 0; ; attempt++) {
        let progressed = false;
        let sessionSeen = false;
        let held: ErrorEvent | undefined;
        let threw = false;
        let thrown: unknown;
        failure = undefined;
        // A resumed session already holds this prompt; retrying would repeat it.
        const canRetry = () =>
          !progressed &&
          !(req.sessionId && sessionSeen) &&
          attempt < req.maxRetries &&
          !req.signal?.aborted;
        try {
          req.signal?.throwIfAborted();
          for await (const event of adapter.stream(req, {
            onNativeEvent: (native) => req.onProviderEvent?.(native),
          })) {
            // Progress or another error rules out a retry; show the held error first.
            if (
              held &&
              (progressEvents.has(event.type) || event.type === "error")
            ) {
              yield frame(held);
              held = undefined;
            }
            if (progressEvents.has(event.type)) progressed = true;
            if (event.type === "error") {
              // Hold a retryable error until the attempt ends; a retry replaces it with a warning.
              if (!failure && event.retryable && canRetry()) {
                failure = held = event;
                continue;
              }
              failure ??= event;
            }
            if (event.type === "session_info") {
              sessionSeen = true;
              this.sessions.set(req.provider, event.id);
              this.latestSession = event.id;
            }
            yield frame(event);
          }
        } catch (cause) {
          if (cause instanceof TraceWriteError) throw cause;
          // A runtime killed by the caller's abort was cancelled, not terminated.
          const terminated =
            cause instanceof ProcessTerminatedError && !req.signal?.aborted;
          if (terminated || cause instanceof ConfigError) {
            if (held) yield frame(held);
            const error: ErrorEvent = {
              type: "error",
              message: cause.message,
              error_type: terminated ? "process_terminated" : "invalid_request",
              retryable: false,
            };
            yield frame(error);
            yield finished(held ?? error);
            throw cause;
          }
          threw = true;
          thrown = cause;
        }
        // The provider's terminal error wins over a later cleanup error.
        const retryReason =
          held?.message ??
          (!failure && thrown instanceof TransientError
            ? thrown.message
            : undefined);
        let error: ErrorEvent | undefined;
        if (retryReason !== undefined && canRetry()) {
          const ms = Math.min(
            req.retryDelayMs * 2 ** Math.min(attempt, 20),
            30_000,
          );
          yield frame({
            type: "warning",
            message: `Transient failure; retry ${attempt + 1}/${req.maxRetries} in ${ms}ms: ${retryReason}`,
          });
          try {
            await delay(ms, undefined, { signal: req.signal });
          } catch {
            /* cancellation is normalized below */
          }
          if (!req.signal?.aborted) continue;
          error = cancelledError();
        } else if (held && req.signal?.aborted) {
          yield frame({ type: "warning", message: held.message });
          error = cancelledError();
        } else if (held) error = held;
        else if (!failure && threw)
          error = req.signal?.aborted ? cancelledError() : errorEvent(thrown);
        if (error) {
          failure = error;
          yield frame(error);
        }
        break;
      }
      yield finished(failure);
    } finally {
      try {
        writer?.close();
      } finally {
        this.active = false;
      }
    }
  }
}

/** Collect one stream, optionally passing each envelope to a callback. */
export async function collectRun(
  events: AsyncIterable<EventEnvelope>,
  onEvent?: (event: EventEnvelope) => void | Promise<void>,
): Promise<RunResult> {
  let result: RunResult | undefined;
  let finished = false;
  let expected = 0;
  for await (const envelope of events) {
    if (
      finished ||
      envelope.sequence !== expected++ ||
      (result && envelope.run_id !== result.run_id)
    )
      throw new ProviderProtocolError("Invalid event envelope order");
    const event = envelope.event;
    if (!result) {
      if (event.type !== "run_started")
        throw new ProviderProtocolError("Expected run_started");
      result = {
        run_id: envelope.run_id,
        provider: event.provider,
        model: event.model ?? null,
        status: "failure",
        ended_reason: "error",
        final_text: "",
        structured_output: null,
        usage: null,
        cost_usd: null,
        duration_ms: 0,
        session_id: null,
        artifacts_dir: null,
        error: null,
        events: [],
      };
    } else if (event.type === "run_started")
      throw new ProviderProtocolError("Duplicate run_started");
    result.events.push(envelope);
    // Native SDKs report the last assistant message as the final response.
    if (event.type === "text") result.final_text = event.text;
    if (event.type === "session_info") {
      result.session_id = event.id;
      if (event.model) result.model = event.model;
    }
    if (event.type === "error") result.error ??= event.message;
    if (event.type === "usage") {
      result.usage ??= emptyUsage();
      for (const key of Object.keys(
        result.usage,
      ) as (keyof typeof result.usage)[])
        result.usage[key] += event.usage[key];
      if (event.cost_usd !== undefined)
        result.cost_usd = (result.cost_usd ?? 0) + event.cost_usd;
    }
    if (event.type === "run_finished") {
      finished = true;
      result.status = event.status;
      result.ended_reason = event.ended_reason;
      result.duration_ms = event.duration_ms;
    }
    await onEvent?.(envelope);
  }
  if (!result || !finished)
    throw new ProviderProtocolError("Stream ended without run_finished");
  return result;
}
