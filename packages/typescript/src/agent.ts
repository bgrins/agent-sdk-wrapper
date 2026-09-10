import { randomUUID } from "node:crypto";
import { setTimeout as delay } from "node:timers/promises";
import {
  ConfigError,
  ProcessTerminatedError,
  ProviderError,
  ProviderProtocolError,
  RuntimeUnavailableError,
  TransientError,
} from "./errors.js";
import {
  emptyUsage,
  type AgentEvent,
  type ErrorEvent,
  type EventEnvelope,
  type Provider,
  type RunEndedReason,
  type RunResult,
  type RunStatus,
} from "./events.js";
import type { ProviderAdapter } from "./providers/base.js";
import { buildProvider } from "./providers/index.js";
import {
  resolveRequest,
  type AgentDefaults,
  type ResolvedRequest,
  type RunRequest,
} from "./request.js";

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
    try {
      await adapter.ensureAvailable(req);
      const start = performance.now();
      const runId = randomUUID();
      let sequence = 0;
      const frame = (event: AgentEvent): EventEnvelope => ({
        run_id: runId,
        sequence: sequence++,
        timestamp: new Date().toISOString(),
        event,
      });
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
      let failure: ErrorEvent | undefined;
      for (let attempt = 0; ; attempt++) {
        let progressed = false;
        try {
          req.signal?.throwIfAborted();
          for await (const event of adapter.stream(req, {
            onNativeEvent: (native) => {
              progressed = true;
              req.onProviderEvent?.(native);
            },
          })) {
            progressed = true;
            if (event.type === "error") failure ??= event;
            if (event.type === "session_info") {
              this.sessions.set(req.provider, event.id);
              this.latestSession = event.id;
            }
            yield frame(event);
          }
          break;
        } catch (cause) {
          if (
            cause instanceof ProcessTerminatedError ||
            cause instanceof ConfigError
          )
            throw cause;
          if (failure) break; // Keep the provider's terminal error over a cleanup error.
          if (
            cause instanceof TransientError &&
            !progressed &&
            attempt < req.maxRetries &&
            !req.signal?.aborted
          ) {
            const ms = Math.min(
              req.retryDelayMs * 2 ** Math.min(attempt, 20),
              30_000,
            );
            yield frame({
              type: "warning",
              message: `Transient failure; retry ${attempt + 1}/${req.maxRetries} in ${ms}ms: ${cause.message}`,
            });
            try {
              await delay(ms, undefined, { signal: req.signal });
            } catch {
              /* cancellation is normalized below */
            }
            if (!req.signal?.aborted) continue;
          }
          failure = {
            type: "error",
            message: req.signal?.aborted
              ? "Run cancelled"
              : cause instanceof Error
                ? cause.message
                : String(cause),
            error_type: req.signal?.aborted
              ? "cancelled"
              : cause instanceof TransientError
                ? "transient_api_error"
                : cause instanceof ProviderError
                  ? cause.errorType
                  : cause instanceof ProviderProtocolError
                    ? "provider_protocol_error"
                    : cause instanceof RuntimeUnavailableError
                      ? "runtime_unavailable"
                      : "provider_exception",
            retryable: cause instanceof TransientError && !req.signal?.aborted,
          };
          yield frame(failure);
          break;
        }
      }
      // A native iterator can finish cleanly after observing its abort signal.
      if (!failure && req.signal?.aborted) {
        failure = {
          type: "error",
          message: "Run cancelled",
          error_type: "cancelled",
          retryable: false,
        };
        yield frame(failure);
      }
      const reason: RunEndedReason = !failure
        ? "success"
        : failure.error_type === "cancelled"
          ? "cancelled"
          : failure.error_type === "max_turns"
            ? "max_turns"
            : failure.error_type === "refused"
              ? "refused"
              : "error";
      const status: RunStatus =
        reason === "success"
          ? "success"
          : reason === "cancelled"
            ? "cancelled"
            : "failure";
      yield frame({
        type: "run_finished",
        status,
        ended_reason: reason,
        duration_ms: Math.max(0, Math.round(performance.now() - start)),
      });
    } finally {
      this.active = false;
    }
  }
}

/** Consume one stream while optionally rendering/writing every envelope. No second model call. */
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
    if (event.type === "text") result.final_text += event.text;
    if (event.type === "session_info") result.session_id = event.id;
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
