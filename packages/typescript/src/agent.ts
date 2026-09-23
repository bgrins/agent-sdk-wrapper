import { randomUUID } from "node:crypto";
import {
  ConfigError,
  ProcessTerminatedError,
  ProviderError,
  ProviderProtocolError,
  RuntimeUnavailableError,
  TraceWriteError,
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

const cancelledError = (): ErrorEvent => ({
  type: "error",
  message: "Run cancelled",
  error_type: "cancelled",
});
function errorEvent(cause: unknown): ErrorEvent {
  return {
    type: "error",
    message: cause instanceof Error ? cause.message : String(cause),
    error_type:
      cause instanceof ProviderError
        ? cause.errorType
        : cause instanceof ProviderProtocolError
          ? "provider_protocol_error"
          : cause instanceof RuntimeUnavailableError
            ? "runtime_unavailable"
            : "provider_exception",
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
    // A per-call sessionId wins; a constructor one gives way to the latest reported session.
    if (req.continueSession && input.sessionId === undefined)
      req.sessionId = this.sessions.get(req.provider) ?? req.sessionId;
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
    let released = false;
    // Runs before the final envelope is yielded: a consumer may stop pulling there.
    const release = () => {
      if (released) return;
      released = true;
      try {
        writer?.close();
      } finally {
        this.active = false;
      }
    };
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
        const envelope = frame({
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
        release();
        return envelope;
      };
      let failure: ErrorEvent | undefined;
      let threw = false;
      let thrown: unknown;
      // A callback exception is the caller's, not the provider's: rethrow it unclassified.
      let callback: { error: unknown } | undefined;
      // An async callback's rejection stops the runtime through the adapter's signal.
      const stop = new AbortController();
      const pending = new Set<Promise<void>>();
      const onNativeEvent = (native: unknown) => {
        let result: unknown;
        try {
          result = req.onProviderEvent?.(native);
        } catch (error) {
          callback ??= { error };
          throw error;
        }
        if (typeof (result as PromiseLike<unknown>)?.then !== "function")
          return;
        const settled: Promise<void> = Promise.resolve(result).then(
          () => {
            pending.delete(settled);
          },
          (error: unknown) => {
            pending.delete(settled);
            callback ??= { error };
            stop.abort(error);
          },
        );
        pending.add(settled);
      };
      const signal = req.signal
        ? AbortSignal.any([req.signal, stop.signal])
        : stop.signal;
      try {
        req.signal?.throwIfAborted();
        for await (const event of adapter.stream(
          { ...req, signal },
          { onNativeEvent },
        )) {
          if (callback) throw callback.error;
          if (event.type === "error") failure ??= event;
          if (event.type === "session_info") {
            this.sessions.set(req.provider, event.id);
            this.latestSession = event.id;
          }
          yield frame(event);
        }
        // A rejection after the last native event still fails the run.
        await Promise.all(pending);
        if (callback) throw callback.error;
      } catch (cause) {
        await Promise.all(pending);
        if (callback) throw callback.error;
        if (cause instanceof TraceWriteError) throw cause;
        // A runtime killed by the caller's abort was cancelled, not terminated.
        const terminated =
          cause instanceof ProcessTerminatedError && !req.signal?.aborted;
        if (terminated || cause instanceof ConfigError) {
          const error: ErrorEvent = {
            type: "error",
            message: cause.message,
            error_type: terminated ? "process_terminated" : "invalid_request",
          };
          yield frame(error);
          yield finished(failure ?? error);
          throw cause;
        }
        threw = true;
        thrown = cause;
      }
      // The provider's terminal error wins over a later cleanup error.
      if (!failure && threw) {
        failure = req.signal?.aborted ? cancelledError() : errorEvent(thrown);
        yield frame(failure);
      }
      yield finished(failure);
    } finally {
      release();
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
        error_type: null,
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
    if (event.type === "error" && result.error === null) {
      result.error = event.message;
      result.error_type = event.error_type;
    }
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
