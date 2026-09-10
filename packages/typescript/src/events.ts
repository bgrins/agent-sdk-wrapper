/** JSON field names deliberately match the language-neutral v1 schemas. */
export type Provider = "anthropic" | "openai";
export type RunStatus = "success" | "failure" | "cancelled";
export type RunEndedReason =
  | "success"
  | "error"
  | "max_turns"
  | "refused"
  | "cancelled";
export interface TokenUsage {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  cache_read_tokens: number;
  cache_write_tokens: number;
  reasoning_output_tokens: number;
  /** Zero means the SDK does not expose the request count. */
  requests: number;
}
export const emptyUsage = (): TokenUsage => ({
  input_tokens: 0,
  output_tokens: 0,
  total_tokens: 0,
  cache_read_tokens: 0,
  cache_write_tokens: 0,
  reasoning_output_tokens: 0,
  requests: 0,
});
type Raw = { raw?: Record<string, unknown> };
export type ErrorEvent = Raw & {
  type: "error";
  message: string;
  error_type: string;
  retryable: boolean;
};
export type ProviderEvent =
  | (Raw & { type: "text"; text: string })
  | (Raw & { type: "thinking"; text: string; redacted_bytes?: number })
  | (Raw & {
      type: "tool_call";
      id: string;
      name: string;
      input?: Record<string, unknown>;
    })
  | (Raw & {
      type: "tool_result";
      id: string;
      name?: string;
      output?: string;
      is_error: boolean;
    })
  | (Raw & { type: "usage"; usage: TokenUsage; cost_usd?: number })
  | { type: "session_info"; id: string }
  | (Raw & { type: "warning"; message: string })
  | ErrorEvent;
export type AgentEvent =
  | ProviderEvent
  | {
      type: "run_started";
      provider: Provider;
      prompt: string;
      model?: string;
      cwd?: string;
      system_prompt?: string;
    }
  | {
      type: "run_finished";
      status: RunStatus;
      ended_reason: RunEndedReason;
      duration_ms: number;
    };
export interface EventEnvelope {
  run_id: string;
  sequence: number;
  timestamp: string;
  event: AgentEvent;
}
export interface RunResult {
  run_id: string;
  provider: Provider;
  model: string | null;
  status: RunStatus;
  ended_reason: RunEndedReason;
  final_text: string;
  structured_output: null;
  usage: TokenUsage | null;
  cost_usd: number | null;
  duration_ms: number;
  session_id: string | null;
  artifacts_dir: null;
  error: string | null;
  events: EventEnvelope[];
}
