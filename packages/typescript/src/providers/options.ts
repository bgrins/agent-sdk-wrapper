import type { Options } from "@anthropic-ai/claude-agent-sdk";
import type { CodexOptions, ThreadOptions } from "@openai/codex-sdk";
/** Deliberately curated native controls. Unknown keys are rejected at runtime. */
export type AnthropicNativeOptions = Pick<
  Options,
  | "permissionMode"
  | "allowDangerouslySkipPermissions"
  | "allowedTools"
  | "disallowedTools"
  | "settingSources"
  | "pathToClaudeCodeExecutable"
  | "maxTurns"
  | "thinking"
> & {
  /** Preset objects are not implemented in this slice. */
  tools?: string[];
  systemPrompt?: string;
  env?: Record<string, string>;
};
export type CodexNativeOptions = Pick<
  CodexOptions,
  "apiKey" | "baseUrl" | "env" | "codexPathOverride"
>;
export type CodexThreadOptions = Pick<
  ThreadOptions,
  | "sandboxMode"
  | "skipGitRepoCheck"
  | "networkAccessEnabled"
  | "webSearchMode"
  | "approvalPolicy"
  | "additionalDirectories"
>;
export type ProviderOptions =
  | { provider: "anthropic"; options?: AnthropicNativeOptions }
  | {
      provider: "openai";
      client?: CodexNativeOptions;
      thread?: CodexThreadOptions;
    };
