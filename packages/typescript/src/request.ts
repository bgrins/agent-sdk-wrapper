import { statSync } from "node:fs";
import { ConfigError } from "./errors.js";
import type { Provider } from "./events.js";
import type { ProviderOptions } from "./providers/options.js";

export type ProviderInput = Provider | "codex";
export type Effort =
  | "minimal"
  | "low"
  | "medium"
  | "high"
  | "xhigh"
  | "max"
  | "ultra"
  | "persistent";
/** Whether the runtime may use its stored login. `require` is Codex-only. */
export type CliLogin = "deny" | "require";
export interface AgentDefaults {
  provider?: ProviderInput;
  model?: string;
  effort?: Effort;
  cwd?: string;
  sessionId?: string;
  continueSession?: boolean;
  includeRaw?: boolean;
  signal?: AbortSignal;
  providerOptions?: ProviderOptions;
  /** A returned promise that rejects fails the run like a thrown exception. */
  onProviderEvent?: (event: unknown) => void | PromiseLike<void>;
  traceFile?: string;
  cliLogin?: CliLogin;
  // Reserved features fail at compile time and at runtime, including empty values.
  tools?: never;
  mcpServers?: never;
  outputSchema?: never;
  systemPrompt?: never;
  subagents?: never;
  maxTurns?: never;
  timeout?: never;
  artifactsDir?: never;
  builtinTools?: never;
  permissionMode?: never;
}
export interface RunRequest extends AgentDefaults {
  prompt: string;
}
export interface ResolvedRequest extends RunRequest {
  provider: Provider;
  continueSession: boolean;
  includeRaw: boolean;
  cliLogin: CliLogin;
}
const keys = new Set([
  "prompt",
  "provider",
  "model",
  "effort",
  "cwd",
  "sessionId",
  "continueSession",
  "includeRaw",
  "signal",
  "providerOptions",
  "onProviderEvent",
  "traceFile",
  "cliLogin",
]);
export function checkKeys(
  value: object,
  allowed: ReadonlySet<string>,
  label: string,
): void {
  for (const key of Object.keys(value)) {
    if (!allowed.has(key))
      throw new ConfigError(`${label}.${key} is not implemented or recognized`);
  }
}
function knownProvider(value: string): Provider | undefined {
  const name = value.trim().toLowerCase();
  if (name === "codex" || name === "openai") return "openai";
  if (name === "anthropic") return name;
  return undefined;
}
export function normalizeProvider(value: string): Provider {
  if (typeof value !== "string")
    throw new ConfigError("provider must be a string");
  const provider = knownProvider(value);
  if (!provider)
    throw new ConfigError(
      `Unknown provider '${value}'; expected anthropic, openai, or codex`,
    );
  return provider;
}
export function resolveProvider(
  provider?: string,
  model?: string,
): { provider: Provider; model?: string } {
  if (provider !== undefined && typeof provider !== "string")
    throw new ConfigError("provider must be a string");
  if (model !== undefined && typeof model !== "string")
    throw new ConfigError("model must be a string");
  // Blank values mean omitted, as with an empty environment variable.
  if (!provider?.trim()) provider = undefined;
  let name = model?.trim() || undefined;
  // Bedrock IDs, ARNs and fine-tune names contain colons; only a provider name is a prefix.
  const colon = name?.indexOf(":") ?? -1;
  const prefix =
    name && colon >= 0 ? knownProvider(name.slice(0, colon)) : undefined;
  if (name && prefix) {
    name = name.slice(colon + 1).trim();
    if (!name)
      throw new ConfigError("Expected provider:model with a non-empty model");
  }
  const explicit =
    provider === undefined ? undefined : normalizeProvider(provider);
  const inferred =
    name &&
    (/^claude/i.test(name)
      ? "anthropic"
      : /^(gpt-|o[1345]|codex|chatgpt-)/i.test(name)
        ? "openai"
        : undefined);
  if (explicit && prefix && explicit !== prefix)
    throw new ConfigError("Model prefix conflicts with provider");
  if ((explicit ?? prefix) && inferred && (explicit ?? prefix) !== inferred)
    throw new ConfigError("Model name belongs to a different provider");
  const selected = explicit ?? prefix ?? inferred;
  if (!selected)
    throw new ConfigError(
      "Specify provider when the model is omitted or cannot be inferred",
    );
  return { provider: selected, ...(name ? { model: name } : {}) };
}
function isDirectory(path: string): boolean {
  try {
    return statSync(path).isDirectory();
  } catch {
    return false;
  }
}
export function resolveRequest(input: RunRequest): ResolvedRequest {
  checkKeys(input, keys, "request");
  if (typeof input.prompt !== "string")
    throw new ConfigError("prompt must be a string");
  for (const key of ["cwd", "sessionId", "traceFile"] as const) {
    if (
      input[key] !== undefined &&
      (typeof input[key] !== "string" || !input[key]?.trim())
    )
      throw new ConfigError(`${key} must be a non-empty string`);
  }
  if (input.traceFile?.includes("\0"))
    throw new ConfigError("traceFile must not contain NUL characters");
  if (input.cwd !== undefined && !isDirectory(input.cwd))
    throw new ConfigError(`cwd is not a directory: ${input.cwd}`);
  for (const key of ["includeRaw", "continueSession"] as const) {
    if (input[key] !== undefined && typeof input[key] !== "boolean")
      throw new ConfigError(`${key} must be boolean`);
  }
  if (
    input.cliLogin !== undefined &&
    input.cliLogin !== "deny" &&
    input.cliLogin !== "require"
  )
    throw new ConfigError("cliLogin must be 'deny' or 'require'");
  if (input.signal !== undefined && !(input.signal instanceof AbortSignal))
    throw new ConfigError("signal must be an AbortSignal");
  if (
    input.onProviderEvent !== undefined &&
    typeof input.onProviderEvent !== "function"
  )
    throw new ConfigError("onProviderEvent must be a function");
  const resolved = resolveProvider(input.provider, input.model);
  const allowedEfforts =
    resolved.provider === "anthropic"
      ? ["low", "medium", "high", "xhigh", "max"]
      : [
          "minimal",
          "low",
          "medium",
          "high",
          "xhigh",
          "max",
          "ultra",
          "persistent",
        ];
  if (input.effort !== undefined && !allowedEfforts.includes(input.effort))
    throw new ConfigError(
      `${resolved.provider} does not support effort '${input.effort}' in the pinned SDK`,
    );
  if (
    input.providerOptions !== undefined &&
    (typeof input.providerOptions !== "object" ||
      input.providerOptions === null ||
      input.providerOptions.provider !== resolved.provider)
  )
    throw new ConfigError(
      "providerOptions.provider must match the canonical provider",
    );
  return {
    ...input,
    provider: resolved.provider,
    model: resolved.model,
    continueSession: input.continueSession ?? false,
    includeRaw: input.includeRaw ?? false,
    cliLogin: input.cliLogin ?? "deny",
  };
}
