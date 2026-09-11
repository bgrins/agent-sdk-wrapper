import { constants } from "node:fs";
import { access } from "node:fs/promises";
import {
  AgentSdkWrapperError,
  ConfigError,
  ProcessTerminatedError,
  ProviderError,
  RuntimeUnavailableError,
  TransientError,
} from "../errors.js";
import type { ErrorEvent } from "../events.js";
import { checkKeys } from "../request.js";

export function object(value: unknown): Record<string, unknown> | undefined {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}
export function options(
  value: unknown,
  allowed: readonly string[],
  label: string,
): void {
  if (value === undefined) return;
  const data = object(value);
  if (!data) throw new ConfigError(`${label} must be an object`);
  checkKeys(data, new Set(allowed), label);
}
export function enumOption(
  value: unknown,
  allowed: readonly string[],
  label: string,
): void {
  if (value !== undefined && !allowed.includes(value as string))
    throw new ConfigError(`Invalid ${label}; expected ${allowed.join(", ")}`);
}
export function stringOption(value: unknown, label: string): void {
  if (value !== undefined && (typeof value !== "string" || !value.trim()))
    throw new ConfigError(`${label} must be a non-empty string`);
}
export function stringsOption(value: unknown, label: string): void {
  if (
    value !== undefined &&
    (!Array.isArray(value) || !value.every((item) => typeof item === "string"))
  )
    throw new ConfigError(`${label} must be a string array`);
}
export function envOption(value: unknown): void {
  const data = object(value);
  if (
    value !== undefined &&
    (!data || !Object.values(data).every((item) => typeof item === "string"))
  )
    throw new ConfigError("Native env must map strings to strings");
}
export async function executable(path: string): Promise<void> {
  try {
    await access(path, constants.X_OK);
  } catch (cause) {
    throw new RuntimeUnavailableError(
      `Provider runtime is not executable: ${path}`,
      { cause },
    );
  }
}
export function classify(
  message: string,
  fallback: string,
  status?: number,
): ErrorEvent {
  const retryable =
    status === 429 ||
    (status !== undefined && status >= 500) ||
    /\b429\b|\b50[0-9]\b|rate.?limit|overloaded|temporarily unavailable|ECONNRESET|ETIMEDOUT/i.test(
      message,
    );
  const errorType = retryable
    ? "transient_api_error"
    : status === 401 ||
        /unauthorized|authentication|invalid.api.key|\b401\b/i.test(message)
      ? "authentication_failed"
      : status === 403 || /\b403\b|permission denied/i.test(message)
        ? "permission_denied"
        : /model.not.found|model.*does not exist/i.test(message)
          ? "model_not_found"
          : /\brefus(?:al|ed)\b/i.test(message)
            ? "refused"
            : status !== undefined
              ? `api_error_${status}`
              : fallback;
  return { type: "error", message, error_type: errorType, retryable };
}
export function nativeError(cause: unknown): AgentSdkWrapperError {
  if (cause instanceof AgentSdkWrapperError) return cause;
  const message = cause instanceof Error ? cause.message : String(cause);
  const data = object(cause);
  if (
    data?.signal ||
    /(?:killed|exited|terminated).*\bSIG[A-Z]+\b/i.test(message)
  )
    return new ProcessTerminatedError(message, { cause });
  if (
    data?.code === "ENOENT" ||
    data?.code === "EACCES" ||
    /Cannot find (?:package|module)|Missing optional dependency|Unsupported platform/i.test(
      message,
    )
  )
    return new RuntimeUnavailableError(message, { cause });
  const error = classify(
    message,
    "runtime_error",
    typeof data?.status === "number" ? data.status : undefined,
  );
  return error.retryable
    ? new TransientError(message, { cause })
    : new ProviderError(message, error.error_type, { cause });
}
