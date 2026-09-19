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
// Status codes only count next to an HTTP marker, never as bare numbers.
const statusPatterns = [
  /\b(?:status(?: code)?|HTTP(?: status)?|API Error)\s*:?\s*(\d{3})\b/i,
  /\b(\d{3}) (?:Bad Request|Unauthorized|Forbidden|Not Found|Too Many Requests|Internal Server Error|Bad Gateway|Service Unavailable|Gateway Timeout)\b/i,
];
const usageLimit =
  /\busage limit\b|\bquota exceeded\b|\binsufficient_quota\b|\bexceeded your current quota\b/i;
const contextWindow =
  /\bprompt is too long\b|\bcontext[_ ]length[_ ]exceeded\b|\bexceeds the context window\b|\bran out of room in the model.s context window\b|\bcontext window exceeded\b/i;
const billing = /\bcredit balance\b|\bbilling\b/i;
const transient =
  /\brate[_ ]?limit|\boverloaded\b|\btemporarily unavailable\b|\bat capacity\b|\bserver (?:is )?busy\b|\bstream disconnected\b|\b(?:connection|request) timed out\b|\bconnection (?:refused|reset)\b|\bConnectionRefused\b|\bECONNRESET\b|\bECONNREFUSED\b|\bETIMEDOUT\b/i;
const authentication =
  /\bunauthorized\b|\bauthentication\b|\binvalid[_ ](?:x-)?api[_ -]?key\b|\bnot logged in\b|\bmissing api key\b/i;
const permission = /\bforbidden\b|\bpermission denied\b/i;
const modelNotFound =
  /\bmodel_not_found\b|\bmodel\b.*\b(?:not found|does not exist)\b|\bunknown model\b/i;
export function classify(
  message: string,
  fallback: string,
  status?: number,
): ErrorEvent {
  const code =
    status ??
    statusPatterns
      .map((pattern) => pattern.exec(message)?.[1])
      .map(Number)
      .find(Number.isInteger);
  const errorType = usageLimit.test(message)
    ? "usage_limit_exceeded"
    : contextWindow.test(message)
      ? "context_window_exceeded"
      : billing.test(message)
        ? "billing_error"
        : code === 429 || (code !== undefined && code >= 500)
          ? "transient_api_error"
          : code === 401
            ? "authentication_failed"
            : code === 403
              ? "permission_denied"
              : modelNotFound.test(message)
                ? "model_not_found"
                : code === 400
                  ? "invalid_request"
                  : code !== undefined
                    ? `api_error_${code}`
                    : transient.test(message)
                      ? "transient_api_error"
                      : authentication.test(message)
                        ? "authentication_failed"
                        : permission.test(message)
                          ? "permission_denied"
                          : /\binvalid_request_error\b/.test(message)
                            ? "invalid_request"
                            : fallback;
  return {
    type: "error",
    message,
    error_type: errorType,
    retryable: errorType === "transient_api_error",
  };
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
    "provider_exception",
    typeof data?.status === "number" ? data.status : undefined,
  );
  return error.retryable
    ? new TransientError(message, { cause })
    : new ProviderError(message, error.error_type, { cause });
}
