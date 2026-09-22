import { constants } from "node:fs";
import { access } from "node:fs/promises";
import {
  AgentSdkWrapperError,
  ConfigError,
  ProcessTerminatedError,
  ProviderError,
  RuntimeUnavailableError,
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
// Mirrors Python's agent_sdk_wrapper/classify.py; docs/fixtures/error-classification-v1.json
// holds the cases both must agree on. Status codes only count next to an HTTP marker.
const statusPatterns = [
  /\b(?:status(?: code)?|HTTP(?: status)?|API Error)\s*:?\s*(\d{3})\b/i,
  /\b(\d{3}) (?:Bad Request|Unauthorized|Payment Required|Forbidden|Not Found|Too Many Requests|Internal Server Error|Bad Gateway|Service Unavailable|Gateway Timeout)\b/i,
];
// "upgrade to Plus" is Codex's text for a ChatGPT plan without Codex access.
const usageLimit =
  /\busage limits?\b|\bquota exceeded\b|\binsufficient_quota\b|\bexceeded your current quota\b|\bupgrade to (?:Plus|Pro)\b/i;
const contextWindow =
  /\bprompt is too long\b|\bcontext[_ ]length[_ ]exceeded\b|\bcontext[ _-]?window\b|\bmaximum context length\b/i;
const billing = /\bcredit balance\b|\bbilling\b/i;
const authentication =
  /\bunauthorized\b|\bauthentication(?:_error)?\b|\binvalid[_ ](?:x-)?api[_ -]?key\b|\bincorrect api key\b|\bnot logged in\b|\bmissing api key\b/i;
const permission = /\bforbidden\b|\bpermission denied\b|\bpermission_error\b/i;
// "Model provider `x` not found" is a configuration error, not a missing model.
const modelNotFound =
  /\bmodel_not_found\b|\bunknown model\b|\bmodel\b(?! provider).{0,80}?\b(?:not found|does not exist|is not supported)\b/i;
const invalidRequest =
  /\binvalid_request_error\b|\binvalid prompt\b|\bbad request\b/i;
const transient =
  /\brate[ _-]?limit|\boverloaded(?:_error)?\b|\bhigh (?:demand|load)\b|\btemporarily unavailable\b|\bat capacity\b|\bserver (?:is )?busy\b|\bstream disconnected\b|\b(?:connection|request) timed out\b|\bconnection (?:refused|reset|error)\b|\bconnection closed before message completed\b|\bConnectionRefused\b|\bECONNRESET\b|\bECONNREFUSED\b|\bETIMEDOUT\b/i;
/**
 * Quota, context and billing text outrank the status, since those arrive as 400 or
 * 429. Otherwise the status decides, then the text.
 */
function errorType(message: string, status?: number): string | undefined {
  const code =
    status ??
    statusPatterns
      .map((pattern) => pattern.exec(message)?.[1])
      .map(Number)
      .find(Number.isInteger);
  if (usageLimit.test(message)) return "usage_limit_exceeded";
  if (contextWindow.test(message)) return "context_window_exceeded";
  if (billing.test(message)) return "billing_error";
  if (code !== undefined) {
    if (code === 408 || code === 409 || code === 429 || code >= 500)
      return "transient_api_error";
    if (code === 401) return "authentication_failed";
    if (code === 402) return "billing_error";
    if (code === 403) return "permission_denied";
  }
  if (modelNotFound.test(message)) return "model_not_found";
  if (authentication.test(message)) return "authentication_failed";
  if (code !== undefined)
    return code === 400 || code === 422
      ? "invalid_request"
      : `api_error_${code}`;
  if (permission.test(message)) return "permission_denied";
  if (invalidRequest.test(message)) return "invalid_request";
  if (transient.test(message)) return "transient_api_error";
  return undefined;
}
export function classify(
  message: string,
  fallback: string,
  status?: number,
): ErrorEvent {
  return {
    type: "error",
    message,
    error_type: errorType(message, status) ?? fallback,
  };
}
export function nativeError(cause: unknown): AgentSdkWrapperError {
  if (cause instanceof AgentSdkWrapperError) return cause;
  const message = cause instanceof Error ? cause.message : String(cause);
  const data = object(cause);
  const exit = /\bexited with (?:exit )?code (-?\d+)\b/i.exec(message)?.[1];
  const code = exit === undefined ? undefined : Number(exit);
  // Shells report a signal exit as 128 + signal; some runtimes report -signal. With an
  // exit code, a signal name in the text belongs to something else, such as a model command.
  const signaled =
    code === undefined
      ? // Signal names are upper case; /i would match words like "sign" and "signal".
        /\b(?:[Kk]illed|[Ee]xited|[Tt]erminated)\b.*\bSIG[A-Z]{2,}\b/.test(
          message,
        )
      : (code >= 129 && code <= 159) || code < 0;
  if (data?.signal || signaled)
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
  return new ProviderError(message, error.error_type, { cause });
}
