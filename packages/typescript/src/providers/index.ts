import type { Provider } from "../events.js";
import { AnthropicAdapter } from "./anthropic.js";
import type { ProviderAdapter } from "./base.js";
import { CodexAdapter } from "./codex.js";
export function buildProvider(provider: Provider): ProviderAdapter {
  return provider === "anthropic" ? new AnthropicAdapter() : new CodexAdapter();
}
