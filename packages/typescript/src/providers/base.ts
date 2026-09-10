import type { Provider, ProviderEvent } from "../events.js";
import type { ResolvedRequest } from "../request.js";
export interface ProviderContext {
  /** Report every native frame, even frames with no normalized equivalent. */
  onNativeEvent(event: unknown): void;
}
export interface ProviderAdapter {
  readonly name: Provider;
  validateRequest(request: ResolvedRequest): void;
  ensureAvailable(request: ResolvedRequest): Promise<void>;
  /** Must throw if the stream ends without a native terminal result. */
  stream(
    request: ResolvedRequest,
    context: ProviderContext,
  ): AsyncIterable<ProviderEvent>;
}
