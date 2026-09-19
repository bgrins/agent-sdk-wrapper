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
  /**
   * Return right after the terminal frame: a normal return means the run
   * completed, and later aborts are ignored. Must throw if the native stream
   * ends without a terminal result.
   */
  stream(
    request: ResolvedRequest,
    context: ProviderContext,
  ): AsyncIterable<ProviderEvent>;
}
