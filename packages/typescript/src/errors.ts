export class AgentSdkWrapperError extends Error {
  readonly retryable: boolean = false;
  constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = new.target.name;
  }
}
export class ConfigError extends AgentSdkWrapperError {}
export class RuntimeUnavailableError extends AgentSdkWrapperError {}
export { RuntimeUnavailableError as ProviderNotAvailableError };
export class TransientError extends AgentSdkWrapperError {
  override readonly retryable = true;
}
export class ProcessTerminatedError extends AgentSdkWrapperError {}
export class ProviderProtocolError extends AgentSdkWrapperError {}
export class ProviderError extends AgentSdkWrapperError {
  constructor(
    message: string,
    readonly errorType: string,
    options?: ErrorOptions,
  ) {
    super(message, options);
  }
}
