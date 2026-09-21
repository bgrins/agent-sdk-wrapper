import assert from "node:assert/strict";
import { availableParallelism } from "node:os";
import { describe, test } from "node:test";
import type {
  AnthropicNativeOptions,
  CodexNativeOptions,
  CodexThreadOptions,
  RunRequest,
} from "../src/index.js";
import {
  cases,
  exemptions,
  exercised,
  type Plan,
  resolveCase,
  runCase,
  unknownFields,
} from "./conformance/runner.js";

describe("conformance cases, offline", {
  concurrency: Math.max(2, Math.floor(availableParallelism() / 2)),
}, () => {
  for (const c of cases) {
    const plan = resolveCase(c, "offline");
    test(c.id, {
      skip: typeof plan === "string" ? plan : false,
      timeout: 90_000 * (1 + (c.runs?.length ?? 0)),
    }, async () => {
      await runCase(plan as Plan, "offline");
    });
  }
});

test("conformance cases use only fields the TypeScript runner implements", () => {
  assert.deepEqual(
    cases.flatMap((c) => unknownFields(c).map((field) => `${c.id}: ${field}`)),
    [],
  );
});

// Every TypeScript option; the types keep these lists exhaustive. Native
// options are named by their providerOptions field.
type Keys<T> = { [K in keyof Required<T>]: true };
const request = Object.keys({
  prompt: true,
  provider: true,
  model: true,
  effort: true,
  cwd: true,
  sessionId: true,
  continueSession: true,
  includeRaw: true,
  signal: true,
  providerOptions: true,
  onProviderEvent: true,
  traceFile: true,
  cliLogin: true,
  tools: true,
  mcpServers: true,
  outputSchema: true,
  systemPrompt: true,
  subagents: true,
  maxTurns: true,
  timeout: true,
  artifactsDir: true,
  builtinTools: true,
  permissionMode: true,
} satisfies Keys<RunRequest>);
const surface = {
  anthropic: [
    ...request,
    ...Object.keys({
      permissionMode: true,
      allowDangerouslySkipPermissions: true,
      allowedTools: true,
      disallowedTools: true,
      settingSources: true,
      pathToClaudeCodeExecutable: true,
      maxTurns: true,
      thinking: true,
      tools: true,
      systemPrompt: true,
      env: true,
    } satisfies Keys<AnthropicNativeOptions>).map((key) => `options.${key}`),
  ],
  codex: [
    ...request,
    ...Object.keys({
      apiKey: true,
      baseUrl: true,
      env: true,
      codexPathOverride: true,
    } satisfies Keys<CodexNativeOptions>).map((key) => `client.${key}`),
    ...Object.keys({
      sandboxMode: true,
      skipGitRepoCheck: true,
      networkAccessEnabled: true,
      webSearchMode: true,
      approvalPolicy: true,
      additionalDirectories: true,
    } satisfies Keys<CodexThreadOptions>).map((key) => `thread.${key}`),
  ],
};

test("every TypeScript option is exercised by a conformance case or exempted", () => {
  const used = new Set(
    cases.flatMap((c) =>
      (["offline", "live"] as const).flatMap((mode) => {
        const plan = resolveCase(c, mode);
        return typeof plan === "string" ? [] : exercised(plan);
      }),
    ),
  );
  const options = Object.entries(surface).flatMap(([provider, keys]) =>
    keys.map((key) => `${provider}:${key}`),
  );
  assert.deepEqual(
    {
      missing: options.filter((key) => !used.has(key) && !exemptions[key]),
      exemptedButExercised: Object.keys(exemptions).filter((key) =>
        used.has(key),
      ),
      unknownExemptions: Object.keys(exemptions).filter(
        (key) => !options.includes(key),
      ),
    },
    { missing: [], exemptedButExercised: [], unknownExemptions: [] },
  );
});
