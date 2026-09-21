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

// Every TypeScript option: null must be set by some case's options, a string
// exempts the key with its reason. The types make this table exhaustive.
type Surface<T> = { [K in keyof Required<T>]: string | null };
const surface = {
  request: {
    prompt: null,
    provider: null,
    model: null,
    effort: null,
    cwd: null,
    sessionId: null,
    continueSession: null,
    includeRaw: null,
    signal: null,
    providerOptions: null,
    onProviderEvent: null,
    traceFile: null,
    cliLogin: null,
    tools: null,
    mcpServers: null,
    outputSchema: null,
    systemPrompt: null,
    subagents: null,
    maxTurns: null,
    timeout: "Python's timeout maps to signal; TypeScript reserves the name",
    artifactsDir: null,
    builtinTools: null,
    permissionMode: null,
  } satisfies Surface<RunRequest>,
  anthropic: {
    permissionMode: null,
    allowDangerouslySkipPermissions: null,
    allowedTools: null,
    disallowedTools: null,
    settingSources: null,
    pathToClaudeCodeExecutable: null,
    maxTurns: null,
    thinking: null,
    tools: null,
    systemPrompt: null,
    env: null,
  } satisfies Surface<AnthropicNativeOptions>,
  "openai.client": {
    apiKey: null,
    baseUrl:
      "Selects the built-in provider, whose websocket transport and retries the mock cannot turn off; cases use a custom provider",
    env: null,
    codexPathOverride:
      "Runtime location; codex-exec.test.ts drives a fake runtime through it",
  } satisfies Surface<CodexNativeOptions>,
  "openai.thread": {
    sandboxMode: null,
    skipGitRepoCheck:
      "codex exec only; the runner sets it because scratch directories are not repositories",
    networkAccessEnabled: null,
    webSearchMode: null,
    approvalPolicy: null,
    additionalDirectories: null,
  } satisfies Surface<CodexThreadOptions>,
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
  const missing: string[] = [];
  const stale: string[] = [];
  for (const [layer, keys] of Object.entries(surface))
    for (const [key, reason] of Object.entries(keys)) {
      const name = `${layer}.${key}`;
      if (reason === null && !used.has(name)) missing.push(name);
      if (reason !== null && used.has(name)) stale.push(name);
    }
  assert.deepEqual({ missing, stale }, { missing: [], stale: [] });
});
