import assert from "node:assert/strict";
import { availableParallelism } from "node:os";
import { describe, test } from "node:test";
import type {
  AnthropicNativeOptions,
  CodexNativeOptions,
  CodexThreadOptions,
  RunRequest,
} from "../src/index.js";
import { exhausted, repeats, startMock } from "./conformance/mock.js";
import {
  cases,
  checkMatch,
  exemptions,
  exercised,
  type Match,
  type Plan,
  resolveCase,
  runCase,
  spec,
  specErrors,
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

test("the conformance spec matches its schema", () => {
  assert.deepEqual(specErrors(), []);
  const ids = cases.map(({ id }) => id);
  assert.equal(new Set(ids).size, ids.length, "duplicate case ids");
});

test("the schema rejects unknown expectations in skipped cases", () => {
  for (const typo of [
    { final_txt: "ok" },
    { request: { count: 1 } },
    { requests: { match: [{ path: "model", equal: "x" }] } },
    { setup_error: "authentication_failed", status: "failure" },
  ]) {
    const probe = structuredClone(spec);
    const [c] = probe.cases;
    assert.ok(c);
    c.languages = {
      python: "unsupported: probe",
      typescript: "unsupported: probe",
    };
    c.expect = { ...c.expect, ...typo };
    assert.notDeepEqual(specErrors(probe), [], JSON.stringify(typo));
  }
});

test("requests.match semantics", () => {
  const request = {
    headers: { "x-api-key": "sk" },
    body: {
      model: "m",
      stream: true,
      thinking: { type: "enabled", budget_tokens: 2 },
    },
  };
  const table: [Match, (typeof request)[], boolean][] = [
    [
      { path: "thinking", equals: { budget_tokens: 2, type: "enabled" } },
      [request],
      true,
    ],
    [{ path: "stream", equals: 1 }, [request], false],
    [{ request: -1, path: "model", equals: "m" }, [request], true],
    [{ request: -2, path: "model", equals: "m" }, [request], false],
    [{ request: 1, path: "model", excludes: "x" }, [request], false],
    [{ path: "model", excludes: "m" }, [], true],
    [{ header: "x-api-key", absent: true }, [], true],
    [{ path: "model", contains: "m" }, [], false],
  ];
  for (const [match, requests, passes] of table)
    if (passes) checkMatch(match, requests);
    else
      assert.throws(() => checkMatch(match, requests), JSON.stringify(match));
});

const post = (url: string) =>
  fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ model: "m", stream: true }),
  });

test("the mock fails once the last step has repeated", async () => {
  const mock = await startMock("codex", [{ text: "a" }, { text: "b" }]);
  try {
    const statuses = [];
    for (let n = 0; n <= 2 + repeats; n++) {
      const response = await post(`${mock.url}/responses`);
      await response.text();
      statuses.push(response.status);
    }
    assert.deepEqual(statuses, [...Array(2 + repeats).fill(200), 400]);
    const response = await post(`${mock.url}/responses`);
    assert.match(await response.text(), new RegExp(exhausted));
  } finally {
    await mock.close();
  }
});

test("the mock's truncate drops a chunked stream", async () => {
  const mock = await startMock("anthropic", [{ truncate: true }]);
  try {
    const response = await post(`${mock.url}/v1/messages`);
    assert.equal(response.headers.get("transfer-encoding"), "chunked");
    await assert.rejects(response.text());
  } finally {
    await mock.close();
  }
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
      config: true,
    } satisfies Keys<CodexNativeOptions>).map((key) => `client.${key}`),
    ...Object.keys({
      sandboxMode: true,
      skipGitRepoCheck: true,
      networkAccessEnabled: true,
      webSearchMode: true,
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
