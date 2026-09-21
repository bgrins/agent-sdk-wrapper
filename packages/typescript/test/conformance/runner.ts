import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import {
  Agent,
  type AgentDefaults,
  type AnthropicNativeOptions,
  type CodexNativeOptions,
  type CodexThreadOptions,
  ConfigError,
  collectRun,
  type EventEnvelope,
  ProviderError,
  type RunRequest,
  type RunResult,
  RuntimeUnavailableError,
} from "../../src/index.js";
import { type Recorded, type Step, startMock } from "./mock.js";

// Runs docs/fixtures/conformance-v1.json against the real runtimes; the format
// is in docs/fixtures/CONFORMANCE.md.
type Options = Record<string, unknown>;
type CaseProvider = "anthropic" | "codex";
export type Mode = "offline" | "live";
interface Match {
  request?: number;
  path?: string;
  header?: string;
  equals?: unknown;
  contains?: string;
  excludes?: string;
  absent?: boolean;
}
export interface Expect {
  status?: string;
  error_type?: string;
  final_text?: string;
  final_text_contains?: string;
  events?: { includes?: string[]; excludes?: string[]; count?: number };
  tool_calls?: string[];
  tool_results?: { contains?: string; is_error?: boolean }[];
  structured_output?: unknown;
  requests?: { count?: number; match?: Match[] };
  same_session?: boolean;
  on_event?: boolean;
  trace_file?: boolean;
  on_provider_event?: boolean;
  raw?: string[];
  artifacts?: string[];
  raises?: string;
  setup_error?: string;
  config_error?: boolean;
}
interface LaterRun {
  prompt: string;
  options?: Options;
  agent?: "same" | "new";
  expect?: Expect;
}
interface Setup {
  files?: Record<string, string>;
  codex_login?: "chatgpt" | "api_key";
  codex_provider?: "builtin";
}
export interface Case {
  id: string;
  provider: CaseProvider;
  options?: Options;
  run_options?: Options;
  prompt: string;
  setup?: Setup;
  mock?: Step[];
  expect?: Expect;
  runs?: LaterRun[];
  languages?: { typescript?: string | { options?: Options; expect?: Expect } };
  live?: {
    prompt?: string;
    options?: Options;
    expect?: Expect;
    runs?: Partial<LaterRun>[];
  };
}
/** A run of a resolved case: a new Agent from `agent`, else the previous Agent with per-call `overrides`. */
interface Turn {
  prompt: string;
  agent?: Options;
  overrides: Options;
  expect: Expect;
}
/** A case as TypeScript runs it in one mode. */
export interface Plan {
  id: string;
  provider: CaseProvider;
  setup: Setup;
  mock: Step[];
  turns: Turn[];
}

export const cases: Case[] = JSON.parse(
  readFileSync(
    new URL(
      "../../../../../docs/fixtures/conformance-v1.json",
      import.meta.url,
    ),
    "utf8",
  ),
).cases;

const liveModels: Record<CaseProvider, [env: string, model: string]> = {
  anthropic: ["AGENT_SDK_WRAPPER_TS_ANTHROPIC_MODEL", "claude-haiku-4-5"],
  codex: ["AGENT_SDK_WRAPPER_TS_OPENAI_MODEL", "gpt-5.6-luna"],
};
const keyEnv: Record<CaseProvider, string> = {
  anthropic: "ANTHROPIC_API_KEY",
  codex: "OPENAI_API_KEY",
};

export function liveSkip(provider: CaseProvider): string | false {
  if (process.env.AGENT_SDK_WRAPPER_TS_RUN_INTEGRATION !== "1")
    return "Set AGENT_SDK_WRAPPER_TS_RUN_INTEGRATION=1";
  const name = keyEnv[provider];
  return process.env[name] ? false : `${name} is required`;
}

/** `provider_options` merge by key; other options replace. */
const mergeOptions = (base: Options, extra: Options): Options => ({
  ...base,
  ...extra,
  ...(base.provider_options || extra.provider_options
    ? {
        provider_options: {
          ...(base.provider_options as Options),
          ...(extra.provider_options as Options),
        },
      }
    : {}),
});

/** Apply the TypeScript override and the mode; a string is the reason to skip. */
export function resolveCase(c: Case, mode: Mode): Plan | string {
  const override = c.languages?.typescript;
  if (typeof override === "string") return override;
  const expect = override?.expect ?? c.expect ?? {};
  const live = mode === "live" ? c.live : undefined;
  if (mode === "live" && !live) return "no live section";
  if (live && expect.config_error)
    return "TypeScript rejects the options before any request";
  const [modelEnv, model] = liveModels[c.provider];
  const options = {
    ...(override?.options ?? c.options),
    ...live?.options,
    ...(live
      ? { model: process.env[modelEnv] || live.options?.model || model }
      : {}),
  };
  const turns: Turn[] = [
    {
      prompt: live?.prompt ?? c.prompt,
      agent: options,
      overrides: c.run_options ?? {},
      expect: live ? (live.expect ?? {}) : expect,
    },
  ];
  (c.runs ?? []).forEach((later, index) => {
    const extra = live?.runs?.[index];
    const run = {
      ...later,
      ...extra,
      options: { ...later.options, ...extra?.options },
    };
    turns.push({
      prompt: run.prompt,
      agent:
        run.agent === "new" ? mergeOptions(options, run.options) : undefined,
      overrides: run.agent === "new" ? {} : run.options,
      expect: run.expect ?? {},
    });
  });
  return {
    id: c.id,
    provider: c.provider,
    setup: c.setup ?? {},
    mock: c.mock ?? [{ text: "ok" }],
    turns,
  };
}

// Python option names mapped onto the TypeScript request. Each mapper returns
// the fields it sets: `request` for RunRequest, `anthropic` for Claude's
// native options, `client`/`thread` for Codex's. Options TypeScript reserves
// map to the reserved name, which it rejects; a missing entry fails the case.
type Layer = "request" | "anthropic" | "client" | "thread";
type Fragment = Partial<Record<Layer, Options>>;
interface Context {
  provider: CaseProvider;
  root: string;
  options: Options;
  natives: unknown[];
}
// biome-ignore lint/suspicious/noExplicitAny: option values come from JSON.
type Mapper = (value: any, ctx: Context) => Fragment;
type Entry = Mapper | Partial<Record<CaseProvider, Mapper>>;
class Unmapped extends Error {
  constructor(option: string) {
    super(
      `No TypeScript mapping for option ${option}; add a languages.typescript override`,
    );
  }
}

const request = (key: keyof RunRequest) => (value: unknown) => ({
  request: { [key]: value },
});
const claude = (key: keyof AnthropicNativeOptions) => (value: unknown) => ({
  anthropic: { [key]: value },
});
const client = (key: keyof CodexNativeOptions) => (value: unknown) => ({
  client: { [key]: value },
});
const thread = (key: keyof CodexThreadOptions) => (value: unknown) => ({
  thread: { [key]: value },
});
const webTools = ["WebSearch", "WebFetch"];

export const pythonOptions: Record<string, Entry> = {
  provider: request("provider"),
  model: request("model"),
  effort: request("effort"),
  cwd: (path: string, ctx) => request("cwd")(join(ctx.root, path)),
  session_id: request("sessionId"),
  continue_session: request("continueSession"),
  include_raw: request("includeRaw"),
  cli_login: request("cliLogin"),
  trace_file: (path: string, ctx) => request("traceFile")(join(ctx.root, path)),
  on_provider_event: (_, ctx) =>
    request("onProviderEvent")((event: unknown) => ctx.natives.push(event)),
  // The runner always collects through collectRun's callback.
  on_event: () => ({}),
  timeout: (seconds: number) =>
    request("signal")(AbortSignal.timeout(seconds * 1000)),
  env: { anthropic: claude("env"), codex: client("env") },
  system_prompt: {
    anthropic: claude("systemPrompt"),
    codex: request("systemPrompt"),
  },
  max_turns: { anthropic: claude("maxTurns"), codex: request("maxTurns") },
  permission_mode: {
    anthropic: (mode: string) => ({
      anthropic: {
        permissionMode: mode,
        ...(mode === "bypassPermissions"
          ? { allowDangerouslySkipPermissions: true }
          : {}),
      },
    }),
    codex: request("permissionMode"),
  },
  builtin_tools: {
    anthropic: (tools: string | string[]) =>
      claude("tools")(tools === "none" ? [] : tools),
    codex: request("builtinTools"),
  },
  web_tools: {
    anthropic: (enabled: boolean, ctx) =>
      !enabled
        ? claude("disallowedTools")(webTools)
        : Array.isArray(ctx.options.builtin_tools)
          ? claude("tools")(webTools)
          : {},
    codex: (enabled: boolean) =>
      thread("webSearchMode")(enabled ? "live" : "disabled"),
  },
  allowed_tools: { anthropic: claude("allowedTools") },
  disallowed_tools: { anthropic: claude("disallowedTools") },
  setting_sources: { anthropic: claude("settingSources") },
  tools: request("tools"),
  subagents: request("subagents"),
  mcp_servers: request("mcpServers"),
  output_schema: request("outputSchema"),
  artifacts_dir: request("artifactsDir"),
  "extra_options.thinking": {
    anthropic: ({ budget_tokens, ...thinking }: Options) =>
      claude("thinking")({
        ...thinking,
        ...(budget_tokens === undefined ? {} : { budgetTokens: budget_tokens }),
      }),
  },
  "provider_options.cli_path": {
    anthropic: claude("pathToClaudeCodeExecutable"),
  },
  "provider_options.api_key": { codex: client("apiKey") },
  "provider_options.sandbox": { codex: thread("sandboxMode") },
  "provider_options.approval_mode": {
    codex: (mode: string) => {
      if (mode !== "deny_all")
        throw new Unmapped(`provider_options.approval_mode=${mode}`);
      return thread("approvalPolicy")("never");
    },
  },
  "provider_options.config.env": { codex: client("env") },
};

function merge(target: Fragment, source: Fragment): Fragment {
  for (const [layer, fields] of Object.entries(source) as [Layer, Options][]) {
    const into = target[layer] ?? {};
    target[layer] = into;
    for (const [key, value] of Object.entries(fields)) {
      const prior = into[key];
      into[key] =
        Array.isArray(prior) && Array.isArray(value)
          ? [...prior, ...value]
          : key === "env"
            ? { ...(prior as Options), ...(value as Options) }
            : value;
    }
  }
  return target;
}
const copy = (fragment: Fragment): Fragment =>
  Object.fromEntries(
    Object.entries(fragment).map(([layer, fields]) => [layer, { ...fields }]),
  );

/** Map Python-named options; throws for an option TypeScript lacks. */
export function mapOptions(options: Options, ctx: Context): Fragment {
  const fragment: Fragment = {};
  const visit = (name: string, value: unknown) => {
    const entry = pythonOptions[name];
    if (entry) {
      const mapper = typeof entry === "function" ? entry : entry[ctx.provider];
      if (!mapper) throw new Unmapped(`${name} for ${ctx.provider}`);
      merge(fragment, mapper(value, ctx));
    } else if (
      typeof value === "object" &&
      value !== null &&
      Object.keys(pythonOptions).some((key) => key.startsWith(`${name}.`))
    )
      for (const [key, inner] of Object.entries(value))
        visit(`${name}.${key}`, inner);
    else throw new Unmapped(name);
  };
  for (const [name, value] of Object.entries(options)) visit(name, value);
  return fragment;
}

const layerNames: Record<Layer, string> = {
  request: "request",
  anthropic: "anthropic",
  client: "openai.client",
  thread: "openai.thread",
};
/** Qualified TypeScript option names a plan's options set; empty when an option is unmapped. */
export function exercised(plan: Plan): string[] {
  const keys = new Set(["request.provider", "request.prompt"]);
  try {
    for (const turn of plan.turns)
      for (const options of [turn.agent ?? {}, turn.overrides]) {
        const ctx = { provider: plan.provider, root: "", options, natives: [] };
        const fragment = mapOptions(options, ctx);
        for (const [layer, fields] of Object.entries(fragment) as [
          Layer,
          Options,
        ][]) {
          if (layer !== "request") keys.add("request.providerOptions");
          for (const key of Object.keys(fields))
            keys.add(`${layerNames[layer]}.${key}`);
        }
      }
  } catch (error) {
    if (error instanceof Unmapped) return [];
    throw error;
  }
  return [...keys];
}

const dead = "http://127.0.0.1:9";
/** The parent env without host credentials or runtime settings, rooted in scratch dirs. */
function isolatedEnv(root: string, mode: Mode): Record<string, string> {
  const env: Record<string, string> = {};
  for (const [key, value] of Object.entries(process.env))
    if (
      value !== undefined &&
      !/^(ANTHROPIC_|OPENAI_|CODEX_|CLAUDE_CODE_|XDG_)/.test(key) &&
      key !== "CLAUDECODE"
    )
      env[key] = value;
  Object.assign(env, {
    HOME: join(root, "home"),
    CODEX_HOME: join(root, "codex_home"),
    CLAUDE_CONFIG_DIR: join(root, "claude_config"),
    ANTHROPIC_CONFIG_DIR: join(root, "anthropic_config"),
    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC: "1",
  });
  if (mode === "offline")
    for (const name of ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"]) {
      const value = name === "NO_PROXY" ? "127.0.0.1,localhost" : dead;
      env[name] = value;
      env[name.toLowerCase()] = value;
    }
  return env;
}

/** The mock as a Codex provider with retries off, selected unless the case keeps the built-in one. */
const codexProvider = (url: string, select: boolean) =>
  [
    ...(select ? ['model_provider = "mock"'] : []),
    "[model_providers.mock]",
    'name = "mock"',
    `base_url = "${url}"`,
    'wire_api = "responses"',
    "requires_openai_auth = true",
    "request_max_retries = 0",
    "stream_max_retries = 0",
    "supports_websockets = false",
    "",
  ].join("\n");

/** A stored ChatGPT login Codex accepts offline, or a stored API key. */
function codexLogin(kind: "chatgpt" | "api_key"): string {
  if (kind === "api_key")
    return JSON.stringify({ auth_mode: "apikey", OPENAI_API_KEY: "sk-stored" });
  const part = (value: object) =>
    Buffer.from(JSON.stringify(value)).toString("base64url");
  const claims = {
    email: "a@b.c",
    exp: 4102444800,
    "https://api.openai.com/auth": {
      chatgpt_plan_type: "pro",
      chatgpt_account_id: "acct_1",
      chatgpt_user_id: "user_1",
    },
  };
  const jwt = (extra: object) =>
    `${part({ alg: "RS256" })}.${part({ ...claims, ...extra })}.c2ln`;
  return JSON.stringify({
    OPENAI_API_KEY: null,
    tokens: {
      id_token: jwt({}),
      access_token: jwt({ sub: "access" }),
      refresh_token: "chatgpt-refresh",
      account_id: "acct_1",
    },
    last_refresh: new Date().toISOString(),
  });
}

/** Replace `{session_id}` in every string of an options tree. */
function withSession(value: unknown, session: string): unknown {
  if (typeof value === "string")
    return value.replaceAll("{session_id}", session);
  if (Array.isArray(value))
    return value.map((item) => withSession(item, session));
  if (typeof value === "object" && value !== null)
    return Object.fromEntries(
      Object.entries(value).map(([key, item]) => [
        key,
        withSession(item, session),
      ]),
    );
  return value;
}

/** What one run produced. */
export interface Outcome {
  result?: RunResult;
  thrown?: unknown;
  envelopes: EventEnvelope[];
  natives: unknown[];
  requests: Recorded[];
  traceFile?: string;
}

/** Run every turn of a plan and check its expectations. */
export async function runCase(plan: Plan, mode: Mode): Promise<Outcome[]> {
  const { provider } = plan;
  const root = await mkdtemp(join(tmpdir(), "agent-sdk-wrapper-conformance-"));
  const mock =
    mode === "offline" ? await startMock(provider, plan.mock) : undefined;
  try {
    const dirs = {
      home: "home",
      codex_home: "codex_home",
      claude_config: "claude_config",
    };
    for (const dir of [...Object.values(dirs), "anthropic_config", "work"])
      await mkdir(join(root, dir));
    const env = isolatedEnv(root, mode);
    const key = mode === "live" ? process.env[keyEnv[provider]] : undefined;
    if (provider === "anthropic") {
      env.ANTHROPIC_API_KEY = key ?? "sk-ant-mock";
      if (mock) {
        env.ANTHROPIC_BASE_URL = mock.url;
        env.CLAUDE_CODE_MAX_RETRIES = "0";
      }
    } else {
      env.OPENAI_API_KEY = key ?? "sk-mock";
      if (mock)
        await writeFile(
          join(root, "codex_home", "config.toml"),
          codexProvider(mock.url, plan.setup.codex_provider !== "builtin"),
        );
    }
    const first = plan.turns[0]?.agent ?? {};
    const cwd = typeof first.cwd === "string" ? first.cwd : "work";
    for (const [target, content] of Object.entries(plan.setup.files ?? {})) {
      const [base = "", ...rest] = target.split("/");
      const path = join(
        root,
        base === "cwd"
          ? cwd
          : (dirs[base as keyof typeof dirs] ??
              assert.fail(`unknown setup root ${base}`)),
        ...rest,
      );
      await mkdir(dirname(path), { recursive: true });
      await writeFile(path, content);
    }
    if (plan.setup.codex_login)
      await writeFile(
        join(root, "codex_home", "auth.json"),
        codexLogin(plan.setup.codex_login),
      );
    const infra = (): Fragment => ({
      request: { cwd: join(root, "work") },
      ...(provider === "anthropic"
        ? { anthropic: { env: { ...env } } }
        : { client: { env: { ...env } }, thread: { skipGitRepoCheck: true } }),
    });
    const deadline = AbortSignal.timeout(
      (mode === "live" ? 170_000 : 60_000) * plan.turns.length,
    );
    const defaults = (fragment: Fragment) => {
      const signal = fragment.request?.signal;
      return {
        provider,
        ...fragment.request,
        signal:
          signal instanceof AbortSignal
            ? AbortSignal.any([signal, deadline])
            : deadline,
        providerOptions:
          provider === "anthropic"
            ? { provider: "anthropic", options: fragment.anthropic }
            : {
                provider: "openai",
                client: fragment.client,
                thread: fragment.thread,
              },
      } as AgentDefaults;
    };
    const natives: unknown[] = [];
    let agent: Agent | undefined;
    let base: Fragment = {};
    let session: string | null | undefined;
    const outcomes: Outcome[] = [];
    for (const [index, turn] of plan.turns.entries()) {
      const ctx = (options: Options) => ({ provider, root, options, natives });
      const outcome: Outcome = { envelopes: [], natives: [], requests: [] };
      outcomes.push(outcome);
      const seen = {
        requests: mock?.requests.length ?? 0,
        natives: natives.length,
      };
      const prompt = withSession(turn.prompt, session ?? "") as string;
      try {
        let fragment = base;
        if (turn.agent) {
          const options = withSession(turn.agent, session ?? "") as Options;
          base = fragment = merge(infra(), mapOptions(options, ctx(options)));
          if (typeof fragment.request?.cwd === "string")
            await mkdir(fragment.request.cwd, { recursive: true });
          agent = new Agent(defaults(fragment));
        }
        const overrides = withSession(turn.overrides, session ?? "") as Options;
        if (Object.keys(overrides).length)
          fragment = merge(
            copy(base),
            mapOptions(overrides, ctx({ ...turn.agent, ...overrides })),
          );
        outcome.traceFile = fragment.request?.traceFile as string | undefined;
        assert.ok(agent, "the first run builds the Agent");
        const input =
          fragment === base
            ? prompt
            : ({ ...defaults(fragment), prompt } as RunRequest);
        outcome.result = await collectRun(agent.stream(input), (envelope) => {
          outcome.envelopes.push(envelope);
        });
      } catch (error) {
        if (error instanceof Unmapped || error instanceof assert.AssertionError)
          throw error;
        outcome.thrown = error;
      }
      outcome.requests = mock?.requests.slice(seen.requests) ?? [];
      outcome.natives = natives.slice(seen.natives);
      try {
        await check(turn.expect, outcome, mode, session);
      } catch (error) {
        if (error instanceof Error)
          error.message = `run ${index}: ${error.message}`;
        throw error;
      }
      session = outcome.result?.session_id;
    }
    return outcomes;
  } finally {
    await mock?.close();
    // A cancelled Claude CLI can still be exiting, and writing, when its run returns.
    await rm(root, { recursive: true, force: true, maxRetries: 10 });
  }
}

function at(value: unknown, path: string): unknown {
  let current = value;
  for (const part of path.split(".")) {
    if (Array.isArray(current)) {
      const index = Number(part);
      current = current[index < 0 ? current.length + index : index];
    } else if (typeof current === "object" && current !== null)
      current = (current as Options)[part];
    else return undefined;
  }
  return current;
}

function matches(value: unknown, match: Match): boolean {
  const text = typeof value === "string" ? value : JSON.stringify(value);
  if (match.absent) return value === undefined;
  if (match.excludes !== undefined)
    return value === undefined || !text.includes(match.excludes);
  if (value === undefined) return false;
  if (match.contains !== undefined) return text.includes(match.contains);
  return JSON.stringify(value) === JSON.stringify(match.equals);
}

const errorType = (thrown: unknown) =>
  thrown instanceof ProviderError
    ? thrown.errorType
    : thrown instanceof RuntimeUnavailableError
      ? "runtime_unavailable"
      : String(thrown);
const shape = (envelopes: EventEnvelope[]) =>
  envelopes.map(({ sequence, event }) => [sequence, event.type]);

async function check(
  expect: Expect,
  outcome: Outcome,
  mode: Mode,
  previousSession: string | null | undefined,
): Promise<void> {
  const { result, thrown, requests } = outcome;
  if (expect.config_error) {
    assert.ok(
      thrown instanceof ConfigError,
      `expected ConfigError: ${String(thrown ?? result?.status)}`,
    );
    assert.deepEqual(
      outcome.envelopes,
      [],
      "ConfigError must precede every event",
    );
    return;
  }
  if (expect.setup_error !== undefined) {
    assert.equal(requests.length, 0, "setup errors precede model requests");
    // TypeScript throws setup failures; Python returns failed results.
    if (thrown) return assert.equal(errorType(thrown), expect.setup_error);
    assert.equal(result?.error_type, expect.setup_error, result?.error ?? "");
  }
  if (thrown) throw thrown;
  assert.ok(result);
  const events = result.events.map(({ event }) => event);
  const types: string[] = events.map((event) => event.type);
  const summary = `${result.status} ${result.error_type} ${result.error}\nevents: ${types.join(", ")}`;
  for (const key of ["status", "error_type", "final_text"] as const)
    if (expect[key] !== undefined)
      assert.equal(result[key], expect[key], summary);
  if (expect.final_text_contains !== undefined)
    assert.ok(
      result.final_text.includes(expect.final_text_contains),
      `final_text: ${result.final_text}`,
    );
  for (const type of expect.events?.includes ?? [])
    assert.ok(types.includes(type), `missing ${type}\n${summary}`);
  for (const type of expect.events?.excludes ?? [])
    assert.ok(!types.includes(type), `unexpected ${type}\n${summary}`);
  if (expect.events?.count !== undefined)
    assert.equal(events.length, expect.events.count, summary);
  const calls = events.flatMap((event) =>
    event.type === "tool_call" ? [event.name] : [],
  );
  if (expect.tool_calls !== undefined)
    assert.deepEqual(calls, expect.tool_calls, summary);
  if (expect.tool_results !== undefined) {
    const results = events.flatMap((event) =>
      event.type === "tool_result" ? [event] : [],
    );
    const got = results.map(({ output = "", is_error }, index) => {
      const contains = expect.tool_results?.[index]?.contains ?? "";
      return {
        contains: output.includes(contains) ? contains : output,
        is_error,
      };
    });
    const want = expect.tool_results.map(
      ({ contains = "", is_error = false }) => ({ contains, is_error }),
    );
    assert.deepEqual(got, want, summary);
  }
  if ("structured_output" in expect)
    assert.deepEqual(result.structured_output, expect.structured_output);
  if (expect.same_session) assert.equal(result.session_id, previousSession);
  if (expect.on_event)
    assert.deepEqual(shape(outcome.envelopes), shape(result.events));
  if (expect.trace_file) {
    assert.ok(
      outcome.traceFile,
      "trace_file expectation without a trace_file option",
    );
    const lines = (await readFile(outcome.traceFile, "utf8"))
      .trim()
      .split("\n");
    assert.deepEqual(
      shape(lines.map((line) => JSON.parse(line))),
      shape(result.events),
    );
  }
  if (expect.on_provider_event)
    assert.ok(outcome.natives.length > 0, "no native events");
  for (const type of expect.raw ?? []) {
    const typed = events.filter((event) => event.type === type);
    assert.ok(
      typed.length > 0 && typed.every((event) => "raw" in event && event.raw),
      `${type} with raw\n${summary}`,
    );
  }
  if (mode === "live") return;
  if (expect.requests?.count !== undefined)
    assert.equal(
      requests.length,
      expect.requests.count,
      `model requests\n${summary}`,
    );
  for (const match of expect.requests?.match ?? []) {
    const pool =
      match.request === undefined
        ? requests
        : requests.slice(match.request).slice(0, 1);
    const values = pool.map(({ headers, body }) =>
      match.header !== undefined
        ? headers[match.header]
        : at(body, match.path ?? ""),
    );
    const every =
      match.request === undefined &&
      (match.absent || match.excludes !== undefined);
    const ok =
      values.length > 0 &&
      (every
        ? values.every((value) => matches(value, match))
        : values.some((value) => matches(value, match)));
    assert.ok(
      ok,
      `request match ${JSON.stringify(match)} failed; saw ${JSON.stringify(values).slice(0, 2000)}`,
    );
  }
}

// Fields this runner implements; anything else fails the case instead of passing unchecked.
const fields = {
  case: [
    "id",
    "provider",
    "options",
    "run_options",
    "prompt",
    "setup",
    "mock",
    "expect",
    "runs",
    "languages",
    "live",
  ],
  setup: ["files", "codex_login", "codex_provider"],
  step: [
    "text",
    "thinking",
    "tool",
    "shell",
    "usage",
    "stop_reason",
    "status",
    "headers",
    "body",
    "stream_error",
    "truncate",
    "hang",
  ],
  run: ["prompt", "options", "agent", "expect"],
  live: ["prompt", "options", "expect", "runs"],
  expect: [
    "status",
    "error_type",
    "final_text",
    "final_text_contains",
    "events",
    "tool_calls",
    "tool_results",
    "structured_output",
    "requests",
    "same_session",
    "on_event",
    "trace_file",
    "on_provider_event",
    "raw",
    "setup_error",
    "config_error",
  ],
};
/** Case fields, steps and expectations the TypeScript runner does not implement. */
export function unknownFields(c: Case): string[] {
  const unknown = (kind: keyof typeof fields, value: object | undefined) =>
    Object.keys(value ?? {})
      .filter((key) => !fields[kind].includes(key))
      .map((key) => `${kind}.${key}`);
  const override = c.languages?.typescript;
  const expects = [
    c.expect,
    c.live?.expect,
    typeof override === "object" ? override.expect : undefined,
  ];
  return [
    ...unknown("case", c),
    ...unknown("setup", c.setup),
    ...unknown("live", c.live),
    ...(c.mock ?? []).flatMap((step) => unknown("step", step)),
    ...[...(c.runs ?? []), ...(c.live?.runs ?? [])].flatMap((run) => {
      expects.push(run.expect);
      return unknown("run", run);
    }),
    ...expects.flatMap((expect) => unknown("expect", expect)),
  ];
}
