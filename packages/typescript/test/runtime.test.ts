import assert from "node:assert/strict";
import {
  chmod,
  cp,
  mkdir,
  mkdtemp,
  rm,
  symlink,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { pathToFileURL } from "node:url";
import { Agent, RuntimeUnavailableError } from "../src/index.js";

test("Claude resolves its runtime through the SDK's symlinked dependency scope", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "agent-sdk-wrapper-nested-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const modules = join(root, "node_modules");
  const sdkScope = join(
    modules,
    ".pnpm",
    "sdk",
    "node_modules",
    "@anthropic-ai",
  );
  const sdk = join(sdkScope, "claude-agent-sdk");
  const platform = `claude-agent-sdk-${process.platform}-${process.arch}`;
  const runtime = join(sdkScope, platform);
  const wrapper = join(modules, "agent-sdk-wrapper");
  await mkdir(sdk, { recursive: true });
  await mkdir(runtime, { recursive: true });
  await mkdir(join(modules, "@anthropic-ai"), { recursive: true });
  await cp(new URL("../src/", import.meta.url), wrapper, { recursive: true });
  await writeFile(
    join(wrapper, "package.json"),
    JSON.stringify({ type: "module" }),
  );
  await writeFile(
    join(sdk, "package.json"),
    JSON.stringify({
      name: "@anthropic-ai/claude-agent-sdk",
      type: "module",
      exports: { ".": { import: "./sdk.mjs" } },
    }),
  );
  await writeFile(
    join(sdk, "sdk.mjs"),
    'export function query() { throw new Error("Must not launch a query"); }',
  );
  await writeFile(
    join(runtime, "package.json"),
    JSON.stringify({ name: `@anthropic-ai/${platform}` }),
  );
  const binary = join(
    runtime,
    process.platform === "win32" ? "claude.exe" : "claude",
  );
  await writeFile(binary, "Must not execute this file", { mode: 0o755 });
  await symlink(
    sdk,
    join(modules, "@anthropic-ai", "claude-agent-sdk"),
    "junction",
  );
  // Only the SDK sees the platform package; the wrapper cannot resolve it directly.
  const { Agent: InstalledAgent } = await import(
    pathToFileURL(join(wrapper, "index.js")).href
  );
  const agent = new InstalledAgent({ provider: "anthropic" });
  await assert.doesNotReject(() => agent.checkRuntime());
  await rm(binary);
  await assert.rejects(() => agent.checkRuntime(), {
    name: "RuntimeUnavailableError",
  });
});

test("Claude accepts readable interpreter entrypoints without executable permissions", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "agent-sdk-wrapper-scripts-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  for (const extension of [".js", ".mjs", ".tsx", ".ts", ".jsx"]) {
    const path = join(root, `claude${extension}`);
    await writeFile(
      path,
      'throw new Error("Must not execute during checkRuntime");',
    );
    await chmod(path, 0o600);
    const agent = new Agent({
      provider: "anthropic",
      providerOptions: {
        provider: "anthropic",
        options: { pathToClaudeCodeExecutable: path },
      },
    });
    await assert.doesNotReject(() => agent.checkRuntime());
    await rm(path);
    await assert.rejects(() => agent.checkRuntime(), RuntimeUnavailableError);
  }
});

test("Claude native overrides still require executable permissions", {
  skip: process.platform === "win32",
}, async (t) => {
  const root = await mkdtemp(join(tmpdir(), "agent-sdk-wrapper-binary-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const path = join(root, "claude");
  await writeFile(path, "Must not execute this file");
  await chmod(path, 0o600);
  const agent = new Agent({
    provider: "anthropic",
    providerOptions: {
      provider: "anthropic",
      options: { pathToClaudeCodeExecutable: path },
    },
  });
  await assert.rejects(() => agent.checkRuntime(), RuntimeUnavailableError);
  await chmod(path, 0o700);
  await assert.doesNotReject(() => agent.checkRuntime());
});
