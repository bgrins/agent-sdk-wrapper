import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { once } from "node:events";
import fs, {
  chmod,
  mkdir,
  mkdtemp,
  readFile,
  rm,
  symlink,
  utimes,
  writeFile,
} from "node:fs/promises";
import { request } from "node:http";
import { syncBuiltinESMExports } from "node:module";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import vm from "node:vm";
import {
  contentSecurityPolicy,
  createTraceServer,
  MAX_DIRECTORIES,
} from "./trace-viewer.mjs";

test("server discovers new traces and reads updates without exposing files outside the results directory", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "trace-viewer-"));
  const results = join(root, "results");
  await mkdir(results);
  const server = createTraceServer(results);
  t.after(async () => {
    server.closeAllConnections();
    await new Promise((resolve) => server.close(resolve));
    await rm(root, { recursive: true, force: true });
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  const base = `http://127.0.0.1:${server.address().port}`;
  assert.deepEqual(await (await fetch(`${base}/api/runs`)).json(), []);
  await mkdir(join(results, "new job"));
  const trace = join(results, "new job", "trace.jsonl");
  await writeFile(trace, '{"event":{"type":"run_started"}}\n');
  const [run] = await (await fetch(`${base}/api/runs`)).json();
  assert.equal(run.label, "new job");
  assert.equal(run.trace, "/results/new%20job/trace.jsonl");
  assert.equal(run.manifest, null);
  assert.match(await (await fetch(base + run.trace)).text(), /run_started/);
  await writeFile(
    trace,
    '{"event":{"type":"run_finished","status":"success"}}\n',
  );
  const updated = await fetch(base + run.trace);
  assert.equal(updated.headers.get("cache-control"), "no-store");
  assert.match(await updated.text(), /run_finished/);
  const page = await fetch(base);
  assert.match(page.url, /index=\/api\/runs/);
  assert.match(await page.text(), /agent-sdk-wrapper Trace Viewer/);
  await writeFile(join(root, ".env"), "private");
  await writeFile(join(root, "secret"), "private");
  await writeFile(join(results, ".hidden"), "private");
  await symlink(join(root, ".env"), join(results, "secret.jsonl"));
  await symlink(join(results, ".hidden"), join(results, "alias.jsonl"));
  await symlink(root, join(results, "outside"));
  await writeFile(join(results, "script.html"), "<script>bad()</script>");
  // Raw paths: fetch() would normalize dot segments before they reach the server.
  for (const path of [
    "/.env",
    "/results/../.env",
    "/results/%2e%2e/.env",
    "/results/%2e%2e%2f.env",
    "/results/.hidden",
    "/results/alias.jsonl",
    "/results/secret.jsonl",
    "/results/outside/.env",
    `/results/${await fs.realpath(join(root, "secret"))}`,
    `/results/${encodeURIComponent(await fs.realpath(join(root, "secret")))}`,
  ]) {
    const response = await get(base, path);
    assert.ok([403, 404].includes(response.status), path);
    assert.ok(!response.body.includes("private"), path);
  }
  assert.equal((await get(base, "/results/a%00b")).status, 400);
  const artifact = await get(base, "/results/script.html");
  assert.equal(artifact.headers["content-type"], "text/plain; charset=utf-8");
  assert.equal(artifact.headers["x-content-type-options"], "nosniff");
  assert.equal(
    (await fetch(`${base}/api/runs`, { method: "POST" })).status,
    405,
  );
  for (const host of [
    "untrusted.test",
    "localhost.untrusted.test",
    "127.0.0.1.untrusted.test",
    "untrusted.localhost",
  ])
    assert.equal((await get(base, "/api/runs", { host })).status, 403, host);
  assert.equal(
    (await get(base, "/api/runs", { host: "localhost:8765" })).status,
    200,
  );

  await mkdir(join(results, ".cache"));
  await writeFile(join(results, ".cache", "trace.jsonl"), "{}\n");
  await mkdir(join(results, "custom trace", "logs"), { recursive: true });
  await writeFile(
    join(results, "custom trace", "manifest.json"),
    JSON.stringify({ files: { trace: "logs/events.jsonl" } }),
  );
  await writeFile(
    join(results, "custom trace", "logs", "events.jsonl"),
    '{"event":{"type":"text","text":"custom trace"}}\n',
  );
  const discovered = await (await fetch(`${base}/api/runs`)).json();
  assert.equal(discovered.length, 2);
  const custom = discovered.find((entry) => entry.label === "custom trace");
  assert.equal(custom.trace, null);
  assert.equal(custom.manifest, "/results/custom%20trace/manifest.json");
  const manifest = await (await fetch(base + custom.manifest)).json();
  const customTrace = await fetch(
    new URL(manifest.files.trace, base + custom.manifest),
  );
  assert.match(await customTrace.text(), /custom trace/);
  const html = await readFile(
    new URL("../docs/trace-viewer.html", import.meta.url),
    "utf8",
  );
  const context = viewerContext(html.match(/<script>([\s\S]*?)<\/script>/)[1]);
  context.window.location = { href: base, search: "", protocol: "http:" };
  context.URLSearchParams = URLSearchParams;
  context.fetch = fetch;
  const runs = await vm.runInContext("findResultRuns()", context);
  context.run = runs.find((entry) => entry.label === "custom trace");
  await vm.runInContext("loadRunFromUrl(run)", context);
  assert.equal(vm.runInContext("loadErrorEl.textContent", context), "");
  assert.match(vm.runInContext("rawTraceText", context), /custom trace/);

  const canonicalTrace = await fs.realpath(trace);
  for (const [replace, status] of [
    [() => symlink(join(root, ".env"), trace), 403],
    [() => execFileSync("mkfifo", [trace]), 413],
    [() => writeFile(trace, Buffer.alloc(16 * 1024 * 1024 + 1)), 413],
  ]) {
    await rm(trace);
    await writeFile(trace, "trace fixture");
    const realpath = fs.realpath;
    let replaced = false;
    const mocked = t.mock.method(fs, "realpath", async (path, ...args) => {
      const resolved = await realpath(path, ...args);
      if (path === canonicalTrace && !replaced) {
        replaced = true;
        await rm(trace);
        await replace();
      }
      return resolved;
    });
    syncBuiltinESMExports();
    try {
      const response = await fetch(base + run.trace, {
        signal: AbortSignal.timeout(2000),
      });
      assert.ok(replaced, "Replace the file after path resolution");
      assert.equal(response.status, status);
      assert.ok(!(await response.text()).includes("private"));
    } finally {
      mocked.mock.restore();
      syncBuiltinESMExports();
    }
  }
});

function get(base, path, headers = {}) {
  return new Promise((resolve, reject) => {
    const req = request(`${base}/`, { path, headers }, (res) => {
      let body = "";
      res.setEncoding("utf8");
      res.on("data", (chunk) => {
        body += chunk;
      });
      res.on("end", () =>
        resolve({ status: res.statusCode, headers: res.headers, body }),
      );
    });
    req.on("error", reject);
    req.end();
  });
}

async function listen(t, directory, options) {
  const server = createTraceServer(directory, options);
  t.after(async () => {
    server.closeAllConnections();
    await new Promise((resolve) => server.close(resolve));
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  return `http://127.0.0.1:${server.address().port}`;
}

test("the viewer page gets a CSP that allows exactly its inline script and style", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "csp-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const base = await listen(t, root);
  const html = await readFile(
    new URL("../docs/trace-viewer.html", import.meta.url),
    "utf8",
  );
  const response = await get(base, "/docs/trace-viewer.html");
  const policy = response.headers["content-security-policy"];
  const hash = (text) =>
    `'sha256-${createHash("sha256").update(text).digest("base64")}'`;
  assert.equal(response.body, html);
  assert.match(policy, /^default-src 'none';/);
  for (const directive of [
    `script-src ${hash(html.match(/<script>([\s\S]*?)<\/script>/)[1])}`,
    `style-src ${hash(html.match(/<style>([\s\S]*?)<\/style>/)[1])}`,
    "connect-src 'self'",
    "base-uri 'none'",
    "form-action 'none'",
    "frame-ancestors 'none'",
  ])
    assert.ok(policy.split("; ").includes(directive), directive);
  // The policy blocks inline handlers and style attributes.
  assert.doesNotMatch(html, /\son[a-z]+=|\sstyle=|setAttribute\("style"/);
  assert.equal(
    contentSecurityPolicy("<style>a\r\nb</style><script>c\rd</script>"),
    contentSecurityPolicy("<style>a\nb</style><script>c\nd</script>"),
  );
  const artifactPolicy = (await get(base, "/api/runs")).headers[
    "content-security-policy"
  ];
  assert.match(artifactPolicy, /default-src 'none'/);
  assert.match(artifactPolicy, /sandbox/);
});

test("run discovery skips unreadable directories and keeps the newest runs when capped", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "run-discovery-"));
  const locked = join(root, "locked");
  t.after(async () => {
    await chmod(locked, 0o700);
    await rm(root, { recursive: true, force: true });
  });
  const base = await listen(t, root);
  const count = MAX_DIRECTORIES + 100;
  const stamp = (index) => new Date(Date.UTC(2026, 0, 1) + index * 60_000);
  await Promise.all(
    Array.from({ length: count }, async (_, index) => {
      // Name order differs from age order, so directory order cannot pass.
      const age = (index * 7919) % count;
      const dir = join(root, `run-${String(index).padStart(4, "0")}`);
      await mkdir(dir);
      await writeFile(join(dir, "trace.jsonl"), "{}\n");
      await utimes(join(dir, "trace.jsonl"), stamp(age), stamp(age));
      await utimes(dir, stamp(age), stamp(age));
    }),
  );
  await mkdir(join(locked, "hidden"), { recursive: true });
  await chmod(locked, 0);
  const response = await get(base, "/api/runs");
  assert.equal(response.status, 200);
  assert.equal(response.headers["x-runs-truncated"], String(MAX_DIRECTORIES));
  const runs = JSON.parse(response.body);
  const newest = Array.from({ length: count }, (_, age) =>
    stamp(age).toISOString(),
  )
    .reverse()
    .slice(0, runs.length);
  // The root and the locked directories use part of the directory budget.
  assert.ok(runs.length >= MAX_DIRECTORIES - 3);
  assert.deepEqual(
    runs.map((run) => run.updated_at),
    newest,
  );
});

test("a symlinked ancestor swapped in after path checks is not followed", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "ancestor-swap-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const results = join(root, "results");
  await mkdir(join(results, "job", "logs"), { recursive: true });
  await mkdir(join(root, "elsewhere"));
  await writeFile(join(results, "job", "logs", "trace.jsonl"), "{}\n");
  await writeFile(join(root, "elsewhere", "trace.jsonl"), "private\n");
  const base = await listen(t, results);
  const canonical = await fs.realpath(
    join(results, "job", "logs", "trace.jsonl"),
  );
  const realpath = fs.realpath;
  let swapped = false;
  const mocked = t.mock.method(fs, "realpath", async (path, ...args) => {
    const resolved = await realpath(path, ...args);
    if (path === canonical && !swapped) {
      swapped = true;
      await fs.rename(join(results, "job", "logs"), join(root, "moved"));
      await symlink(join(root, "elsewhere"), join(results, "job", "logs"));
    }
    return resolved;
  });
  syncBuiltinESMExports();
  try {
    const response = await get(base, "/results/job/logs/trace.jsonl");
    assert.ok(swapped, "Swap the ancestor after path resolution");
    assert.equal(response.status, 403);
    assert.ok(!response.body.includes("private"));
  } finally {
    mocked.mock.restore();
    syncBuiltinESMExports();
  }
});

function viewerContext(script) {
  const element = () => ({
    addEventListener() {},
    replaceChildren() {},
    append() {},
    setAttribute() {},
    dataset: {},
    classList: { toggle() {} },
    querySelectorAll: () => [],
  });
  const context = vm.createContext({
    document: {
      querySelector: element,
      querySelectorAll: () => [],
      createElement: element,
      body: element(),
    },
    window: { location: { protocol: "file:" } },
    localStorage: { getItem() {} },
    setInterval() {},
    URL,
    Blob,
  });
  vm.runInContext(script, context);
  return context;
}

test("trace-only metadata follows the latest resumed run and sums usage across turns", async () => {
  const html = await readFile(
    new URL("../docs/trace-viewer.html", import.meta.url),
    "utf8",
  );
  const context = viewerContext(html.match(/<script>([\s\S]*?)<\/script>/)[1]);
  context.trace = [
    {
      run_id: "one",
      event: { type: "run_started", provider: "openai", model: "test-model" },
    },
    {
      run_id: "one",
      event: { type: "usage", usage: { input_tokens: 10 }, cost_usd: 0.01 },
    },
    {
      run_id: "one",
      event: { type: "run_finished", status: "success", duration_ms: 200 },
    },
    {
      run_id: "two",
      event: { type: "run_started", provider: "openai", model: "test-model" },
    },
    { run_id: "two", event: { type: "session_info", id: "session" } },
  ];
  const metadata = () =>
    JSON.parse(
      vm.runInContext("JSON.stringify(traceMetadata(trace))", context),
    );
  assert.equal(metadata().status, "running");
  assert.equal(metadata().provider, "openai");
  assert.equal(metadata().model, "test-model");
  assert.equal(metadata().session_id, "session");
  context.trace.push(
    {
      run_id: "two",
      event: { type: "usage", usage: { input_tokens: 5 }, cost_usd: 0.02 },
    },
    {
      run_id: "two",
      event: { type: "run_finished", status: "success", duration_ms: 100 },
    },
  );
  assert.equal(metadata().status, "success");
  assert.equal(metadata().run_id, "two");
  assert.equal(metadata().duration_ms, 300);
  const stats = JSON.parse(
    vm.runInContext("JSON.stringify(traceStats(trace, null))", context),
  );
  assert.equal(stats.usage.input_tokens, 15);
  assert.equal(stats.costLabel, "$0.0300");
  context.trace.at(-1).event.status = "failure";
  assert.equal(metadata().status, "failure");
});

test("polling restores an unchanged trace after a failed manual refresh", async () => {
  const html = await readFile(
    new URL("../docs/trace-viewer.html", import.meta.url),
    "utf8",
  );
  const context = viewerContext(html.match(/<script>([\s\S]*?)<\/script>/)[1]);
  const trace = `${JSON.stringify({
    run_id: "one",
    event: { type: "run_started", provider: "openai" },
  })}\n`;
  let unavailable = false;
  context.fetch = async () =>
    unavailable
      ? new Response("Unavailable", { status: 503 })
      : new Response(trace);
  context.run = {
    runUrl: new URL("http://localhost/results/run/trace.jsonl"),
    traceRel: "trace.jsonl",
  };
  const visibleTrace = () => vm.runInContext("rawTraceText", context);
  const error = () => vm.runInContext("loadErrorEl.textContent", context);
  const poll = () =>
    vm.runInContext("loadRunFromUrl(run, { refresh: true })", context);

  await vm.runInContext("loadRunFromUrl(run)", context);
  assert.equal(error(), "");
  assert.equal(visibleTrace(), trace);

  unavailable = true;
  await vm.runInContext("loadRunFromUrl(run)", context);
  assert.match(error(), /503/);
  assert.equal(visibleTrace(), "");

  unavailable = false;
  await poll();
  assert.equal(error(), "");
  assert.equal(visibleTrace(), trace);

  unavailable = true;
  await poll();
  assert.match(error(), /503/);
  assert.equal(visibleTrace(), trace);

  unavailable = false;
  await poll();
  assert.equal(error(), "");
  assert.equal(visibleTrace(), trace);
});

test("mounted job traces update directly without reading agent-created directories or links", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "mounted-traces-"));
  const jobs = join(root, "jobs");
  const job = join(jobs, "job");
  await mkdir(join(job, "nested"), { recursive: true });
  const server = createTraceServer(jobs, { depth: 1 });
  t.after(async () => {
    server.closeAllConnections();
    await new Promise((resolve) => server.close(resolve));
    await rm(root, { recursive: true, force: true });
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  const base = `http://127.0.0.1:${server.address().port}`;
  const started = '{"event":{"type":"run_started"}}\n';
  const finished = '{"event":{"type":"run_finished"}}\n';
  await writeFile(
    join(job, "0001.trace.jsonl"),
    started + finished.slice(0, 10),
  );
  await writeFile(join(job, "0002.trace.jsonl"), started);
  await writeFile(join(job, "nested", "private.trace.jsonl"), "private\n");
  await writeFile(join(root, "secret"), "private\n");
  await symlink(join(root, "secret"), join(job, "link.trace.jsonl"));
  await fs.link(join(root, "secret"), join(job, "hardlink.trace.jsonl"));
  execFileSync("mkfifo", [join(job, "fifo.trace.jsonl")]);
  const runs = await (await fetch(`${base}/api/runs`)).json();
  assert.deepEqual(runs.map((run) => run.label).sort(), [
    "job/0001.trace.jsonl",
    "job/0002.trace.jsonl",
  ]);
  const url = `${base}/results/job/0001.trace.jsonl`;
  assert.equal(await (await fetch(url)).text(), started);
  await fs.appendFile(join(job, "0001.trace.jsonl"), finished.slice(10));
  assert.equal(await (await fetch(url)).text(), started + finished);
  for (const [path, status] of [
    ["nested/private.trace.jsonl", 404],
    ["link.trace.jsonl", 403],
    ["hardlink.trace.jsonl", 413],
    ["fifo.trace.jsonl", 413],
  ]) {
    const response = await fetch(`${base}/results/job/${path}`, {
      signal: AbortSignal.timeout(2000),
    });
    assert.equal(response.status, status);
    assert.ok(!(await response.text()).includes("private"));
  }
});

test("an untrusted manifest cannot fetch outside its run directory", async () => {
  const html = await readFile(
    new URL("../docs/trace-viewer.html", import.meta.url),
    "utf8",
  );
  const context = viewerContext(html.match(/<script>([\s\S]*?)<\/script>/)[1]);
  const manifest = "http://localhost/results/job/manifest.json";
  context.run = { runUrl: new URL(manifest), manifestUrl: new URL(manifest) };
  for (const trace of [
    "http://other.test/secret",
    "../other-job/trace.jsonl",
    "javascript:alert(1)",
  ]) {
    const requested = [];
    context.fetch = async (url) => {
      requested.push(url.href);
      return new Response(JSON.stringify({ files: { trace } }));
    };
    await vm.runInContext("loadRunFromUrl(run)", context);
    assert.deepEqual(requested, [manifest]);
    assert.match(
      vm.runInContext("loadErrorEl.textContent", context),
      /inside the run directory/,
    );
  }
});
