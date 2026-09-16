import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { once } from "node:events";
import fs, {
  mkdir,
  mkdtemp,
  readFile,
  rm,
  symlink,
  writeFile,
} from "node:fs/promises";
import { request } from "node:http";
import { syncBuiltinESMExports } from "node:module";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import vm from "node:vm";
import { createTraceServer } from "./trace-viewer.mjs";

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
  await symlink(join(root, ".env"), join(results, "secret.jsonl"));
  await symlink(root, join(results, "outside"));
  await writeFile(join(results, "script.html"), "<script>bad()</script>");
  for (const path of [
    "/.env",
    "/results/../.env",
    "/results/%2e%2e%2f.env",
    "/results/secret.jsonl",
    "/results/outside/.env",
  ]) {
    const response = await fetch(base + path);
    assert.ok([403, 404].includes(response.status));
    assert.ok(!(await response.text()).includes("private"));
  }
  assert.equal(
    (await fetch(`${base}/results/script.html`)).headers.get("content-type"),
    "text/plain; charset=utf-8",
  );
  assert.equal(
    (await fetch(`${base}/api/runs`, { method: "POST" })).status,
    405,
  );
  const blocked = await new Promise((resolve, reject) => {
    const req = request(
      `${base}/api/runs`,
      { headers: { host: "untrusted.test" } },
      (res) => {
        res.resume();
        resolve(res.statusCode);
      },
    );
    req.on("error", reject);
    req.end();
  });
  assert.equal(blocked, 403);

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
