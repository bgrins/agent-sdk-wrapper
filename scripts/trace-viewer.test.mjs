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
  MAX_RUNS,
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
  const context = await loadViewer(`${base}/docs/trace-viewer.html`);
  context.fetch = fetch;
  const { runs } = await vm.runInContext("findResultRuns()", context);
  context.run = runs.find((entry) => entry.label === "custom trace");
  await vm.runInContext("loadRunFromUrl(run)", context);
  assert.equal(vm.runInContext("loadErrorEl.textContent", context), "");
  assert.match(visibleTrace(context), /custom trace/);

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

async function listen(t, directory, options, create = createTraceServer) {
  const server = create(directory, options);
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
  const count = MAX_RUNS + 100;
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
  assert.equal(response.headers["x-runs-found"], String(count));
  assert.equal(response.headers["x-directories-scanned"], undefined);
  const runs = JSON.parse(response.body);
  const newest = Array.from({ length: count }, (_, age) =>
    stamp(age).toISOString(),
  )
    .reverse()
    .slice(0, runs.length);
  assert.equal(runs.length, MAX_RUNS);
  assert.deepEqual(
    runs.map((run) => run.updated_at),
    newest,
  );
});

test("past the directory cap the scan skips the oldest directories and says so", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "run-directory-cap-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  for (let index = 0; index < MAX_DIRECTORIES + 10; index += 500)
    await Promise.all(
      Array.from({ length: Math.min(500, MAX_DIRECTORIES + 10 - index) }, (_, i) =>
        mkdir(join(root, `old-${index + i}`)),
      ),
    );
  const newest = join(root, "newest");
  await mkdir(newest);
  await writeFile(join(newest, "trace.jsonl"), "{}\n");
  const later = new Date(Date.now() + 60_000);
  await utimes(newest, later, later);
  // List the newest directory last, as readdir order sometimes does.
  const readdir = fs.readdir;
  const mocked = t.mock.method(fs, "readdir", async (path, ...args) => {
    const entries = await readdir(path, ...args);
    return path === root
      ? [
          ...entries.filter((entry) => entry.name !== "newest"),
          ...entries.filter((entry) => entry.name === "newest"),
        ]
      : entries;
  });
  syncBuiltinESMExports();
  t.after(() => {
    mocked.mock.restore();
    syncBuiltinESMExports();
  });
  const base = await listen(t, root, { depth: 1 });
  const response = await get(base, "/api/runs");
  assert.deepEqual(
    JSON.parse(response.body).map((run) => run.label),
    ["newest"],
  );
  assert.equal(response.headers["x-runs-found"], undefined);
  assert.equal(
    response.headers["x-directories-scanned"],
    String(MAX_DIRECTORIES),
  );
});

test("a fresh run's large subtree does not hide its older sibling runs", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "run-subtree-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const old = join(root, "run-old");
  await mkdir(old);
  await writeFile(join(old, "trace.jsonl"), "{}\n");
  const hour = new Date(Date.now() - 3_600_000);
  await utimes(join(old, "trace.jsonl"), hour, hour);
  await utimes(old, hour, hour);
  const fresh = join(root, "run-new");
  await Promise.all(
    Array.from({ length: 600 }, (_, index) =>
      mkdir(join(fresh, "workspace", `d${index}`), { recursive: true }),
    ),
  );
  await writeFile(join(fresh, "trace.jsonl"), "{}\n");
  const base = await listen(t, root, { depth: 3 });
  const runs = JSON.parse((await get(base, "/api/runs")).body);
  assert.deepEqual(runs.map((run) => run.label).sort(), ["run-new", "run-old"]);
});

test("a newer deep run is listed ahead of older shallow runs", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "run-depth-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const old = new Date(Date.now() - 3_600_000);
  await Promise.all(
    Array.from({ length: MAX_RUNS + 20 }, async (_, index) => {
      const dir = join(root, `old-${index}`);
      await mkdir(dir);
      await writeFile(join(dir, "trace.jsonl"), "{}\n");
      await utimes(join(dir, "trace.jsonl"), old, old);
    }),
  );
  const deep = join(root, "jobs", "a", "b", "fresh");
  await mkdir(deep, { recursive: true });
  await writeFile(join(deep, "trace.jsonl"), "{}\n");
  const base = await listen(t, root, { depth: 5 });
  const runs = JSON.parse((await get(base, "/api/runs")).body);
  assert.equal(runs[0].label, "jobs/a/b/fresh");
});

// freebsd stands in for platforms with neither O_NOFOLLOW_ANY nor /proc.
for (const platform of [process.platform, "freebsd"]) {
  test(`a symlinked ancestor swapped in after path checks is not followed on ${platform}`, async (t) => {
    const original = Object.getOwnPropertyDescriptor(process, "platform");
    Object.defineProperty(process, "platform", { value: platform });
    t.after(() => Object.defineProperty(process, "platform", original));
    // The module picks its open flags at import.
    const viewer = await import(`./trace-viewer.mjs?platform=${platform}`);
    const root = await mkdtemp(join(tmpdir(), "ancestor-swap-"));
    t.after(() => rm(root, { recursive: true, force: true }));
    const results = join(root, "results");
    await mkdir(join(results, "job", "logs"), { recursive: true });
    await mkdir(join(root, "elsewhere"));
    await writeFile(join(results, "job", "logs", "trace.jsonl"), "{}\n");
    await writeFile(join(results, "top.trace.jsonl"), "top\n");
    await writeFile(join(root, "elsewhere", "trace.jsonl"), "private\n");
    const base = await listen(t, results, {}, viewer.createTraceServer);
    assert.equal((await get(base, "/results/top.trace.jsonl")).body, "top\n");
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
}

// Just enough DOM for the viewer script: nodes, text, keys and listeners.
class FakeNode {
  constructor(tagName) {
    this.tagName = tagName;
    this.children = [];
    this.parentNode = null;
    this.text = "";
    this.dataset = {};
    this.attributes = {};
    this.listeners = {};
    this.classes = new Set();
    this.classList = {
      toggle: (name, force = !this.classes.has(name)) => {
        if (force) this.classes.add(name);
        else this.classes.delete(name);
        return force;
      },
      contains: (name) => this.classes.has(name),
    };
    this.scrollTop = 0;
    this.scrollHeight = 0;
    this.clientHeight = 0;
  }
  get className() {
    return [...this.classes].join(" ");
  }
  set className(value) {
    this.classes = new Set(String(value).split(/\s+/).filter(Boolean));
  }
  get textContent() {
    return this.text + this.children.map((node) => node.textContent).join("");
  }
  set textContent(value) {
    this.replaceChildren();
    this.text = String(value ?? "");
  }
  get firstElementChild() {
    return this.children.find((node) => node.tagName !== "#text") ?? null;
  }
  get nextElementSibling() {
    const siblings = this.parentNode?.children ?? [];
    return (
      siblings
        .slice(siblings.indexOf(this) + 1)
        .find((node) => node.tagName !== "#text") ?? null
    );
  }
  append(...nodes) {
    for (const node of nodes) this.insertBefore(node, null);
  }
  replaceChildren(...nodes) {
    for (const node of this.children) node.parentNode = null;
    this.children = [];
    this.text = "";
    this.append(...nodes);
  }
  insertBefore(node, reference) {
    if (typeof node === "string") {
      const text = new FakeNode("#text");
      text.text = node;
      node = text;
    }
    if (node.tagName === "#fragment") {
      const moved = node.children;
      node.replaceChildren();
      for (const child of moved) this.insertBefore(child, reference);
      return node;
    }
    node.remove();
    const index = reference ? this.children.indexOf(reference) : -1;
    this.children.splice(index < 0 ? this.children.length : index, 0, node);
    node.parentNode = this;
    return node;
  }
  remove() {
    if (!this.parentNode) return;
    const siblings = this.parentNode.children;
    siblings.splice(siblings.indexOf(this), 1);
    this.parentNode = null;
  }
  setAttribute(name, value) {
    this.attributes[name] = String(value);
  }
  getAttribute(name) {
    return this.attributes[name] ?? null;
  }
  addEventListener(type, listener) {
    this.listeners[type] = [...(this.listeners[type] || []), listener];
  }
  dispatch(type) {
    for (const listener of this.listeners[type] || []) listener({});
  }
  find(predicate) {
    for (const node of this.children) {
      if (predicate(node)) return node;
      const found = node.find(predicate);
      if (found) return found;
    }
    return null;
  }
}

const VIEWS = [
  "conversation-view",
  "timeline-view",
  "events-view",
  "files-view",
  "raw-view",
];

async function loadViewer(href) {
  const html = await readFile(
    new URL("../docs/trace-viewer.html", import.meta.url),
    "utf8",
  );
  const elements = new Map();
  const tabs = VIEWS.map((view) => {
    const tab = new FakeNode("button");
    tab.dataset.view = view;
    return tab;
  });
  const views = VIEWS.map((id) =>
    Object.assign(new FakeNode("section"), { id }),
  );
  const context = vm.createContext({
    document: {
      querySelector(selector) {
        if (!elements.has(selector))
          elements.set(selector, new FakeNode("div"));
        return elements.get(selector);
      },
      querySelectorAll: (selector) =>
        ({ ".tab-btn": tabs, ".view": views })[selector] ?? [],
      createElement: (tagName) => new FakeNode(tagName),
      createDocumentFragment: () => new FakeNode("#fragment"),
      body: new FakeNode("body"),
    },
    window: { location: { protocol: "file:" } },
    localStorage: { getItem() {} },
    setInterval() {},
    URL,
    URLSearchParams,
    Blob,
  });
  vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1], context);
  if (href) {
    const url = new URL(href);
    context.window.location = {
      href,
      origin: url.origin,
      search: url.search,
      protocol: url.protocol,
    };
  }
  return context;
}

const visibleTrace = (context) =>
  vm.runInContext("shown?.traceText ?? ''", context);

const traceText = (rows) =>
  rows.map((row) => `${JSON.stringify(row)}\n`).join("");

// Copy values out of the page's realm so deepEqual compares plain data.
const evaluate = (context, code) =>
  JSON.parse(vm.runInContext(`JSON.stringify(${code})`, context));

test("trace-only metadata follows the latest resumed run and sums usage across turns", async () => {
  const context = await loadViewer();
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
  context.trace.push({
    run_id: "two",
    event: { type: "session_info", id: "session", model: "resolved-model" },
  });
  assert.equal(metadata().model, "resolved-model");
});

test("polling restores an unchanged trace after a failed manual refresh", async () => {
  const context = await loadViewer();
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
  const error = () => vm.runInContext("loadErrorEl.textContent", context);
  const poll = () =>
    vm.runInContext("loadRunFromUrl(run, { refresh: true })", context);

  await vm.runInContext("loadRunFromUrl(run)", context);
  assert.equal(error(), "");
  assert.equal(visibleTrace(context), trace);

  unavailable = true;
  await vm.runInContext("loadRunFromUrl(run)", context);
  assert.match(error(), /503/);
  assert.equal(visibleTrace(context), "");

  unavailable = false;
  await poll();
  assert.equal(error(), "");
  assert.equal(visibleTrace(context), trace);

  unavailable = true;
  await poll();
  assert.match(error(), /503/);
  assert.equal(visibleTrace(context), trace);

  unavailable = false;
  await poll();
  assert.equal(error(), "");
  assert.equal(visibleTrace(context), trace);
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
  const context = await loadViewer();
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

test("unsafe manifest file entries render as text without blanking the run", async () => {
  const context = await loadViewer();
  const manifest = "http://localhost/results/job/manifest.json";
  context.run = { runUrl: new URL(manifest), manifestUrl: new URL(manifest) };
  const files = {
    trace: "trace.jsonl",
    parent: "../other/trace.jsonl",
    script: "javascript:alert(1)",
    absolute: "/etc/passwd",
    count: 5,
  };
  context.fetch = async (url) =>
    new Response(
      url.pathname.endsWith("manifest.json")
        ? JSON.stringify({ files })
        : traceText([{ run_id: "r", event: { type: "text", text: "kept" } }]),
    );
  await vm.runInContext("loadRunFromUrl(run)", context);
  vm.runInContext('switchView("files-view")', context);
  assert.equal(vm.runInContext("loadErrorEl.textContent", context), "");
  assert.match(vm.runInContext("conversationEl.textContent", context), /kept/);
  const list = vm.runInContext("filesEl", context);
  assert.deepEqual(
    list.children.flatMap((item) => item.children.map((link) => link.href)),
    [
      "http://localhost/results/job/manifest.json",
      "http://localhost/results/job/trace.jsonl",
    ],
  );
  for (const name of [files.parent, files.script, files.absolute])
    assert.ok(list.textContent.includes(`${name} (outside the run directory)`));
});

test("the viewer only lists runs from its own origin", async () => {
  const requested = [];
  const spoofed = await loadViewer(
    "http://localhost:8765/docs/trace-viewer.html?index=http://evil.test/runs.json",
  );
  spoofed.fetch = async (url) => {
    requested.push(String(url));
    return new Response("[]");
  };
  await assert.rejects(
    vm.runInContext("findResultRuns()", spoofed),
    /this trace server/,
  );
  assert.deepEqual(requested, []);

  const context = await loadViewer(
    "http://localhost:8765/docs/trace-viewer.html?index=/api/runs",
  );
  context.fetch = async (url) => {
    requested.push(String(url));
    if (url.pathname !== "/api/runs")
      return new Response(
        traceText([{ run_id: "r", event: { type: "text", text: "local" } }]),
      );
    const runs = [
      { label: "spoofed", trace: "http://evil.test/trace.jsonl" },
      { label: "spoofed manifest", manifest: "//evil.test/manifest.json" },
      { label: "local", trace: "/results/local/trace.jsonl" },
    ];
    return new Response(JSON.stringify(runs), {
      headers: { "x-runs-found": "812", "x-directories-scanned": "5000" },
    });
  };
  await vm.runInContext("discoverResults()", context);
  assert.deepEqual(
    evaluate(context, "discoveredRuns.map((run) => run.label)"),
    ["local"],
  );
  assert.ok(requested.every((url) => url.startsWith("http://localhost:8765/")));
  assert.match(visibleTrace(context), /local/);
  assert.equal(
    vm.runInContext("resultsStatusEl.textContent", context),
    "Showing the 1 most recently updated of 812 runs. Scanned only the 5000 most recently modified directories.",
  );
});

test("local artifacts open as plain text rather than pages in the viewer origin", async () => {
  const context = await loadViewer();
  const types = [];
  context.URL = class extends URL {
    static createObjectURL(blob) {
      types.push(blob.type);
      return `blob:viewer/${types.length}`;
    }
    static revokeObjectURL() {}
  };
  context.files = [
    new File(["<script>document.title = 'owned'</script>"], "evil.html", {
      type: "text/html",
    }),
    new File(
      [traceText([{ run_id: "r", event: { type: "text", text: "hi" } }])],
      "trace.jsonl",
    ),
  ];
  await vm.runInContext("loadFiles(files)", context);
  vm.runInContext('switchView("files-view")', context);
  assert.equal(types.length, 2);
  for (const type of types) assert.match(type, /^text\/plain/);
});

test("invalid trace lines are skipped with a visible warning", async () => {
  const context = await loadViewer();
  const text =
    traceText([{ run_id: "n", event: { type: "run_started", prompt: "n" } }]) +
    'null\n[1]\n"text"\n{"event":5}\nnot json\n' +
    traceText([
      { run_id: "n", event: { type: "thinking", text: { summary: "object" } } },
      { run_id: "n", event: { type: "text", text: "after null" } },
    ]) +
    '{"run_id":"n","event":{"type":"te';
  context.files = [new File([text], "trace.jsonl")];
  await vm.runInContext("loadFiles(files)", context);
  assert.equal(vm.runInContext("loadErrorEl.textContent", context), "");
  assert.equal(vm.runInContext("shown.trace.length", context), 3);
  const warning = vm.runInContext("loadWarningEl.textContent", context);
  assert.match(warning, /^Skipped 6 invalid trace lines: /);
  assert.match(warning, /trace\.jsonl line 2 is not a trace event/);
  assert.match(
    evaluate(context, "shown.warnings.at(-1)"),
    /line 9 is incomplete/,
  );
  const conversation = vm.runInContext("conversationEl.textContent", context);
  assert.match(conversation, /after null/);
  assert.match(conversation, /"summary":"object"/);
});

test("the conversation shows every prompt, the session model, run ends and unmatched results", async () => {
  const context = await loadViewer();
  const row = (run_id, event) => ({ run_id, event });
  const rows = [
    row("a", { type: "run_started", prompt: "first", system_prompt: "sys" }),
    row("a", { type: "session_info", id: "session-1", model: "resolved" }),
    row("a", { type: "tool_call", id: "item_0", input: { command: "ls A" } }),
    row("a", { type: "tool_result", id: "item_0", output: "RESULT-A" }),
    row("a", { type: "tool_call", id: "item_1", input: { command: "sleep" } }),
    row("a", { type: "run_finished", status: "cancelled", duration_ms: 1500 }),
    row("b", { type: "run_started", prompt: "second", system_prompt: "sys" }),
    row("b", { type: "tool_call", id: "item_1", input: { command: "ls B" } }),
    row("b", { type: "tool_result", id: "item_1", output: "RESULT-B" }),
    row("b", { type: "tool_result", id: "missing", output: "ORPHAN" }),
    row("b", { type: "run_finished", status: "success", duration_ms: 20 }),
  ];
  context.files = [new File([traceText(rows)], "trace.jsonl")];
  await vm.runInContext("loadFiles(files)", context);
  const items = evaluate(context, "buildConversation(shown.trace)");
  assert.deepEqual(
    items.filter((item) => item.role === "user").map((item) => item.text),
    ["first", "second"],
  );
  assert.equal(items.filter((item) => item.kind === "system").length, 1);
  assert.deepEqual(
    items
      .filter((item) => item.kind === "tool")
      .map((item) => [item.call?.input.command, item.result?.output]),
    [
      ["ls A", "RESULT-A"],
      ["sleep", undefined],
      ["ls B", "RESULT-B"],
      [undefined, "ORPHAN"],
    ],
  );
  assert.deepEqual(
    items
      .filter((item) => item.kind === "banner")
      .map((item) => `${item.title} | ${item.body}`),
    [
      "Session | id: session-1 · model: resolved",
      "Run finished: cancelled | 1.5 s",
      "Run finished: success | 20 ms",
    ],
  );
  const conversation = vm.runInContext("conversationEl.textContent", context);
  for (const text of ["model: resolved", "ORPHAN", "second", "RESULT-B"])
    assert.ok(conversation.includes(text), text);
});

test("timeline rows show the content of lifecycle events", async () => {
  const context = await loadViewer();
  const events = [
    { type: "agent_updated", name: "planner" },
    { type: "subagent_started", task_id: "t1", name: "reviewer", description: "check diff" },
    { type: "subagent_ended", task_id: "t1", status: "completed", summary: "looks fine" },
    { type: "context_compacted", trigger: "auto", pre_tokens: 1200 },
    { type: "run_finished", status: "failed", ended_reason: "timeout", duration_ms: 1500 },
  ];
  assert.deepEqual(
    events.map((event) => evaluate(context, `eventBody(${JSON.stringify(event)})`)),
    ["planner", "check diff", "looks fine", "trigger: auto", "reason: timeout · 1.5 s"],
  );
});

test("polling renders only the active view and appends new lines to it", async () => {
  const context = await loadViewer();
  const limit = vm.runInContext("OUTPUT_LIMIT", context);
  const rawLimit = vm.runInContext("RAW_LIMIT", context);
  let text = traceText([
    { run_id: "r", event: { type: "run_started", prompt: "go" } },
    { run_id: "r", event: { type: "tool_call", id: "t", name: "shell" } },
  ]);
  context.fetch = async () => new Response(text);
  context.run = {
    runUrl: new URL("http://localhost/results/r/trace.jsonl"),
    traceRel: "trace.jsonl",
  };
  const poll = () =>
    vm.runInContext("loadRunFromUrl(run, { refresh: true })", context);
  const conversation = vm.runInContext("conversationEl", context);
  const timeline = vm.runInContext("timelineEl", context);
  const raw = vm.runInContext("rawEl", context);
  const toolState = () =>
    conversation.children[1].find((node) => node.classes.has("tool-state"))
      .textContent;

  await vm.runInContext("loadRunFromUrl(run)", context);
  const prompt = conversation.firstElementChild;
  assert.equal(toolState(), "pending");
  assert.equal(timeline.children.length, 0, "Inactive views wait to render");

  const output = "x".repeat(limit * 2);
  text += traceText([
    { run_id: "r", event: { type: "tool_result", id: "t", output } },
    { run_id: "r", event: { type: "text", text: "finished" } },
  ]);
  await poll();
  assert.equal(conversation.firstElementChild, prompt, "Keep existing items");
  assert.equal(conversation.children.length, 3);
  assert.equal(toolState(), "done");
  const pre = conversation.children[1].find(
    (node) => node.tagName === "pre" && node.textContent.startsWith("x"),
  );
  assert.equal(pre.textContent.length, limit);
  conversation.children[1]
    .find((node) => node.classes.has("show-all"))
    .dispatch("click");
  assert.equal(pre.textContent, output);
  assert.equal(timeline.children.length, 0);

  vm.runInContext('switchView("timeline-view")', context);
  assert.equal(timeline.children.length, 4);
  const firstRow = timeline.firstElementChild;
  text += traceText([
    { run_id: "r", event: { type: "text", text: "y".repeat(rawLimit) } },
  ]);
  await poll();
  assert.equal(timeline.firstElementChild, firstRow, "Append timeline rows");
  assert.equal(timeline.children.length, 5);

  vm.runInContext('switchView("raw-view")', context);
  assert.equal(raw.textContent, text.slice(0, rawLimit));
  vm.runInContext("rawMoreButton", context).dispatch("click");
  assert.equal(raw.textContent, text);
  text += traceText([{ run_id: "r", event: { type: "run_finished" } }]);
  await poll();
  assert.equal(raw.textContent, text);
  assert.equal(vm.runInContext("rawMoreButton.hidden", context), true);
});

test("a click during a poll never shows the previous run as the selection", async () => {
  const context = await loadViewer(
    "http://localhost/docs/trace-viewer.html?index=/api/runs",
  );
  let release;
  let requestsForB = 0;
  context.fetch = async (url) => {
    if (url.pathname === "/api/runs")
      return new Response(
        JSON.stringify([
          { label: "a", trace: "/results/a/trace.jsonl", updated_at: "2" },
          { label: "b", trace: "/results/b/trace.jsonl", updated_at: "1" },
        ]),
      );
    const run = url.pathname.split("/")[2];
    if (run === "a")
      return new Response(
        traceText([{ run_id: "a", event: { type: "text", text: "RUN A" } }]),
      );
    requestsForB += 1;
    if (requestsForB > 1) return new Response("Unavailable", { status: 503 });
    await new Promise((resolve) => {
      release = resolve;
    });
    return new Response(
      traceText([{ run_id: "b", event: { type: "text", text: "RUN B" } }]),
    );
  };
  await vm.runInContext("discoverResults()", context);
  assert.match(visibleTrace(context), /RUN A/);
  const click = vm.runInContext(
    "followLatest = false; loadRunFromUrl(discoveredRuns[1])",
    context,
  );
  await vm.runInContext("discoverResults()", context);
  release();
  await click;
  assert.doesNotMatch(visibleTrace(context), /RUN A/);
  assert.match(vm.runInContext("loadErrorEl.textContent", context), /503/);
});
