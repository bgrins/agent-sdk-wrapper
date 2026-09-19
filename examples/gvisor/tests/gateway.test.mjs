import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { once } from "node:events";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import http from "node:http";
import net from "node:net";
import { test } from "node:test";

async function gateway(t, provider, upstream) {
  const root = mkdtempSync("/tmp/caddy-test-");
  const socketPath = `${root}/gateway.sock`;
  const child = spawn(
    "caddy",
    ["run", "--config", "/example/gateway/Caddyfile", "--adapter", "caddyfile"],
    {
      env: {
        ...process.env,
        GATEWAY_LISTEN: `unix/${socketPath}|0666`,
        PROVIDER: provider,
        UPSTREAM_KEY: "upstream-secret",
        GATEWAY_TOKEN: "job-token",
        ANTHROPIC_UPSTREAM: upstream,
        OPENAI_UPSTREAM: upstream,
      },
      stdio: ["ignore", "pipe", "pipe"],
    },
  );
  let stdout = "";
  let stderr = "";
  child.stdout.on("data", (data) => (stdout += data));
  child.stderr.on("data", (data) => (stderr += data));
  t.after(async () => {
    child.kill("SIGTERM");
    if (child.exitCode === null) await once(child, "exit");
    assert.ok(!stdout.includes("upstream-secret"));
    assert.ok(!stderr.includes("upstream-secret"));
    rmSync(root, { recursive: true, force: true });
  });
  for (let i = 0; i < 100 && !existsSync(socketPath); i++)
    await new Promise((resolve) => setTimeout(resolve, 10));
  assert.ok(existsSync(socketPath), stderr);
  return socketPath;
}
async function upstream(t, handler) {
  const server = http.createServer(handler);
  const sockets = new Set();
  server.on("connection", (socket) => {
    sockets.add(socket);
    socket.on("close", () => sockets.delete(socket));
  });
  t.after(() => {
    for (const socket of sockets) socket.destroy();
    server.close();
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  return { server, url: `http://127.0.0.1:${server.address().port}` };
}
function call(socketPath, path, headers = {}, method = "POST", body = "{}") {
  return new Promise((resolve, reject) => {
    const req = http.request(
      { socketPath, path, headers, method, timeout: 3000 },
      (res) => {
        let body = "";
        const times = [];
        res.on("data", (data) => {
          body += data;
          times.push(Date.now());
        });
        res.on("end", () => resolve({ status: res.statusCode, body, times }));
      },
    );
    req.on("error", reject);
    req.on("connect", (res, socket) => {
      socket.destroy();
      resolve({ status: res.statusCode });
    });
    req.on("timeout", () => req.destroy(Error("Timeout")));
    req.end(body);
  });
}

test("gateway replaces credentials, restricts routes and streams responses", async (t) => {
  for (const provider of ["anthropic", "openai"]) {
    const received = [];
    const target = await upstream(t, (req, res) => {
      received.push({ headers: req.headers, path: req.url });
      req.resume();
      req.on("end", () => {
        res.writeHead(200, { "content-type": "text/event-stream" });
        res.write("data: first\n\n");
        setTimeout(() => res.end("data: last\n\n"), 50);
      });
    });
    const socket = await gateway(t, provider, target.url);
    const path = provider === "anthropic" ? "/v1/messages" : "/v1/responses";
    const auth =
      provider === "anthropic"
        ? { "x-api-key": "job-token", authorization: "incoming-secret" }
        : { authorization: "Bearer job-token", "x-api-key": "incoming-secret" };
    const result = await call(socket, `${path}?beta=true`, {
      ...auth,
      host: "evil.test",
      "proxy-authorization": "hidden",
      "x-forwarded-host": "evil.test",
      "openai-project": "other",
      connection: "x-strip",
      "x-strip": "hidden",
    });
    assert.equal(result.status, 200);
    assert.match(result.body, /first.*last/s);
    assert.ok(result.times.at(-1) > result.times[0]);
    const { headers, path: forwarded } = received.at(-1);
    assert.equal(forwarded, `${path}?beta=true`);
    assert.equal(
      headers[provider === "anthropic" ? "x-api-key" : "authorization"],
      provider === "anthropic" ? "upstream-secret" : "Bearer upstream-secret",
    );
    assert.equal(
      headers[provider === "anthropic" ? "authorization" : "x-api-key"],
      undefined,
    );
    assert.equal(headers.host, new URL(target.url).host);
    for (const name of [
      "proxy-authorization",
      "x-forwarded-host",
      "openai-project",
      "x-strip",
    ])
      assert.equal(headers[name], undefined);
    assert.equal((await call(socket, path)).status, 403);
    assert.equal(
      (await call(socket, path, auth, "POST", Buffer.alloc(17000000))).status,
      413,
    );
    const count = received.length;
    // A path matcher cleans most of these to allowed routes; the proxy would forward them raw.
    for (const bad of [
      "//evil.test/v1/messages",
      "/v1/../admin",
      "/v1/%2e%2e/admin",
      "/v1/files",
      `/v1/files/..${path.slice(3)}`,
      `/v1/files%2F..%2F${path.slice(4)}`,
      path.toUpperCase(),
      path.replace("/v1/", "/v1//"),
      `${path}/`,
      `${path}/.`,
      path.replace(/s$/, "%73"),
    ])
      assert.equal((await call(socket, bad, auth)).status, 403, bad);
    assert.equal(received.length, count);
    // Absolute-form targets must be rejected or reach this same fixed upstream.
    const absolute = await call(socket, `http://evil.test${path}`, auth);
    if (absolute.status === 200) {
      assert.equal(received.length, count + 1);
      assert.equal(received.at(-1).headers.host, new URL(target.url).host);
    } else {
      assert.equal(absolute.status, 403);
      assert.equal(received.length, count);
    }
    assert.equal((await call(socket, path, auth, "DELETE")).status, 403);
    const beforeConnect = received.length;
    const connect = await call(socket, "evil.test:443", auth, "CONNECT", "");
    assert.ok([400, 403].includes(connect.status));
    assert.equal(received.length, beforeConnect);
  }
});

test("gateway allows only the betas the pinned SDKs send", async (t) => {
  const received = [];
  const target = await upstream(t, (req, res) => {
    received.push(req.headers["anthropic-beta"]);
    req.resume();
    req.on("end", () => res.end("ok"));
  });
  const socket = await gateway(t, "anthropic", target.url);
  const auth = { "x-api-key": "job-token" };
  const sdk =
    "claude-code-20250219,interleaved-thinking-2025-05-14,context-management-2025-06-27";
  const allowed = [
    ["/v1/messages?beta=true", sdk],
    ["/v1/messages/count_tokens?beta=true", "token-counting-2024-11-01"],
  ];
  for (const [path, beta] of allowed)
    assert.equal(
      (await call(socket, path, { ...auth, "anthropic-beta": beta })).status,
      200,
    );
  for (const beta of [
    "mcp-client-2025-11-20",
    `${sdk},mcp-client-2025-11-20`,
    `${sdk}, mcp-client-2025-11-20`,
    [sdk, "mcp-client-2025-11-20"],
    "Claude-code-20250219",
  ])
    assert.equal(
      (await call(socket, "/v1/messages", { ...auth, "anthropic-beta": beta }))
        .status,
      403,
      String(beta),
    );
  assert.deepEqual(
    received,
    allowed.map(([, beta]) => beta),
  );
});

test("gateway blocks upstream redirects", async (t) => {
  let count = 0;
  const target = await upstream(t, (_req, res) => {
    count++;
    res.writeHead(307, { location: "http://169.254.169.254/" }).end();
  });
  const socket = await gateway(t, "anthropic", target.url);
  assert.equal(
    (await call(socket, "/v1/messages", { "x-api-key": "job-token" })).status,
    502,
  );
  assert.equal(count, 1);
});

test("gateway forwards WebSocket bytes with the upstream credential", async (t) => {
  const target = await upstream(t);
  target.server.on("upgrade", (req, socket, head) => {
    assert.equal(req.headers.authorization, "Bearer upstream-secret");
    socket.write(
      "HTTP/1.1 101 Switching Protocols\r\nConnection: Upgrade\r\nUpgrade: websocket\r\n\r\n",
    );
    if (head.length) socket.write(head);
    socket.pipe(socket);
  });
  const socketPath = await gateway(t, "openai", target.url);
  await new Promise((resolve, reject) => {
    const socket = net.connect(socketPath);
    let data = "";
    socket.setTimeout(3000, () => socket.destroy(Error("Timeout")));
    socket.on("error", reject);
    socket.on("data", (chunk) => {
      data += chunk;
      if (data.includes("payload")) {
        assert.match(data, /101 Switching/);
        socket.destroy();
        resolve();
      }
    });
    socket.on("connect", () =>
      socket.write(
        "GET /v1/responses HTTP/1.1\r\nHost: evil.test\r\nAuthorization: Bearer job-token\r\nConnection: Upgrade\r\nUpgrade: websocket\r\n\r\npayload",
      ),
    );
  });
});
