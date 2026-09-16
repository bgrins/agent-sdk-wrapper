import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { existsSync, readFileSync, statfsSync, writeFileSync } from "node:fs";
import http from "node:http";
import net from "node:net";
import os from "node:os";
import { test } from "node:test";

test("worker has no host credentials, privileged mounts or writable image", () => {
  assert.match(os.release(), /gvisor/);
  assert.notEqual(process.getuid(), 0);
  assert.deepEqual(Object.keys(os.networkInterfaces()), ["lo"]);
  for (const path of ["/var/run/docker.sock", "/run/docker.sock"])
    assert.equal(existsSync(path), false);
  for (const name of ["UPSTREAM_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"])
    assert.equal(process.env[name], undefined);
  assert.ok(
    !readFileSync("/proc/1/environ", "utf8").includes("outer-only-canary"),
  );
  for (const path of ["/etc/agent-escape", "/inference/replaced.sock"])
    assert.throws(
      () => writeFileSync(path, "escape"),
      (error) => ["EACCES", "EROFS"].includes(error.code),
    );
  const stat = statfsSync("/tmp");
  assert.ok(stat.bsize * stat.blocks <= 128 * 1024 * 1024);
  const child = spawnSync(
    "node",
    [
      "--input-type=module",
      "-e",
      "import fs from 'node:fs';fs.writeFileSync('/tmp/large','');fs.truncateSync('/tmp/large',17*1024*1024)",
    ],
    { encoding: "utf8", timeout: 5000 },
  );
  assert.ok(
    child.signal === "SIGXFSZ" ||
      (child.status === 1 && child.stderr.includes("EFBIG")),
    child.stderr,
  );
});

test("worker cannot reach the internet, metadata service or Docker host", async () => {
  for (const [host, port] of [
    ["1.1.1.1", 443],
    ["169.254.169.254", 80],
    ["172.17.0.1", 2375],
    ["::ffff:1.1.1.1", 443],
    ["example.com", 443],
  ]) {
    await new Promise((resolve, reject) => {
      const socket = net.connect({ host, port });
      socket.setTimeout(1500, () => {
        socket.destroy();
        resolve();
      });
      socket.on("error", resolve);
      socket.on("connect", () => {
        socket.destroy();
        reject(new Error(`Unexpected access: ${host}:${port}`));
      });
    });
  }
});

test("only the job token reaches the offline inference endpoint", async () => {
  const claude = process.env.PROVIDER === "anthropic";
  const call = (token) =>
    new Promise((resolve, reject) => {
      const req = http.request(
        {
          socketPath: "/inference/gateway.sock",
          path: claude ? "/v1/messages" : "/v1/responses",
          method: "POST",
          headers: claude
            ? { "x-api-key": token }
            : { authorization: `Bearer ${token}` },
          timeout: 3000,
        },
        (res) => {
          res.resume();
          res.on("end", () => resolve(res.statusCode));
        },
      );
      req.on("error", reject);
      req.on("timeout", () => req.destroy(new Error("Gateway timeout")));
      req.end("{}");
    });
  assert.equal(await call(process.env.GATEWAY_TOKEN), 400);
  assert.equal(await call(""), 403);
  assert.equal(await call("other-job"), 403);
});
