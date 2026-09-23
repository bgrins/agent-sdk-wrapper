import assert from "node:assert/strict";
import { spawn, spawnSync } from "node:child_process";
import { once } from "node:events";
import {
  existsSync,
  readFileSync,
  rmSync,
  statfsSync,
  statSync,
} from "node:fs";
import http from "node:http";
import net from "node:net";
import os from "node:os";
import { test } from "node:test";

test("worker has no host credentials, privileges or writable image", () => {
  assert.match(os.release(), /gvisor/);
  assert.notEqual(process.getuid(), 0);
  const status = Object.fromEntries(
    readFileSync("/proc/self/status", "utf8")
      .split("\n")
      .map((line) => line.split(":\t")),
  );
  for (const name of ["CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb"])
    assert.equal(status[name], "0000000000000000", name);
  assert.equal(status.NoNewPrivs, "1");
  assert.deepEqual(Object.keys(os.networkInterfaces()), ["lo"]);
  for (const path of ["/var/run/docker.sock", "/run/docker.sock"])
    assert.equal(existsSync(path), false);
  for (const name of ["UPSTREAM_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"])
    assert.equal(process.env[name], undefined);
  assert.ok(
    !readFileSync("/proc/1/environ", "utf8").includes("outer-only-canary"),
  );
  const mounts = new Map(
    readFileSync("/proc/self/mounts", "utf8")
      .trim()
      .split("\n")
      .map((line) => line.split(" "))
      .map(([, target, , options]) => [target, options.split(",")]),
  );
  for (const target of ["/", "/inference"])
    assert.ok(mounts.get(target)?.includes("ro"), target);
  const stat = statfsSync("/tmp");
  assert.ok(stat.bsize * stat.blocks <= 128 * 1024 * 1024);
});

test("file size limit covers tmpfs and gofer-backed volumes", (t) => {
  for (const path of ["/tmp/large", "/job/large"]) {
    t.after(() => rmSync(path, { force: true }));
    const child = spawnSync(
      "node",
      [
        "--input-type=module",
        "-e",
        `import fs from 'node:fs';fs.writeFileSync('${path}','');fs.truncateSync('${path}',17*1024*1024)`,
      ],
      { encoding: "utf8", timeout: 5000 },
    );
    assert.ok(
      child.signal === "SIGXFSZ" ||
        (child.status === 1 && child.stderr.includes("EFBIG")),
      `${path}: ${child.stderr}`,
    );
  }
});

test("worker cannot reach the internet, metadata service or Docker host", async () => {
  // With a route, these would connect, be refused or time out instead.
  for (const [host, port, codes] of [
    ["1.1.1.1", 443, ["ENETUNREACH"]],
    ["169.254.169.254", 80, ["ENETUNREACH"]],
    ["172.17.0.1", 2375, ["ENETUNREACH"]],
    ["::ffff:1.1.1.1", 443, ["ENETUNREACH"]],
    ["example.com", 443, ["EAI_AGAIN", "ENOTFOUND"]],
  ]) {
    const code = await new Promise((resolve) => {
      const socket = net.connect({ host, port });
      socket.setTimeout(3000, () => {
        socket.destroy();
        resolve("timeout");
      });
      socket.on("error", (error) => resolve(error.code));
      socket.on("connect", () => {
        socket.destroy();
        resolve("connected");
      });
    });
    assert.ok(codes.includes(code), `${host}:${port}: ${code}`);
  }
});

test("only the job token reaches the non-root offline gateway", async () => {
  // The gateway creates its socket as its own user.
  assert.notEqual(statSync("/inference/gateway.sock").uid, 0);
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

test("process limit stops fork bombs", async (t) => {
  const children = [];
  t.after(async () => {
    const running = children.filter(
      (child) => child.pid && child.exitCode === null && !child.signalCode,
    );
    for (const child of running) child.kill("SIGKILL");
    await Promise.all(running.map((child) => once(child, "exit")));
  });
  let failure;
  while (!failure && children.length < 300) {
    try {
      const child = spawn("sleep", ["30"], { stdio: "ignore" });
      children.push(child);
      failure = await new Promise((resolve) => {
        child.once("spawn", () => resolve());
        child.once("error", resolve);
      });
    } catch (error) {
      failure = error;
    }
  }
  assert.ok(["EAGAIN", "ENOMEM"].includes(failure?.code), String(failure));
  assert.ok(children.length < 256);
});
