import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import {
  linkSync,
  mkdtempSync,
  rmSync,
  symlinkSync,
  truncateSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { readOutputFile } from "./output-file.mjs";

test("output reads reject links, FIFOs and large files", (t) => {
  const root = mkdtempSync(join(tmpdir(), "gvisor-output-"));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const path = (name) => join(root, `${name}.trace.jsonl`);
  writeFileSync(path("run"), "{}\n");
  assert.equal(readOutputFile(path("run")), "{}\n");
  symlinkSync(path("run"), path("symlink"));
  writeFileSync(path("linked"), "{}\n");
  linkSync(path("linked"), path("hardlink"));
  writeFileSync(path("large"), "");
  truncateSync(path("large"), 16 * 1024 * 1024 + 1);
  for (const name of ["symlink", "hardlink", "large"])
    assert.throws(() => readOutputFile(path(name)), name);
  execFileSync("mkfifo", [path("fifo")]);
  // A blocking open would hang this process, so read the FIFO in a child.
  const child = spawnSync(
    process.execPath,
    [
      "--input-type=module",
      "-e",
      `import { readOutputFile } from ${JSON.stringify(new URL("./output-file.mjs", import.meta.url).href)};
       readOutputFile(${JSON.stringify(path("fifo"))});`,
    ],
    { encoding: "utf8", timeout: 5000 },
  );
  assert.equal(child.signal, null, "FIFO read blocked");
  assert.match(child.stderr, /Unsupported output file/);
});
