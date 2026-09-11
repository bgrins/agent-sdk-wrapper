import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import {
  copyFileSync,
  mkdirSync,
  mkdtempSync,
  rmSync,
  symlinkSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const packageRoot = fileURLToPath(new URL("../", import.meta.url));
const repository = fileURLToPath(new URL("../../../", import.meta.url));
const temp = mkdtempSync(join(tmpdir(), "agent-sdk-wrapper-package-"));
try {
  const [pack] = JSON.parse(
    execFileSync(
      "npm",
      [
        "pack",
        "--offline",
        "--ignore-scripts",
        "--json",
        "--pack-destination",
        temp,
      ],
      { cwd: packageRoot, encoding: "utf8" },
    ),
  );
  const paths = new Set(pack.files.map((file) => file.path));
  for (const path of [
    "dist/index.js",
    "dist/index.d.ts",
    "README.md",
    "PARITY.md",
    "VALIDATION.md",
    "LICENSE",
  ])
    assert.ok(paths.has(path), `Missing packed file: ${path}`);
  for (const path of paths)
    assert.ok(
      path.startsWith("dist/") ||
        [
          "package.json",
          "README.md",
          "PARITY.md",
          "VALIDATION.md",
          "LICENSE",
        ].includes(path),
      `Unexpected packed file: ${path}`,
    );
  const installed = join(temp, "node_modules", "agent-sdk-wrapper");
  mkdirSync(installed, { recursive: true });
  execFileSync("tar", [
    "-xzf",
    join(temp, pack.filename),
    "-C",
    installed,
    "--strip-components=1",
  ]);
  // Reuse the exact locked dependencies without a registry request or install.
  // Consumer code resolves the wrapper from the extracted tarball, not the workspace.
  symlinkSync(
    join(repository, "node_modules"),
    join(installed, "node_modules"),
    "dir",
  );
  symlinkSync(
    join(repository, "node_modules", "@types"),
    join(temp, "node_modules", "@types"),
    "dir",
  );
  const consumer = join(temp, "consumer.mts");
  copyFileSync(join(packageRoot, "fixtures", "package-consumer.mts"), consumer);
  execFileSync(
    process.execPath,
    [
      join(repository, "node_modules", "typescript", "bin", "tsc"),
      "--module",
      "NodeNext",
      "--target",
      "ES2022",
      "--strict",
      "--noUncheckedIndexedAccess",
      "--skipLibCheck",
      "--types",
      "node",
      "--outDir",
      join(temp, "out"),
      consumer,
    ],
    { cwd: temp, stdio: "inherit" },
  );
  execFileSync(process.execPath, [join(temp, "out", "consumer.mjs")], {
    cwd: temp,
    stdio: "inherit",
  });
} finally {
  rmSync(temp, { recursive: true, force: true });
}
