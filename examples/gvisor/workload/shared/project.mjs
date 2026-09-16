import { execFileSync } from "node:child_process";
import {
  accessSync,
  constants,
  cpSync,
  existsSync,
  mkdirSync,
  statSync,
  symlinkSync,
} from "node:fs";

function prepare() {
  // A restarted worker uses the same project and SDK session files.
  if (existsSync("/job/work")) return;
  mkdirSync("/job/work");
  mkdirSync("/job/home", { recursive: true });
  cpSync("/sample-app", "/job/work", {
    recursive: true,
    filter: (source) => !source.includes("/node_modules"),
  });
  symlinkSync("/sample-app/node_modules", "/job/work/node_modules");
  const git = (args) =>
    execFileSync("git", args, { cwd: "/job/work", stdio: "pipe" });
  git(["init", "-q"]);
  git(["add", "."]);
  git([
    "-c",
    "user.name=Example",
    "-c",
    "user.email=example@example.invalid",
    "commit",
    "-qm",
    "Initial sample",
  ]);
}

function check() {
  accessSync("/job/work/package.json", constants.R_OK);
  accessSync("/job/work/node_modules/ms/index.js", constants.R_OK);
  const patch = statSync("/job/output/fix.patch", { throwIfNoEntry: false });
  if (!patch?.isFile() || patch.size === 0)
    throw new Error("Missing or empty output patch");
}

if (process.argv[2] === "prepare") prepare();
else if (process.argv[2] === "check") check();
else throw new Error("Use prepare or check");
