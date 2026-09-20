import { createHash } from "node:crypto";
import { constants } from "node:fs";
import {
  lstat,
  open,
  readdir,
  readFile,
  readlink,
  realpath,
} from "node:fs/promises";
import { createServer } from "node:http";
import { resolve, sep } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const viewer = fileURLToPath(
  new URL("../docs/trace-viewer.html", import.meta.url),
);
const urlPath = (path) => path.split("/").map(encodeURIComponent).join("/");

export const MAX_DIRECTORIES = 500;

async function listRuns(directory, depth) {
  // Scan level by level, newest first, so the cap drops the oldest runs and one
  // run's subtree cannot use up the budget before its siblings are read.
  const queue = [{ relative: "", level: 0, mtime: Infinity }];
  const runs = [];
  for (let visited = 0; queue.length && visited < MAX_DIRECTORIES; visited++) {
    let newest = 0;
    for (let index = 1; index < queue.length; index++) {
      const [a, b] = [queue[index], queue[newest]];
      if (a.level < b.level || (a.level === b.level && a.mtime > b.mtime))
        newest = index;
    }
    const { relative } = queue[newest];
    queue[newest] = queue.at(-1);
    queue.pop();
    let entries;
    try {
      entries = await readdir(resolve(directory, relative), {
        withFileTypes: true,
      });
    } catch (error) {
      // An unreadable or replaced directory must not hide the other runs.
      if (!relative && error.code !== "ENOENT") throw error;
      continue;
    }
    const prefix = relative ? `${relative}/` : "";
    const hasManifest = entries.some(
      (entry) => entry.name === "manifest.json" && entry.isFile(),
    );
    const traceCount = entries.filter(
      (entry) =>
        entry.isFile() &&
        (entry.name === "trace.jsonl" || entry.name.endsWith(".trace.jsonl")),
    ).length;
    const descend =
      depth > 0 && (!relative || relative.split("/").length < depth);
    await Promise.all(
      entries.map(async (entry) => {
        if (entry.name.startsWith(".")) return;
        const path = prefix + entry.name;
        const isRun =
          entry.isFile() &&
          (hasManifest
            ? entry.name === "manifest.json"
            : entry.name === "trace.jsonl" ||
              entry.name.endsWith(".trace.jsonl"));
        if (!isRun && !(descend && entry.isDirectory())) return;
        const info = await lstat(resolve(directory, path)).catch(() => null);
        if (info?.isDirectory() && !isRun)
          return queue.push({
            relative: path,
            level: path.split("/").length,
            mtime: info.mtimeMs,
          });
        if (!isRun || !info?.isFile() || info.nlink !== 1) return;
        runs.push({
          label: !hasManifest && traceCount > 1 ? path : relative || entry.name,
          trace: hasManifest ? null : `/results/${urlPath(path)}`,
          manifest: hasManifest ? `/results/${urlPath(path)}` : null,
          updated_at: info.mtime.toISOString(),
        });
      }),
    );
  }
  return {
    runs: runs.sort((a, b) => b.updated_at.localeCompare(a.updated_at)),
    truncated: queue.length > 0,
  };
}

const inlineHashes = (html, tag) =>
  [...html.matchAll(new RegExp(`<${tag}\\b[^>]*>([\\s\\S]*?)</${tag}>`, "g"))]
    .map(
      ([, text]) =>
        `'sha256-${createHash("sha256").update(text).digest("base64")}'`,
    )
    .join(" ") || "'none'";

// Hash the served bytes so the policy always matches the page.
export function contentSecurityPolicy(html) {
  // Browsers hash inline blocks after the parser normalizes newlines.
  html = html.replace(/\r\n?/g, "\n");
  return [
    "default-src 'none'",
    `script-src ${inlineHashes(html, "script")}`,
    `style-src ${inlineHashes(html, "style")}`,
    "connect-src 'self'",
    "img-src 'self' data:",
    "base-uri 'none'",
    "form-action 'none'",
    "frame-ancestors 'none'",
  ].join("; ");
}

// O_NOFOLLOW guards only the last component. macOS O_NOFOLLOW_ANY refuses a
// symlink anywhere in the path; Linux reports the opened file's real path.
// Elsewhere, comparing inodes after open narrows the race but cannot close it.
const openFlags =
  constants.O_RDONLY |
  constants.O_NONBLOCK |
  (process.platform === "darwin" ? 0x20000000 : constants.O_NOFOLLOW);

async function openedElsewhere(handle, file) {
  if (process.platform === "darwin") return false;
  if (process.platform === "linux")
    return (await readlink(`/proc/self/fd/${handle.fd}`)) !== file;
  const [opened, current] = await Promise.all([handle.stat(), lstat(file)]);
  return opened.dev !== current.dev || opened.ino !== current.ino;
}

// Ancestors must be host-controlled. Depth 1 reads only files in fixed job mounts.
export function createTraceServer(directory, { depth = 20 } = {}) {
  if (!Number.isInteger(depth) || depth < 0 || depth > 20)
    throw new Error("Trace directory depth must be 0–20");
  directory = resolve(directory);
  return createServer(async (request, response) => {
    const send = (status, type, body, headers) => {
      response.writeHead(status, {
        "content-type": type,
        "cache-control": "no-store",
        "x-content-type-options": "nosniff",
        "content-security-policy":
          "default-src 'none'; frame-ancestors 'none'; sandbox",
        ...headers,
      });
      response.end(request.method === "HEAD" ? undefined : body);
    };
    if (!/^(localhost|127\.0\.0\.1)(:\d+)?$/.test(request.headers.host || "")) {
      return send(403, "text/plain", "Forbidden");
    }
    if (!["GET", "HEAD"].includes(request.method))
      return send(405, "text/plain", "Read only");
    try {
      const path = decodeURIComponent(
        new URL(request.url, "http://localhost").pathname,
      );
      if (path.includes("\0")) return send(400, "text/plain", "Invalid path");
      if (path === "/") {
        response.writeHead(302, {
          location: "/docs/trace-viewer.html?index=/api/runs",
        });
        return response.end();
      }
      if (path === "/docs/trace-viewer.html") {
        const html = await readFile(viewer, "utf8");
        return send(200, "text/html; charset=utf-8", html, {
          "content-security-policy": contentSecurityPolicy(html),
        });
      }
      if (path === "/api/runs") {
        const { runs, truncated } = await listRuns(directory, depth);
        return send(
          200,
          "application/json",
          JSON.stringify(runs),
          truncated ? { "x-runs-truncated": String(MAX_DIRECTORIES) } : {},
        );
      }
      if (
        !path.startsWith("/results/") ||
        path.split("/").some((part) => part.startsWith(".")) ||
        path.slice("/results/".length).split("/").length > depth + 1
      ) {
        return send(404, "text/plain", "Not found");
      }
      const root = await realpath(directory);
      const requested = resolve(root, path.slice("/results/".length));
      const file = await realpath(requested);
      if (!file.startsWith(root + sep))
        return send(403, "text/plain", "Forbidden");
      if (file !== requested) return send(403, "text/plain", "Forbidden");
      const handle = await open(file, openFlags);
      try {
        if (await openedElsewhere(handle, file))
          return send(403, "text/plain", "Forbidden");
        const info = await handle.stat();
        if (!info.isFile() || info.nlink !== 1 || info.size > 16 * 1024 * 1024)
          return send(413, "text/plain", "Unsupported file");
        // Keep one descriptor and a bounded read if the file changes or grows.
        const bytes = Buffer.alloc(info.size);
        let offset = 0;
        while (offset < bytes.length) {
          const { bytesRead } = await handle.read(
            bytes,
            offset,
            bytes.length - offset,
            offset,
          );
          if (!bytesRead) break;
          offset += bytesRead;
        }
        let content = bytes.subarray(0, offset);
        // A live SDK trace can end between writes. Serve complete JSONL lines.
        if (file.endsWith(".jsonl"))
          content = content.subarray(0, content.lastIndexOf(10) + 1);
        // Artifacts are data, even when named .html or .js.
        return send(200, "text/plain; charset=utf-8", content);
      } finally {
        await handle.close();
      }
    } catch (error) {
      if (error.code === "ENOENT" || error.code === "ENOTDIR")
        return send(404, "text/plain", "Not found");
      if (error.code === "ELOOP") return send(403, "text/plain", "Forbidden");
      if (error instanceof URIError)
        return send(400, "text/plain", "Invalid path");
      console.error(error.message);
      return send(500, "text/plain", "Could not read trace");
    }
  });
}

if (
  process.argv[1] &&
  pathToFileURL(resolve(process.argv[1])).href === import.meta.url
) {
  const [directory = "results", option, depth] = process.argv.slice(2);
  if (
    (option !== undefined && option !== "--depth") ||
    (option && depth === undefined) ||
    process.argv.length > 5
  )
    throw new Error("Use: trace-viewer [directory] [--depth 0–20]");
  const port = Number(process.env.TRACE_VIEWER_PORT || 8765);
  if (!Number.isInteger(port) || port < 0 || port > 65535)
    throw new Error("Invalid TRACE_VIEWER_PORT");
  const server = createTraceServer(directory, {
    depth: depth === undefined ? 20 : Number(depth),
  });
  server.on("error", (error) => {
    console.error(error.message);
    process.exitCode = 1;
  });
  server.listen(port, "127.0.0.1", () => {
    console.log(`Trace viewer: http://127.0.0.1:${server.address().port}`);
    console.log(`Watching ${resolve(directory)}`);
  });
}
