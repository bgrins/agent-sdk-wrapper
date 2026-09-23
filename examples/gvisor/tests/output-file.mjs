import { closeSync, constants, fstatSync, openSync, readSync } from "node:fs";

// Agents control job output: never follow links, block on FIFOs or read unbounded files.
export function readOutputFile(path) {
  const fd = openSync(
    path,
    constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK,
  );
  try {
    const info = fstatSync(fd);
    if (!info.isFile() || info.nlink !== 1 || info.size > 16 * 1024 * 1024)
      throw new Error(`Unsupported output file: ${path}`);
    const bytes = Buffer.alloc(info.size);
    let offset = 0;
    while (offset < bytes.length) {
      const read = readSync(fd, bytes, offset, bytes.length - offset, offset);
      if (!read) break;
      offset += read;
    }
    return bytes.subarray(0, offset).toString("utf8");
  } finally {
    closeSync(fd);
  }
}
