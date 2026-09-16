import { closeSync, mkdirSync, openSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { TraceWriteError } from "./errors.js";
import type { EventEnvelope } from "./events.js";

export class TraceWriter {
  private readonly fd: number;

  constructor(private readonly path: string) {
    try {
      mkdirSync(dirname(path), { recursive: true });
      this.fd = openSync(path, "w");
    } catch (cause) {
      throw new TraceWriteError(`Cannot open trace file '${path}'`, { cause });
    }
  }

  write(envelope: EventEnvelope): void {
    try {
      writeFileSync(this.fd, `${JSON.stringify(envelope)}\n`);
    } catch (cause) {
      throw new TraceWriteError(`Cannot write trace file '${this.path}'`, {
        cause,
      });
    }
  }

  close(): void {
    try {
      closeSync(this.fd);
    } catch (cause) {
      throw new TraceWriteError(`Cannot close trace file '${this.path}'`, {
        cause,
      });
    }
  }
}
