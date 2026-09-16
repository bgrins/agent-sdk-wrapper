import ms from "ms";

export function deadlineAfter(start, duration) {
  const milliseconds = ms(duration);
  if (typeof milliseconds !== "number" || milliseconds < 0) {
    throw new TypeError("Expected a nonnegative duration");
  }
  return new Date(start.getTime() + milliseconds / 1000);
}
