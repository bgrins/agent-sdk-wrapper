#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."

fail() { echo "$*" >&2; exit 1; }
if [[ ${1:-} == cleanup && $# == 1 ]]; then
  docker ps -aq --filter label=agent-sdk-wrapper.gvisor-example=1 |
    while IFS= read -r id; do docker rm -f "$id"; done
  docker volume ls -q --filter label=agent-sdk-wrapper.gvisor-example=1 |
    while IFS= read -r id; do docker volume rm "$id"; done
  exit
fi
[[ $# -ge 2 ]] || fail 'Use: run-job.sh seconds service [command...]'
: "${COMPOSE_FILE:?Set COMPOSE_FILE}"
: "${UPSTREAM_KEY:?Set UPSTREAM_KEY}"
limit=$1
shift
[[ $limit =~ ^[1-9][0-9]*$ && ${#limit} -le 3 && $limit -le 180 ]] || fail 'Job timeout must be 1–180 seconds'
docker info --format '{{range $name, $_ := .Runtimes}}{{println $name}}{{end}}' |
  grep -qx runsc-agent || fail 'Install runsc-agent; no fallback runtime is allowed'

random() { od -An -N16 -tx1 /dev/urandom | tr -d ' \n'; }
COMPOSE_PROJECT_NAME=agent-example-$(random)
GATEWAY_TOKEN=$(random)
export COMPOSE_PROJECT_NAME GATEWAY_TOKEN
root=${GVISOR_OUTPUT_DIR:-results/gvisor-output}
mkdir -p "$root"
GVISOR_RUN_OUTPUT="$(cd "$root" && pwd -P)/$COMPOSE_PROJECT_NAME"
export GVISOR_RUN_OUTPUT
mkdir "$GVISOR_RUN_OUTPUT"
# The worker's uid may differ from the host's; the parent stays host-controlled.
chmod 0777 "$GVISOR_RUN_OUTPUT"
echo "Output: $GVISOR_RUN_OUTPUT" >&2
compose=(docker compose)
cli= timer=
cleanup() {
  status=$?
  trap - EXIT
  trap '' INT TERM USR1
  if [[ -n $timer ]]; then kill "$timer" 2>/dev/null || :; wait "$timer" 2>/dev/null || :; fi
  if [[ -n $cli ]]; then kill -KILL "$cli" 2>/dev/null || :; wait "$cli" 2>/dev/null || :; fi
  # Stopping the attached CLI alone does not stop its container.
  docker rm -f "$COMPOSE_PROJECT_NAME-agent" >/dev/null 2>&1 || :
  if ! "${compose[@]}" --profile '*' down --volumes --remove-orphans --timeout 2; then
    echo 'Cleanup failed; run: bash examples/gvisor/workload/run.sh cleanup' >&2
    status=1
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'echo "Job deadline exceeded" >&2; exit 124' USR1
(
  trap 'kill "${sleeper:-}" 2>/dev/null || :; wait "${sleeper:-}" 2>/dev/null || :; exit' TERM INT
  sleep "$limit" & sleeper=$!
  wait "$sleeper"
  kill -USR1 "$$"
) & timer=$!

# Background commands let Bash handle signals while waiting for Docker.
"${compose[@]}" up -d --wait --wait-timeout 10 --no-deps gateway >&2 & cli=$!
wait "$cli"
cli=
"${compose[@]}" run --rm -T --no-deps --name "$COMPOSE_PROJECT_NAME-agent" "$@" & cli=$!
wait "$cli"
cli=
