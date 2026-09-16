#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."

case ${1:-sdk} in
  sdk) worker=("${LANGUAGE:?Set LANGUAGE}-agent") ;;
  security) worker=(test-worker --test tests/security.test.mjs) ;;
  fixture) worker=(test-worker tests/fixture-worker.mjs) ;;
  *) echo 'Use: tests/run.sh [sdk|security|fixture]' >&2; exit 1 ;;
esac
export PROVIDER=${PROVIDER:-anthropic}
case $PROVIDER in
  anthropic) model=claude-haiku-4-5 ;;
  codex|openai) export PROVIDER=openai; model=gpt-5.6-luna ;;
  *) echo 'Invalid test provider' >&2; exit 1 ;;
esac
export UPSTREAM_KEY=outer-only-canary GVISOR_MODEL=$model
export GVISOR_OUTPUT_DIR=${GVISOR_OUTPUT_DIR:-results/gvisor-output/tests}
export COMPOSE_FILE="$PWD/examples/gvisor/compose.yaml:$PWD/examples/gvisor/tests/compose.yaml"
exec bash examples/gvisor/infra/run-job.sh "${TEST_TIMEOUT_SECONDS:-180}" "${worker[@]}"
