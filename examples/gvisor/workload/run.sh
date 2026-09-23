#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."

fail() { echo "$*" >&2; exit 1; }
if [[ ${1:-} == cleanup && $# == 1 ]]; then
  exec bash examples/gvisor/infra/run-job.sh cleanup
fi
[[ $# == 0 ]] || fail 'Use: run.sh [cleanup]'
case ${LANGUAGE:-} in
  python|typescript) ;;
  *) fail 'Set LANGUAGE=python or typescript' ;;
esac
export PROVIDER=${PROVIDER:-anthropic}
case $PROVIDER in
  anthropic) key=${ANTHROPIC_API_KEY:-}; model=claude-haiku-4-5 ;;
  codex|openai) export PROVIDER=openai; key=${OPENAI_API_KEY:-}; model=gpt-6-luna ;;
  *) fail 'Use PROVIDER=anthropic or codex' ;;
esac
[[ -n $key ]] || fail "Missing $PROVIDER API key"
export UPSTREAM_KEY=$key GVISOR_MODEL=${GVISOR_MODEL:-$model}
export COMPOSE_FILE="$PWD/examples/gvisor/compose.yaml"
exec bash examples/gvisor/infra/run-job.sh 180 "$LANGUAGE-agent"
