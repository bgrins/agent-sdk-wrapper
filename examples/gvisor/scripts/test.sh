#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:---offline}
[[ $mode == --offline || $mode == --live ]] || { echo 'Use --offline or --live' >&2; exit 1; }
for target in gateway-tests checks; do
  docker build -f examples/gvisor/infra/Dockerfile --target "$target" \
    -t "agent-sdk-wrapper-gvisor-$target" .
done
docker run --rm --network none --read-only --tmpfs /tmp:rw,nosuid,nodev,size=128m \
  --cap-drop ALL --security-opt no-new-privileges \
  --entrypoint node agent-sdk-wrapper-gvisor-gateway-tests --test --test-timeout=15000 /example/tests/gateway.test.mjs
node_args=(examples/gvisor/tests/smoke.mjs "$mode")
if [[ $mode == --live && -f .env ]]; then node_args=(--env-file=.env "${node_args[@]}"); fi
mkdir -p results/gvisor-checks
for language in python typescript; do
  for provider in anthropic openai; do
    LANGUAGE=$language PROVIDER=$provider node "${node_args[@]}" \
      > "results/gvisor-checks/$language-$provider-${mode#--}.log"
  done
done
if [[ $mode == --offline ]]; then
  for provider in anthropic openai; do
    PROVIDER=$provider node examples/gvisor/tests/smoke.mjs --security \
      > "results/gvisor-checks/$provider-security.log"
  done
  node examples/gvisor/tests/smoke.mjs --fixture > results/gvisor-checks/fixture.log
  node examples/gvisor/tests/smoke.mjs --bad-patch > results/gvisor-checks/fixture-bad-patch.log
  node examples/gvisor/tests/smoke.mjs --missing-output > results/gvisor-checks/fixture-missing-output.log
  node --test examples/gvisor/tests/output-file.test.mjs examples/gvisor/tests/lifecycle.test.mjs
fi
