#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
for target in gateway python typescript; do
  docker build -f examples/gvisor/infra/Dockerfile --target "$target" \
    -t "agent-sdk-wrapper-gvisor-$target" .
done
