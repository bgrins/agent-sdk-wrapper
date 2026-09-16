#!/usr/bin/env bash
set -euo pipefail
[[ $(uname -s) == Linux ]] || { echo 'Run this on the Linux Docker host.' >&2; exit 1; }
case "$(uname -m)" in
  x86_64|aarch64) arch=$(uname -m) ;;
  *) echo 'Unsupported architecture' >&2; exit 1 ;;
esac
version=20260831.0
download_dir=$(mktemp -d)
trap 'rm -rf "$download_dir"' EXIT
cd "$download_dir"
base="https://storage.googleapis.com/gvisor/releases/release/$version/$arch"
curl --fail --silent --show-error --location --proto '=https' -O "$base/gvisor.tar.bz2"
curl --fail --silent --show-error --location --proto '=https' -O "$base/gvisor.tar.bz2.sha512"
sha512sum -c gvisor.tar.bz2.sha512
# Install the whole bundle: runsc needs its sibling gvisor-bin directory.
sudo tar -xjf gvisor.tar.bz2 -C /usr/local/bin
sudo /usr/local/bin/runsc install --runtime=runsc-agent -- --host-uds=open
sudo systemctl reload docker
/usr/local/bin/runsc --version
