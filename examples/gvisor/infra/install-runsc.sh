#!/usr/bin/env bash
set -euo pipefail
[[ $(uname -s) == Linux ]] || { echo 'Run this on the Linux Docker host.' >&2; exit 1; }
version=20260831.0
arch=$(uname -m)
case $arch in
  x86_64) sha512=c7b4f22c60be3ea5a0ace478ed390403255d818729aaccff82d5b60026f3060d2ea31ab6670249a34a98eef59726217018c91ec631f7d753459127d930658392 ;;
  aarch64) sha512=9e60dd883aaa9aa341f4708c9de3d0e34b91fb5c4f62534520e636fa509cb0401116428497a2324f69003b88fdabadf195e8bb66c00de3baed07a9fa95583b90 ;;
  *) echo 'Unsupported architecture' >&2; exit 1 ;;
esac
download_dir=$(mktemp -d)
trap 'rm -rf "$download_dir"' EXIT
cd "$download_dir"
base="https://storage.googleapis.com/gvisor/releases/release/$version/$arch"
curl --fail --silent --show-error --location --proto '=https' -O "$base/gvisor.tar.bz2"
echo "$sha512  gvisor.tar.bz2" | sha512sum -c -
# Install the whole bundle: runsc needs its sibling gvisor-bin directory.
sudo tar -xjf gvisor.tar.bz2 -C /usr/local/bin
sudo /usr/local/bin/runsc install --runtime=runsc-agent -- --host-uds=open
sudo systemctl reload docker
/usr/local/bin/runsc --version
