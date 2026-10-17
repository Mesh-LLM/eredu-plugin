#!/bin/bash
# Build the GitHub-installable archive: eredu-plugin-<version>-aarch64-apple-darwin.tar.gz
set -euo pipefail
version=${1:?usage: package-macos.sh <version, e.g. v0.1.7>}
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
[[ $(uname -s) == Darwin && $(uname -m) == arm64 ]]
cargo build --locked --release --features mlx
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
pkg="$stage/eredu-plugin"
mkdir -p "$pkg" dist
cp target/release/eredu-plugin README.md LICENSE "$pkg/"
sed "s/^version = .*/version = \"${version#v}\"/" plugin.toml > "$pkg/plugin.toml"
codesign --force --sign - "$pkg/eredu-plugin"
codesign --verify --strict "$pkg/eredu-plugin"
# Must be self-contained: only system libraries/frameworks.
if otool -L "$pkg/eredu-plugin" | tail -n +2 | grep -vE '^\s*(/usr/lib/|/System/Library/)'; then
  echo "non-system dynamic dependency found" >&2; exit 1
fi
"$pkg/eredu-plugin" --version
for name in "eredu-plugin-$version-aarch64-apple-darwin.tar.gz" "eredu-plugin-aarch64-apple-darwin.tar.gz"; do
  tar -czf "dist/$name" -C "$stage" eredu-plugin
  (cd dist && shasum -a 256 "$name" > "$name.sha256")
  echo "dist/$name"
done
