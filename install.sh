#!/usr/bin/env bash
# Install hermes-acp-agents into a Hermes install's plugin directory.
#
# Copies (default) or symlinks (--link) each providers/<name>-acp/ into
# $HERMES_HOME/plugins/model-providers/<name>-acp/, where providers/__init__.py's
# discovery picks it up automatically — no Hermes core changes needed.
set -euo pipefail

link=false
if [[ "${1:-}" == "--link" ]]; then
  link=true
fi

hermes_home="${HERMES_HOME:-$HOME/.hermes}"
target="$hermes_home/plugins/model-providers"
src="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/providers"

mkdir -p "$target"

for dir in "$src"/*/; do
  name="$(basename "$dir")"
  dest="$target/$name"
  if [[ -e "$dest" || -L "$dest" ]]; then
    echo "skip: $dest already exists (remove it first to reinstall)" >&2
    continue
  fi
  if $link; then
    ln -s "$dir" "$dest"
    echo "linked $name -> $dest"
  else
    cp -r "$dir" "$dest"
    echo "copied $name -> $dest"
  fi
done

echo
echo "Done. Restart Hermes (or the gateway container) so providers/ re-discovers plugins."
