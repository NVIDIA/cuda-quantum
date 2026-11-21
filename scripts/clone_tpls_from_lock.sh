# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
#
# Clones each tpl submodule at the commit given in the lock file into
# SOURCES_ROOT/tpls/<lib>. Uses git clone --no-checkout --filter=tree:0
# then fetch + checkout for minimal history, and initializes nested submodules
# (e.g. nanobind's robin_map). Exits non-zero if any library could not be cloned.
#
# Usage: SOURCES_ROOT=/sources .gitmodules=/path/.gitmodules lock_file=/path/tpls_lock.txt \
#        ./scripts/clone_tpls_from_lock.sh

set -euo pipefail

: "${SOURCES_ROOT:?SOURCES_ROOT not set}"
: "${GITMODULES:?GITMODULES not set}"
: "${lock_file:?lock_file not set}"

[ -f "$lock_file" ] || { echo "Lock file not found: $lock_file" >&2; exit 1; }

mkdir -p "${SOURCES_ROOT}/tpls"
cd "${SOURCES_ROOT}/tpls"
failures=0

while IFS= read -r line || [ -n "$line" ]; do
  [ -z "$line" ] && continue
  commit="${line%% *}"
  path="${line#* }"
  [ -z "$commit" ] || [ -z "$path" ] && continue

  repo="$(git config --file "$GITMODULES" --get "submodule.${path}.url")" || {
    echo "ERROR: no url for $path" >&2
    failures=$((failures + 1))
    continue
  }
  lib="$(basename "$path")"
  dest="${SOURCES_ROOT}/tpls/${lib}"

  echo "Cloning $lib @ $commit from $repo ..."
  git clone --no-checkout --filter=tree:0 "$repo" "$dest" \
    && git -C "$dest" fetch --depth 1 origin "$commit" \
    && git -C "$dest" checkout --detach FETCH_HEAD \
    && git -C "$dest" submodule update --init --recursive --depth 1 \
    || { echo "Failed to clone $lib" >&2; rm -rf "$dest" 2>/dev/null; failures=$((failures + 1)); }
done < "$lock_file"

[ "$failures" -eq 0 ] || { echo "ERROR: $failures tpls could not be cloned" >&2; exit 1; }
