#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Fetches the sources for the package lists in $SCRIPTS_DIR into $SOURCES_ROOT:
#   <variant>/apt   source packages (.dsc, orig and debian tarballs) at the exact
#                   versions the image ships; <variant> is cudaq, cudaqx or
#                   cudaq-decoding
#   <variant>/pip   sdists; <variant> is cudaq, cudaqx, cudaq-decoding or macos
#   tpls            the tpls submodules at the commits in the lock file
# The fetches run in parallel with prefixed output. Packages without obtainable
# source are listed in <variant>/{apt,pip}/*omitted_packages.txt; whether that
# is acceptable is decided by check_package_sources.sh. A failure to clone tpls
# is an error here.
#
# Expects in $SCRIPTS_DIR (see docker/build/package_sources.Dockerfile):
#   apt_sources_<variant>_trimmed.txt  source=source_version lines
#   pip_packages_<variant>_trimmed.txt package==version lines
#   pip_source_aliases.txt             "<wheel-only name> <name of the project with the sdist>"
#   tpls_commits.lock, .gitmodules, clone_tpls_from_lock.sh

set -uo pipefail

: "${SOURCES_ROOT:?SOURCES_ROOT not set}"
: "${SCRIPTS_DIR:?SCRIPTS_DIR not set}"

for tool in unearth jq curl git pull-lp-source apt-get; do
  command -v "$tool" > /dev/null || { echo "ERROR: $tool is required" >&2; exit 1; }
done

retry() {
  for n in 1 2 3; do
    "$@" && return 0
    [ "$n" -lt 3 ] && sleep $((10 * n))
  done
  return 1
}

# apt-get source needs a writable directory for the _apt sandbox user.
fetch_apt() {
  local variant=$1 dir="$SOURCES_ROOT/$1/apt" list="$SCRIPTS_DIR/apt_sources_$1_trimmed.txt"
  mkdir -p "$dir" && chmod 777 "$dir" && cd "$dir" || return 1
  : > apt_omitted_packages.txt
  [ -f "$list" ] || return 0
  local spec src ver
  while IFS= read -r spec || [ -n "$spec" ]; do
    [ -z "$spec" ] && continue
    src=${spec%%=*}
    ver=${spec#*=}
    # The exact installed version first, from the current archive; Launchpad
    # keeps the versions that the archive has since superseded.
    if retry apt-get source --download-only -y "$spec" ||
       pull-lp-source -d "$src" "$ver"; then
      echo "fetched $spec"
    else
      echo "$spec" >> apt_omitted_packages.txt
      echo "no source for $spec"
    fi
  done < "$list"
}

# Fetches the sdist of name==version, or of the project it is an alias of when
# the name only ships wheels (e.g. cupy-cuda12x is built from the cupy sdist).
fetch_pip() {
  local variant=$1 dir="$SOURCES_ROOT/$1/pip" list="$SCRIPTS_DIR/pip_packages_$1_trimmed.txt"
  local omitted=pip_omitted_packages.txt
  [ "$variant" = macos ] && omitted=macos_pip_omitted_packages.txt
  mkdir -p "$dir" && cd "$dir" || return 1
  : > "$omitted"
  [ -f "$list" ] || return 0
  local package name version alias url
  while IFS= read -r package || [ -n "$package" ]; do
    [ -z "$package" ] && continue
    name=${package%%==*}
    version=${package#*==}
    url=$(unearth --no-binary "$package" 2>/dev/null | jq -r '.link.url')
    if [ -z "$url" ] || [ "$url" = null ]; then
      alias=$(awk -v n="$name" '$1 == n {print $2; exit}' "$SCRIPTS_DIR/pip_source_aliases.txt" 2>/dev/null)
      if [ -n "$alias" ]; then
        url=$(unearth --no-binary "$alias==$version" 2>/dev/null | jq -r '.link.url')
      fi
    fi
    if [ -n "$url" ] && [ "$url" != null ] && retry curl -fsSL -O "$url"; then
      echo "fetched $package"
    else
      echo "$package" >> "$omitted"
      echo "no sdist for $package"
    fi
  done < "$list"
}

fetch_tpls() {
  GITMODULES="$SCRIPTS_DIR/.gitmodules" lock_file="$SCRIPTS_DIR/tpls_commits.lock" \
    bash "$SCRIPTS_DIR/clone_tpls_from_lock.sh"
}

mkdir -p "$SOURCES_ROOT/tpls"
pids=()
names=()
start() {
  local name=$1
  shift
  ( set -o pipefail; "$@" 2>&1 | sed "s/^/[$name] /"; exit "${PIPESTATUS[0]}" ) &
  pids+=($!)
  names+=("$name")
}

for variant in cudaq cudaqx cudaq-decoding; do
  start "$variant-apt" fetch_apt "$variant"
  start "$variant-pip" fetch_pip "$variant"
done
start macos-pip fetch_pip macos
start tpls fetch_tpls

failed=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "ERROR: fetching ${names[$i]} failed" >&2
    failed=1
  fi
done
exit $failed
