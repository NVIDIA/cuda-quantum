#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Fails if a package that the image ships has neither its source in
# $SOURCES_ROOT nor an accepted reason to have none:
#   - it is proprietary and listed in docker/build/package_sources_proprietary.txt
#     (apt and pip; its license files are kept instead), or
#   - its source is part of a DOCA/OFED source bundle downloaded to
#     $SOURCES_ROOT/prereqs/doca-ofed-src* (apt only: rdma-core, ucx, xpmem, knem
#     and the other Mellanox packages that replace Ubuntu's), or
#   - the DOCA <source>-dkms package of exactly that version, which holds the
#     full source tree (mlnx-ofed-kernel), is in the same directories.
# Also checks that every apt source package that was not omitted is on disk at
# the exact version, so that a fetch that quietly got another version fails here.
#
# Expects in $SCRIPTS_DIR the lists used by fetch_package_sources.sh and
# package_sources_proprietary.txt.

set -uo pipefail

: "${SOURCES_ROOT:?SOURCES_ROOT not set}"
: "${SCRIPTS_DIR:?SCRIPTS_DIR not set}"

proprietary="$SCRIPTS_DIR/package_sources_proprietary.txt"
failures=0

is_proprietary() { # <apt|pip> <name>
  local kind=$1 name=$2 glob
  while read -r k glob; do
    [ "$k" = "$kind" ] || continue
    # shellcheck disable=SC2254
    case "$name" in $glob) return 0 ;; esac
  done < <(grep -vE '^\s*(#|$)' "$proprietary")
  return 1
}

# Files of the DOCA/OFED source bundle, e.g. rdma-core_2507mlnx58.orig.tar.gz.
ofed_files=""
for bundle in "$SOURCES_ROOT"/prereqs/doca-ofed-src*/MLNX_OFED_SRC-*.tgz; do
  [ -f "$bundle" ] && ofed_files+=$'\n'"$(tar -tzf "$bundle" | xargs -n1 basename)"
done

# 1:2.3-4build1 -> 2.3, 2507mlnx58-1.2507097 -> 2507mlnx58
upstream_version() {
  local v=${1#*:}
  [[ $v == *-* ]] && v=${v%-*}
  echo "$v"
}

for variant in cudaq cudaqx cudaq-decoding; do
  dir="$SOURCES_ROOT/$variant/apt"
  list="$SCRIPTS_DIR/apt_sources_${variant}_trimmed.txt"
  [ -f "$list" ] || continue
  omitted="$dir/apt_omitted_packages.txt"
  while IFS= read -r spec || [ -n "$spec" ]; do
    [ -z "$spec" ] && continue
    src=${spec%%=*}
    ver=${spec#*=}
    if grep -qxF "$spec" "$omitted" 2>/dev/null; then
      if is_proprietary apt "$src"; then
        echo "ok   $variant apt $spec: proprietary, license files only"
      elif awk -v p="${src}_$(upstream_version "$ver").orig.tar" 'index($0, p) == 1 {f = 1} END {exit !f}' <<< "$ofed_files"; then
        echo "ok   $variant apt $spec: source is in the DOCA/OFED bundle"
      elif ls "$SOURCES_ROOT"/prereqs/doca-ofed-src*/"${src}-dkms_${ver}_all.deb" > /dev/null 2>&1; then
        echo "ok   $variant apt $spec: source is in the DOCA ${src}-dkms package"
      else
        echo "FAIL $variant apt $spec: no source and not on the proprietary list"
        failures=$((failures + 1))
      fi
    elif ! ls "$dir/${src}_${ver#*:}.dsc" > /dev/null 2>&1; then
      echo "FAIL $variant apt $spec: ${src}_${ver#*:}.dsc was not fetched"
      failures=$((failures + 1))
    fi
  done < "$list"
done

for variant in cudaq cudaqx cudaq-decoding macos; do
  omitted="$SOURCES_ROOT/$variant/pip/pip_omitted_packages.txt"
  [ "$variant" = macos ] && omitted="$SOURCES_ROOT/macos/pip/macos_pip_omitted_packages.txt"
  [ -s "$omitted" ] || continue
  while IFS= read -r package || [ -n "$package" ]; do
    [ -z "$package" ] && continue
    if is_proprietary pip "${package%%==*}"; then
      echo "ok   $variant pip $package: proprietary, license files only"
    else
      echo "FAIL $variant pip $package: no sdist and not on the proprietary list"
      failures=$((failures + 1))
    fi
  done < "$omitted"
done

if [ "$failures" -ne 0 ]; then
  echo "ERROR: $failures package(s) ship without source; fix the fetch or list them in" \
       "docker/build/package_sources_proprietary.txt if they are proprietary." >&2
  exit 1
fi
echo "All packages have their source or an accepted reason to have none."
