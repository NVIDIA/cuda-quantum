#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Builds an architecture-independent metapackage deb that selects a CUDA
# variant via Debian alternatives.
#
# Usage:
#   bash scripts/build_deb_metapackage.sh -V 0.14.0 -o out             # cudaq
#   bash scripts/build_deb_metapackage.sh -V 0.14.0 -f realtime -o out # cudaq-realtime
#
# Options:
#   -f <flavor>    core (emits cudaq) or realtime (emits cudaq-realtime).
#                  Defaults to core.
#   -V <version>   Version string. Must match the version of the variant
#                  debs so the pinned "(= <version>)" Depends resolves.
#   -o <out_dir>   Output directory. Defaults to "out".
#   -v             Verbose.

set -euo pipefail

flavor="core"
cudaq_version=""
output_dir="out"
verbose=false

__optind__=$OPTIND
OPTIND=1
while getopts ":f:V:o:v" opt; do
  case $opt in
    f) flavor="$OPTARG" ;;
    V) cudaq_version="$OPTARG" ;;
    o) output_dir="$OPTARG" ;;
    v) verbose=true ;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
    :)  echo "Option -$OPTARG requires an argument" >&2; exit 1 ;;
  esac
done
OPTIND=$__optind__

if [ -z "$cudaq_version" ]; then
  echo "Error: version required (-V <version>)" >&2
  exit 1
fi

case "$flavor" in
  core)
    meta_name="cudaq"
    variant_a="cuda-quantum-cu13"
    variant_b="cuda-quantum-cu12"
    description_short="NVIDIA CUDA-Q (meta-package)"
    description_long=" Installs the CUDA-Q toolkit. Pulls in one of
 cuda-quantum-cu13 or cuda-quantum-cu12, preferring cu13 when both are
 available. To force a specific CUDA variant, install it explicitly
 alongside this metapackage."
    ;;
  realtime)
    meta_name="cudaq-realtime"
    variant_a="cuda-quantum-realtime-cu13"
    variant_b="cuda-quantum-realtime-cu12"
    description_short="NVIDIA CUDA-Q realtime (meta-package)"
    description_long=" Installs the CUDA-Q realtime toolkit. Pulls in one
 of cuda-quantum-realtime-cu13 or cuda-quantum-realtime-cu12, preferring
 cu13 when both are available. Coexists with the non-realtime
 cuda-quantum-cu* packages."
    ;;
  *)
    echo "Error: flavor must be 'core' or 'realtime', got: $flavor" >&2
    exit 1
    ;;
esac

command -v dpkg-deb >/dev/null 2>&1 || {
  echo "Error: dpkg-deb not found on PATH" >&2
  exit 1
}

mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"

work_dir="$(mktemp -d -t cudaq-meta.XXXXXX)"
trap 'rm -rf "$work_dir"' EXIT

pkgroot="$work_dir/pkgroot"
mkdir -p "$pkgroot/DEBIAN" "$pkgroot/usr/share/doc/${meta_name}"

# Debian policy still wants copyright + changelog even for empty
# metapackages. Keep them minimal.
{
  echo "Source: https://github.com/NVIDIA/cuda-quantum"
  echo "Upstream-Name: CUDA-Q"
  echo
  echo "Files: *"
  echo "Copyright: 2022 - $(date +%Y) NVIDIA Corporation & Affiliates."
  echo "License: Apache-2.0"
} > "$pkgroot/usr/share/doc/${meta_name}/copyright"
chmod 0644 "$pkgroot/usr/share/doc/${meta_name}/copyright"

changelog="$pkgroot/usr/share/doc/${meta_name}/changelog.Debian"
{
  echo "${meta_name} (${cudaq_version}) unstable; urgency=medium"
  echo
  echo "  * Automated metapackage build for CUDA-Q ${cudaq_version}."
  echo
  echo " -- NVIDIA Corporation <cuda-quantum@nvidia.com>  $(date -R 2>/dev/null || date -u '+%a, %d %b %Y %H:%M:%S +0000')"
} > "$changelog"
gzip -9n "$changelog"
chmod 0644 "${changelog}.gz"

installed_size=$(du -sk --exclude=DEBIAN "$pkgroot" | cut -f1)

cat > "$pkgroot/DEBIAN/control" << CONTROL
Package: ${meta_name}
Version: ${cudaq_version}
Architecture: all
Maintainer: NVIDIA Corporation <cuda-quantum@nvidia.com>
Installed-Size: ${installed_size}
Depends: ${variant_a} (= ${cudaq_version}) | ${variant_b} (= ${cudaq_version})
Section: metapackages
Priority: optional
Homepage: https://developer.nvidia.com/cuda-q
Description: ${description_short}
${description_long}
CONTROL
chmod 0644 "$pkgroot/DEBIAN/control"

find "$pkgroot" -type d -exec chmod 0755 {} +

deb_file="${output_dir}/${meta_name}_${cudaq_version}_all.deb"
if $verbose; then
  dpkg-deb --build --root-owner-group -Zxz "$pkgroot" "$deb_file"
else
  dpkg-deb --build --root-owner-group -Zxz "$pkgroot" "$deb_file" >/dev/null
fi

echo "Built $deb_file"

if command -v lintian >/dev/null 2>&1; then
  lintian --no-tag-display-limit "$deb_file" || true
fi
