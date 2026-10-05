#!/bin/bash

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Rewrite a wheel's metadata version without touching compiled binaries.
#
# Usage:
# bash scripts/restamp_wheel.sh WHEEL VERSION OUTPUT_DIR
#
# Requires the `wheel` package to be importable by $PYTHON (default python3).

set -euo pipefail

wheel=$1 version=$2 out=$3
python=${PYTHON:-python3}

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

"$python" -m wheel unpack "$wheel" -d "$tmp"
info=$(echo "$tmp"/*/*.dist-info)
old=$(sed -n 's/^Version: //p' "$info/METADATA" | head -1)

sed -e "s/^Version: .*/Version: $version/" \
    -e "s/^\(Requires-Dist:.*\)==$old/\1==$version/" \
    "$info/METADATA" > "$info/METADATA.new"
mv "$info/METADATA.new" "$info/METADATA"
[ "$old" = "$version" ] || mv "$info" "${info%-$old.dist-info}-$version.dist-info"

mkdir -p "$out"
"$python" -m wheel pack "$(dirname "$info")" -d "$out"
