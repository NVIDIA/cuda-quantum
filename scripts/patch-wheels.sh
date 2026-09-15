#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                               #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Renames the version on locally-built cudaq release artifacts (metapackage
# sdist + cuda-quantum-cu12/cu13 wheels) without changing any content, so
# PyPI accepts a distinguishable re-upload. Does not touch cudaq-logical.
# Does not upload anything -- see patch-wheels-README.md for that step,
# done by hand.
set -euo pipefail

# --- Set these ---
ORIG_VER=0.16.0
NEW_VER=0.16.0.post1        # PEP 440 post-release segment; PyPI rejects
                             # anything else (e.g. "0.16.0.patch1" is invalid).
SRC_DIR="$HOME/wheels"       # unmodified GH Actions artifact zips, see README
OUT_DIR="$HOME/wheels_patched"

VENV_DIR=$(mktemp -d)
python3 -m venv "$VENV_DIR" --clear
"$VENV_DIR/bin/pip" install --quiet --upgrade pip build wheel
PY="$VENV_DIR/bin/python3"

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR" "$VENV_DIR"' EXIT

mkdir -p "$OUT_DIR/aarch64" "$OUT_DIR/x86_64" "$OUT_DIR/sdist"

# --- cudaq metapackage sdist ---
work="$TMP_DIR/metapkg"
mkdir -p "$work"
unzip -q "$SRC_DIR/cudaq-metapackage-${ORIG_VER}.zip" -d "$work"
tar -xzf "$work/cudaq-${ORIG_VER}.tar.gz" -C "$work"
src="$work/cudaq-${ORIG_VER}"
echo "$NEW_VER" > "$src/_version.txt"
# The existing sdist's PKG-INFO is static and takes precedence over
# _version.txt's dynamic source unless removed first -- otherwise the
# rebuild silently keeps the old version with no error.
rm -f "$src/PKG-INFO"
( cd "$src" && CUDAQ_META_SDIST_BUILD=1 "$PY" -m build . --sdist --outdir "$OUT_DIR/sdist" )
echo "sdist -> $OUT_DIR/sdist/cudaq-${NEW_VER}.tar.gz"

# --- cuda-quantum-cuXX wheels, per architecture ---
for arch in aarch64 x86_64; do
  for zip in "$SRC_DIR/${arch}"-cu*-py*-wheels.zip; do
    [ -e "$zip" ] || continue
    work="$TMP_DIR/$(basename "$zip" .zip)"
    mkdir -p "$work"
    unzip -q "$zip" -d "$work"
    whl=$(ls "$work"/*.whl)

    "$PY" -m wheel unpack "$whl" -d "$work/unpacked" >/dev/null

    orig_dir=$(find "$work/unpacked" -mindepth 1 -maxdepth 1 -type d)
    pkgname=$(basename "$orig_dir" | sed "s/-${ORIG_VER}\$//")
    new_dir="$work/unpacked/${pkgname}-${NEW_VER}"
    mv "$orig_dir" "$new_dir"
    mv "$new_dir/${pkgname}-${ORIG_VER}.dist-info" "$new_dir/${pkgname}-${NEW_VER}.dist-info"
    sed -i "s/^Version: ${ORIG_VER}\$/Version: ${NEW_VER}/" \
      "$new_dir/${pkgname}-${NEW_VER}.dist-info/METADATA"

    "$PY" -m wheel pack "$new_dir" -d "$OUT_DIR/$arch" >/dev/null
    echo "$(basename "$whl") -> $arch/$(basename "$(ls "$OUT_DIR/$arch"/"${pkgname}"-"${NEW_VER}"*.whl | tail -1)")"
  done
done

echo
echo "Done. Contents of $OUT_DIR:"
find "$OUT_DIR" -type f | sort
