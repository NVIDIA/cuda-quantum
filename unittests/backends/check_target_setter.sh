#!/usr/bin/env bash
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
#
# Usage: check_target_setter.sh <nvq++_path> <docs_cpp_dir>

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <nvq++_path> <docs_cpp_dir>" >&2
  exit 2
fi

NVQPP="$1"
DOCS_CPP_DIR="$2"

if [[ ! -x "$NVQPP" ]]; then
  echo "nvq++ not found or not executable: $NVQPP" >&2
  exit 2
fi

# (target, source-relative-path) pairs to check. Extend when new
# `gen-target-backend: true` analog targets land an example here.
EXAMPLES=(
  "pasqal pasqal.cpp"
  "quera  quera_basic.cpp"
  "quera  quera_intro.cpp"
)

TMPDIR="$(mktemp -d -t target_setter_check_XXXX)"
trap 'rm -rf "$TMPDIR"' EXIT

failures=0
for entry in "${EXAMPLES[@]}"; do
  read -r target src <<<"$entry"
  src_path="$DOCS_CPP_DIR/$src"
  out="$TMPDIR/${src%.cpp}.out"

  if [[ ! -f "$src_path" ]]; then
    echo "FAIL: source not found: $src_path"
    failures=$((failures + 1))
    continue
  fi

  if ! "$NVQPP" --target "$target" "$src_path" -o "$out" >"$TMPDIR/$src.log" 2>&1; then
    echo "FAIL: $src failed to compile with --target $target"
    cat "$TMPDIR/$src.log" >&2
    failures=$((failures + 1))
    continue
  fi

  # Capture nm output to a variable to avoid SIGPIPE on grep -q under
  # `set -o pipefail` (grep -q exits on first match, closing the pipe).
  syms="$(nm -C "$out" 2>/dev/null || true)"
  if grep -qE 'cudaq::__internal__::targetSetter' <<<"$syms"; then
    echo "PASS: $src has TargetSetter ctor (target=$target)"
  else
    echo "FAIL: $src is missing cudaq::__internal__::targetSetter (target=$target)"
    echo "      Runtime backend wiring will not run; expect a segfault on launch."
    failures=$((failures + 1))
  fi
done

if [[ $failures -gt 0 ]]; then
  echo "$failures example(s) missing TargetSetter — see above." >&2
  exit 1
fi

echo "All examples wired up correctly."
