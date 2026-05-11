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
# Run from TMPDIR so cudaq-quake's intermediate .ll/.qke files land there
# instead of polluting (or racing in) the caller's cwd.
cd "$TMPDIR"

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

  # Prefer `llvm-nm` from the CUDA-Q LLVM toolchain
  nm_tool=""
  if [[ -n "${LLVM_INSTALL_PREFIX:-}" &&
        -x "$LLVM_INSTALL_PREFIX/bin/llvm-nm" ]]; then
    nm_tool="$LLVM_INSTALL_PREFIX/bin/llvm-nm"
  elif command -v llvm-nm >/dev/null 2>&1; then
    nm_tool="$(command -v llvm-nm)"
  elif command -v nm >/dev/null 2>&1; then
    nm_tool="$(command -v nm)"
  fi

  # Capture nm output to a variable to avoid `SIGPIPE` on `grep -q` under
  # `set -o pipefail` (grep -q exits on first match, closing the pipe).
  syms=""
  if [[ -n "$nm_tool" ]]; then
    syms="$("$nm_tool" -C "$out" 2>/dev/null || true)"
  fi

  # Fallback: when `nm` cannot list the symbol (e.g. stripped binary), look for
  # the backend config literal nvq++ embeds in `.rodata` via NVQPP_TARGET_BACKEND_CONFIG.
  marker="${target};emulate;false;disable_qubit_mapping;false"
  if grep -qE 'cudaq::__internal__::targetSetter' <<<"$syms" ||
      LC_ALL=C grep -aFq "$marker" "$out"; then
    echo "PASS: $src has TargetSetter ctor (target=$target)"
  else
    echo "FAIL: $src is missing cudaq::__internal__::targetSetter (target=$target)"
    echo "      Runtime backend wiring will not run; expect a segfault on launch."
    echo "      Diagnostic: nm tool: ${nm_tool:-<none>}"
    echo "      Diagnostic: backend config marker not found: $marker"
    if command -v file >/dev/null 2>&1; then
      echo "      Diagnostic: file output: $(file "$out" 2>/dev/null || true)"
    fi
    failures=$((failures + 1))
  fi
done

if [[ $failures -gt 0 ]]; then
  echo "$failures example(s) missing TargetSetter — see above." >&2
  exit 1
fi

echo "All examples wired up correctly."
