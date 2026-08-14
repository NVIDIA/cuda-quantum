#!/bin/bash

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Check that every llvm::/mlir:: symbol a CUDA-Q shared object leaves undefined
# is exported by libcudaqMLIR.
#
# CUDA-Q ships a single shared MLIR/LLVM instance (libcudaqMLIR) and links no
# static MLIR/LLVM archives into its consumers, so that there is exactly one
# copy of every MLIR/LLVM symbol in the process. This script verifies that all
# undefined symbols are exported by libcudaqMLIR to detect any runtime
# dlopen/ImportError at compile time.
#
# Usage: check_mlir_symbols.sh <consumer> <libcudaqMLIR>

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <consumer-library> <libcudaqMLIR>" >&2
    exit 2
fi

consumer="$1"
provider="$2"

for f in "$consumer" "$provider"; do
    if [ ! -f "$f" ]; then
        echo "check_mlir_symbols: no such file: $f" >&2
        exit 2
    fi
done

# Itanium-mangled names whose first nested component is llvm:: or mlir::,
# including vtables/typeinfo (_ZTV/_ZTI/_ZTS). Mach-O prefixes an underscore.
mlir_symbols='^_?_Z(T[VIS])?N?K?[0-9]*4(llvm|mlir)'

# mlir::python:: is upstream MLIR's Python binding internals. Those live in
# their own libMLIRPythonSupport DSO, which the extensions link directly, and
# are deliberately not part of libcudaqMLIR.
not_provided_by_cudaqmlir='^_?_Z(T[VIS])?N?K?[0-9]*4mlir6python'

if [ "$(uname)" = "Darwin" ]; then
    undefined_of() { nm -u "$1"; }
    defined_of() { nm -gU "$1" | awk '{print $NF}'; }
else
    undefined_of() { nm -D --undefined-only "$1" | awk '{print $NF}'; }
    defined_of() { nm -D --defined-only "$1" | awk '{print $NF}'; }
fi

undefined=$(undefined_of "$consumer" \
    | grep -E "$mlir_symbols" \
    | grep -Ev "$not_provided_by_cudaqmlir" \
    | sort -u || true)
if [ -z "$undefined" ]; then
    exit 0
fi

defined=$(defined_of "$provider" | sort -u)
missing=$(comm -23 <(echo "$undefined") <(echo "$defined"))

if [ -z "$missing" ]; then
    exit 0
fi

count=$(echo "$missing" | wc -l | tr -d ' ')
cat >&2 <<EOF

================================================================================
$(basename "$consumer") references $count MLIR/LLVM symbol(s) that
$(basename "$provider") does not export:

$(echo "$missing" | head -20 | sed 's/^/  /')
$([ "$count" -gt 20 ] && echo "  ... and $((count - 20)) more")

A library you require may be missing in cmake/modules/mlir-bundled-libs.txt.
To find that library:

  for a in \$LLVM_INSTALL_PREFIX/lib/lib{MLIR,LLVM}*.a; do
    nm --defined-only "\$a" 2>/dev/null | grep -q '$(echo "$missing" | head -1)' \\
      && echo "\$a"
  done
================================================================================
EOF
exit 1
