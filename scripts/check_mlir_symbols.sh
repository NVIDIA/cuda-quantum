#!/bin/bash

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Verify that a CUDA-Q consumer resolves MLIR/LLVM from libcudaqMLIR only.
#
# Phase 1: every llvm::/mlir:: symbol the consumer leaves undefined must be
# exported by libcudaqMLIR (or an extra provider).
#
# Phase 2: the consumer must not define any strong symbol that libcudaqMLIR
# already exports. Weak/vague-linkage copies from non-bundled static archives
# are expected and ignored.
#
# Additional providers may be named after libcudaqMLIR. A downstream project's
# own common CAPI library is the usual case: template instantiations over
# downstream types are mangled into namespace mlir (e.g.
# mlir::detail::TypeIDResolver<my::Dialect>::id), so they match the MLIR symbol
# pattern below even though only the downstream library can ever define them.
#
# Usage: check_mlir_symbols.sh <consumer> <libcudaqMLIR> [extra-provider...]

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <consumer-library> <libcudaqMLIR> [extra-provider...]" >&2
    exit 2
fi

consumer="$1"
provider="$2"
shift 2
extra_providers=("$@")

for f in "$consumer" "$provider" "${extra_providers[@]+"${extra_providers[@]}"}"; do
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
    strong_exports_of() { nm -gU -m "$1" | awk '!/ weak external /{print $NF}'; }
else
    undefined_of() { nm -D --undefined-only "$1" | awk '{print $NF}'; }
    defined_of() { nm -D --defined-only "$1" | awk '{print $NF}'; }
    strong_exports_of() {
        nm -D --defined-only --extern-only "$1" | awk '$2 ~ /^[TDBRiSG]$/ {print $NF}'
    }
fi

provider_names=$(for p in "$provider" "${extra_providers[@]+"${extra_providers[@]}"}"; do
    basename "$p"
done | paste -sd' ' -)

exit_code=0

# ---- Phase 1: undefined MLIR/LLVM symbols must be provided ---------------- #
undefined=$(undefined_of "$consumer" \
    | grep -E "$mlir_symbols" \
    | grep -Ev "$not_provided_by_cudaqmlir" \
    | sort -u || true)

if [ -n "$undefined" ]; then
    defined=$({ defined_of "$provider"
                for extra in "${extra_providers[@]+"${extra_providers[@]}"}"; do
                    defined_of "$extra"
                done
              } | sort -u)
    missing=$(comm -23 <(echo "$undefined") <(echo "$defined"))

    if [ -n "$missing" ]; then
        count=$(echo "$missing" | wc -l | tr -d ' ')
        cat >&2 <<EOF

================================================================================
$(basename "$consumer") references $count MLIR/LLVM symbol(s) that
$provider_names does not export:

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
        exit_code=1
    fi
fi

# ---- Phase 2: no duplicate strong definitions of libcudaqMLIR exports ----- #
owned=$(strong_exports_of "$provider" | sort -u)
consumer_defs=$(strong_exports_of "$consumer" | sort -u)
duplicates=$(comm -12 <(echo "$consumer_defs") <(echo "$owned") || true)

if [ -n "$duplicates" ]; then
    count=$(echo "$duplicates" | wc -l | tr -d ' ')
    cat >&2 <<EOF

================================================================================
$(basename "$consumer") defines $count strong symbol(s) already exported by
$(basename "$provider"):

$(echo "$duplicates" | head -20 | sed 's/^/  /')
$([ "$count" -gt 20 ] && echo "  ... and $((count - 20)) more")

Static MLIR/LLVM archives were linked into this library instead of resolving
from libcudaqMLIR. Link cudaq::cudaqMLIR (or cudaq::MLIR) and ensure its
INTERFACE_LINK_OPTIONS place libcudaqMLIR before any bundled component archive.
================================================================================
EOF
    exit_code=1
fi

exit "$exit_code"
