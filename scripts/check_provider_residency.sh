#!/bin/bash

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Loaded-provider residency: a consumer binary must resolve the shared
# MLIR/LLVM providers from the intended SDK prefix, never from a preloaded
# or foreign system copy. The check is fail-closed: an ldd error, an
# unresolvable ('not found') provider, a provider missing from the output
# entirely, or a resolution whose canonical path escapes the prefix's lib/
# are all failures.
#
# Caveat: ldd observes the binary's load-time resolution (DT_NEEDED +
# RUNPATH); libraries dlopen'ed later or injected at run time are outside
# its view.
#
# Usage: check_provider_residency.sh <binary> <prefix> [provider-soname...]
# Without explicit SONAMEs the protected set is derived from the prefix's
# llvm-config: libMLIR/libMLIR-C/libMLIRExecutionEngineShared/libLLVM at the
# pinned <major>.<minor> version.

set -euo pipefail
export LC_ALL=C

if [ $# -lt 2 ] || [ ! -f "$1" ] || [ ! -d "$2" ]; then
    echo "Usage: $0 <binary> <prefix> [provider-soname...]" >&2
    exit 2
fi
binary="$1"
prefix="$(realpath "$2")"
shift 2

if [ $# -gt 0 ]; then
    protected=("$@")
else
    sover="$("$prefix/bin/llvm-config" --version | sed 's/\.[0-9]*$//')"
    protected=("libMLIR.so.$sover" "libMLIR-C.so.$sover"
               "libMLIRExecutionEngineShared.so.$sover" "libLLVM.so.$sover")
fi

ldd_out="$(ldd "$binary" 2>&1)" || {
    echo "check_provider_residency: ldd failed on $binary:" >&2
    echo "$ldd_out" >&2
    exit 1
}
[ -n "$ldd_out" ] || {
    echo "check_provider_residency: ldd produced no output for $binary" >&2
    exit 1
}

failures=0
seen=""
while IFS= read -r line; do
    line="${line#"${line%%[![:space:]]*}"}"   # ltrim
    case "$line" in
    *" => "*)
        soname="${line%% => *}"
        resolved="${line#* => }"
        resolved="${resolved%% (*}" ;;
    *" ("*)    # loader/vdso lines carry no SONAME -> no provider evidence
        continue ;;
    *) continue ;;
    esac
    soname="$(basename "$soname")"
    is_protected=false
    for p in "${protected[@]}"; do
        [ "$soname" = "$p" ] && is_protected=true
    done
    $is_protected || continue
    seen="$seen $soname"
    if [ "$resolved" = "not found" ] || [ ! -e "$resolved" ]; then
        echo "check_provider_residency: $soname unresolved ('$resolved')" >&2
        failures=$((failures + 1))
        continue
    fi
    canon="$(realpath "$resolved")"
    case "$canon" in
    "$prefix/lib"/*) ;;
    *)
        echo "check_provider_residency: $soname resolves outside the prefix:" \
             "$canon (expected under $prefix/lib)" >&2
        failures=$((failures + 1)) ;;
    esac
done <<< "$ldd_out"

for p in "${protected[@]}"; do
    case "$seen" in
    *" $p"*) ;;
    *)
        echo "check_provider_residency: expected provider $p not found in the" \
             "binary's load resolution (missing evidence)" >&2
        failures=$((failures + 1)) ;;
    esac
done

if [ "$failures" -ne 0 ]; then
    exit 1
fi
echo "check_provider_residency: all ${#protected[@]} providers resolve inside $prefix"
