#!/bin/bash

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Verify that an installed LLVM/MLIR prefix provides the shared runtime
# provider set that the CUDA-Q SDK contract requires:
#
#   libLLVM.so.<sover>                       upstream LLVM implementation
#   libMLIR.so.<sover>                       upstream MLIR components
#   libMLIRExecutionEngineShared.so.<sover>  execution engine (excluded from libMLIR)
#   libMLIR-C.so.<sover>                     full upstream MLIR C API
#
# Checks, for the pinned baseline:
#  1. presence of each provider under its canonical SONAME and link names,
#  2. the required dependency direction between providers (DT_NEEDED) — the
#     structural anti-duplication proof: a provider that secretly embeds the
#     implementation needs no DT_NEEDED on the real providers,
#  3. no strong duplicate llvm::/mlir:: definitions across providers
#     (weak/vague-linkage copies are expected and ignored),
#  4. export closure: every public defined symbol of the upstream CAPI
#     archives listed in MLIRConfig.cmake (MLIR_UPSTREAM_CAPI_LIBS) is
#     exported by libMLIR-C (catches macro-generated pass/dialect APIs),
#  5. stock llvm-config coherence for shared and static queries.
#
# Coverage labeling: on an installed prefix this is a PRELIMINARY check —
# it cannot see statically embedded copies that --exclude-libs,ALL hides from
# the dynamic symbol table. The DT_NEEDED direction (2) and the libMLIR-C
# size gate are the structural proofs available post-strip. The mandatory
# build-stage evidence of spec section 3 (link maps, pre-strip full symbol
# tables) comes from passing --build-dir on the producer, added together with
# the producer wiring in the provider-audit slice; without it, a clean
# dynamic-symbol scan alone does not prove a single implementation.
#
# Usage: check_mlir_providers.sh <llvm-install-prefix> [--build-dir <dir>]

set -euo pipefail

# sort/comm must use byte order; locale-dependent collation breaks comm -23/-12.
export LC_ALL=C

if [ $# -lt 1 ] || [ ! -d "$1" ]; then
    echo "Usage: $0 <llvm-install-prefix> [--build-dir <dir>]" >&2
    exit 2
fi
prefix="$(cd "$1" && pwd)"
build_dir=""
if [ "${2:-}" = "--build-dir" ]; then
    build_dir="$3"
    [ -d "$build_dir" ] || { echo "no such build dir: $build_dir" >&2; exit 2; }
fi

failures=0
fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
note() { echo "check_mlir_providers: $*"; }

# Shared libraries carry a SONAME of the form lib<name>.so.<sover> where
# <sover> is <major>.<minor> derived from the installed llvm-config.
llvm_config="$prefix/bin/llvm-config"
if [ ! -x "$llvm_config" ]; then
    echo "check_mlir_providers: no executable $llvm_config" >&2
    exit 2
fi
version="$("$llvm_config" --version)"
sover="${version%.*}"
note "prefix=$prefix version=$version sover=$sover"

realpath_of() { readlink -f "$prefix/lib/$1"; }
soname_of() { readelf -d "$1" | sed -n 's/.*SONAME.*\[\(.*\)\]/\1/p'; }
needed_of() { readelf -d "$1" | sed -n 's/.*NEEDED.*Shared library: \[\(.*\)\]/\1/p'; }

# 1./2. Provider presence, canonical SONAMEs, link names, dependency direction.
check_provider() {
    local base="$1"; shift
    local versioned="lib${base}.so.${sover}"
    local path="$prefix/lib/$versioned"
    [ -e "$path" ] || { fail "missing provider $versioned"; return; }
    local real soname
    real="$(realpath_of "$versioned")"
    soname="$(soname_of "$real")"
    [ "$soname" = "$versioned" ] || fail "$versioned has SONAME '$soname'"
    [ -e "$prefix/lib/lib${base}.so" ] || fail "missing link name lib${base}.so"
    local dep
    for dep in "$@"; do
        needed_of "$real" | grep -qx "lib${dep}.so.${sover}" \
            || fail "lib${base} does not depend on lib${dep}.so.${sover}"
    done
}

check_provider LLVM
check_provider MLIR LLVM
check_provider MLIRExecutionEngineShared MLIR LLVM
check_provider MLIR-C MLIR MLIRExecutionEngineShared
# Upstream compatibility link name for the LLVM dylib.
[ -e "$prefix/lib/libLLVM-${sover%%.*}.so" ] \
    || fail "missing compatibility name libLLVM-${sover%%.*}.so"

# 3. No strong duplicate llvm::/mlir:: definitions across providers.
strong_mlir_defs() {
    nm -D --defined-only --extern-only "$1" 2>/dev/null \
        | awk '$2 ~ /^[TDBRiSG]$/ {print $NF}' \
        | sed 's/@.*$//' \
        | grep -E '^_Z(T[VIS])?N?K?[0-9]*4(llvm|mlir)' || true
}
providers=(LLVM MLIR MLIRExecutionEngineShared MLIR-C)
for ((i = 0; i < ${#providers[@]}; i++)); do
    defs_i="$(strong_mlir_defs "$(realpath_of "lib${providers[$i]}.so.${sover}")" | sort -u)"
    for ((j = i + 1; j < ${#providers[@]}; j++)); do
        dup="$(comm -12 <(echo "$defs_i") \
            <(strong_mlir_defs "$(realpath_of "lib${providers[$j]}.so.${sover}")" | sort -u))"
        [ -z "$dup" ] || fail "strong duplicate definitions in lib${providers[$i]} and lib${providers[$j]}:"$'\n'"$(echo "$dup" | head -5)"
    done
done
# libMLIR-C must not carry its own strong copy of the C++ implementation.
mlir_c_strong="$(strong_mlir_defs "$(realpath_of "libMLIR-C.so.${sover}")")"
[ -z "$mlir_c_strong" ] || fail "libMLIR-C defines strong llvm::/mlir:: symbols:"$'\n'"$(echo "$mlir_c_strong" | head -5)"

# 4a. Structural size evidence: libMLIR-C wraps MLIR's C API in a thin layer;
# an embedded implementation would make it a full copy of libMLIR. Reject if
# libMLIR-C exceeds a fifteenth of libMLIR (measured: 0.8MB vs 127MB).
size_of() { stat -c %s "$1"; }
mlir_size=$(size_of "$(realpath_of "libMLIR.so.${sover}")")
mlir_c_size=$(size_of "$(realpath_of "libMLIR-C.so.${sover}")")
[ $((mlir_c_size * 15)) -lt "$mlir_size" ] ||
    fail "libMLIR-C (${mlir_c_size}B) is suspiciously large next to libMLIR (${mlir_size}B); a hidden static copy is likely embedded"

# 4. Export closure: public defined symbols of the pinned CAPI archives
# must be exported by libMLIR-C. Reading the archives (not headers) also
# covers macro-generated pass/dialect registration functions.
# The export closure fails closed: a missing MLIRConfig.cmake, an empty
# MLIR_UPSTREAM_CAPI_LIBS, or an unreadable archive is a failure, never a
# silent pass.
mlir_config="$prefix/lib/cmake/mlir/MLIRConfig.cmake"
[ -f "$mlir_config" ] || fail "missing $mlir_config (devel side of the prefix is required)"
if [ -f "$mlir_config" ]; then
    capi_libs=$(sed -n 's/.*MLIR_UPSTREAM_CAPI_LIBS "\(.*\)")/\1/p' "$mlir_config" | tr ';' ' ')
    [ -n "$capi_libs" ] || fail "MLIRConfig.cmake lists no MLIR_UPSTREAM_CAPI_LIBS"
    capi_count=0
    for lib in $capi_libs; do capi_count=$((capi_count + 1)); done
    [ "$capi_count" -ge 50 ] || fail "only $capi_count upstream CAPI libraries listed (expected 58)"
    capi_exports="$(mktemp)"
    missing="$(mktemp)"
    trap 'rm -f "$capi_exports" "$missing"' EXIT
    for lib in $capi_libs; do
        archive="$prefix/lib/lib${lib}.a"
        [ -f "$archive" ] || { fail "missing CAPI archive lib${lib}.a (listed in MLIRConfig.cmake)"; continue; }
        # C API symbols only: C++ (mangled) symbols in the archives are
        # resolved from the shared providers, not re-exported by libMLIR-C.
        if ! nm -g --defined-only "$archive" > /dev/null 2>&1; then
            fail "could not read symbols from lib${lib}.a"
            continue
        fi
        nm -g --defined-only "$archive" 2>/dev/null \
            | awk '$2 ~ /^[TDBRW]$/ {print $NF}' | grep -v '^_Z' >> "$capi_exports" || true
    done
    sort -u "$capi_exports" -o "$capi_exports"
    [ -s "$capi_exports" ] || fail "collected no public symbols from the upstream CAPI archives"
    if ! nm -D --defined-only "$(realpath_of "libMLIR-C.so.${sover}")" 2>/dev/null \
        | awk '$2 ~ /^[TDBRiSGW]$/ {print $NF}' | sort -u > "$missing.tmp"; then
        fail "could not read dynamic symbols of libMLIR-C.so.${sover}"
    fi
    comm -23 "$capi_exports" "$missing.tmp" > "$missing"
    rm -f "$missing.tmp"
    if [ -s "$missing" ]; then
        fail "libMLIR-C does not export $(wc -l < "$missing") symbols from the upstream CAPI archives, e.g.:"$'\n'"$(head -5 "$missing")"
    else
        note "CAPI export closure holds ($(wc -l < "$capi_exports") symbols)"
    fi
fi

# 5. Stock llvm-config coherence.
[ -d "$("$llvm_config" --includedir)" ] || fail "llvm-config --includedir does not exist"
[ -d "$("$llvm_config" --libdir)" ] || fail "llvm-config --libdir does not exist"
shared_names="$("$llvm_config" --link-shared --libnames)"
[ "$shared_names" = "libLLVM-${sover%%.*}.so" ] \
    || fail "llvm-config --link-shared --libnames returned '$shared_names', expected 'libLLVM-${sover%%.*}.so'"
static_names="$("$llvm_config" --link-static --libnames)"
echo "$static_names" | grep -q "libLLVMSupport.a" \
    || fail "llvm-config --link-static --libnames lost static archives: '$static_names'"

if [ "$failures" -ne 0 ]; then
    echo "check_mlir_providers: $failures check(s) failed for $prefix" >&2
    exit 1
fi
note "all checks passed for $prefix"
