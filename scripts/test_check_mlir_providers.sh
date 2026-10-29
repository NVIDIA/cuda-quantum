#!/bin/bash

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Focused negative tests for scripts/check_mlir_providers.sh: the audit must
# fail closed. Each case builds a symlink farm of a good provider prefix,
# breaks one fixture, and requires the audit to fail with the right message.
#
# Usage: test_check_mlir_providers.sh <llvm-install-prefix-with-providers>

set -u
export LC_ALL=C

if [ $# -ne 1 ] || [ ! -d "$1" ]; then
    echo "Usage: $0 <llvm-install-prefix-with-providers>" >&2
    exit 2
fi
prefix="$(cd "$1" && pwd)"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

failures=0
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT

# expect_fail <name> <expected-message-substring> <mutator commands...>
expect_fail() {
    local name="$1" expect="$2"; shift 2
    local farm="$scratch/farm"
    rm -rf "$farm"
    cp -as "$prefix" "$farm"
    (cd "$farm" && "$@")
    local out rc
    out="$(bash "$script_dir/check_mlir_providers.sh" "$farm" 2>&1)" && rc=0 || rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "NEG-TEST FAIL: $name: audit passed, expected failure"
        failures=$((failures + 1)); return
    fi
    if ! grep -qF "$expect" <<< "$out"; then
        echo "NEG-TEST FAIL: $name: missing message '$expect' in:"; echo "$out" | head -3
        failures=$((failures + 1)); return
    fi
    echo "NEG-TEST OK: $name"
}

# Positive control: the audit must pass on the untouched prefix.
if bash "$script_dir/check_mlir_providers.sh" "$prefix" > /dev/null 2>&1; then
    echo "NEG-TEST OK: positive control (prefix passes as-is)"
else
    echo "NEG-TEST FAIL: positive control: prefix must pass the audit unmutated"
    failures=$((failures + 1))
fi

# 1. MLIRConfig.cmake absent.
expect_fail "absent MLIRConfig.cmake" "devel side of the prefix is required" \
    rm "lib/cmake/mlir/MLIRConfig.cmake"

# 2. MLIR_UPSTREAM_CAPI_LIBS present but empty.
expect_fail "empty CAPI target manifest" "lists no MLIR_UPSTREAM_CAPI_LIBS" \
    bash -c 'rm lib/cmake/mlir/MLIRConfig.cmake && \
             echo "set_property(GLOBAL PROPERTY MLIR_UPSTREAM_CAPI_LIBS \"\")" \
               > lib/cmake/mlir/MLIRConfig.cmake'

# 3. A listed CAPI archive is missing.
first_capi=$(sed -n 's/.*MLIR_UPSTREAM_CAPI_LIBS "\(MLIRCAPI[A-Za-z]*\).*")/\1/p' \
    "$prefix/lib/cmake/mlir/MLIRConfig.cmake" | cut -d';' -f1)
expect_fail "missing CAPI archive" "missing CAPI archive" \
    rm "lib/lib${first_capi}.a"

# 4. A listed CAPI archive is unreadable/garbage (nm failure must surface).
expect_fail "corrupt CAPI archive" "could not read symbols" \
    bash -c "rm lib/lib${first_capi}.a && echo not-an-archive > lib/lib${first_capi}.a"

# 5. All CAPI archives present but empty: nothing collected must fail.
expect_fail "empty collected symbol set" "collected no public symbols" \
    bash -c 'ar rc empty.a && \
             for a in lib/libMLIRCAPI*.a lib/libMLIRCAPIDebug.a; do \
               [ -e "$a" ] && rm "$a" && cp empty.a "$a"; done; rm -f empty.a'

if [ "$failures" -ne 0 ]; then
    echo "NEG-TESTS: $failures failure(s)" >&2
    exit 1
fi
echo "NEG-TESTS: all passed"
