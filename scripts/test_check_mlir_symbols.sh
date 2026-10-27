#!/bin/bash

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Negative tests for the phase-3 hidden-copy audit in check_mlir_symbols.sh.
# Exercises the ownership proof (unstripped consumers) and the link-input
# provenance (stripped consumers) with realistic fixtures:
#   - a SMALL upstream component archive (libMLIRSupport.a, ~300 named
#     definitions) embedded behind --exclude-libs,ALL must FAIL the ownership
#     proof even though it is far below any population threshold;
#   - relative/direct-object/rsp/variable/missing link inputs, inputs that
#     only appear in LINK_LIBRARIES or rspfile contents, provider roots
#     nested under the build tree, and archives copied into the build tree
#     must all be classified or rejected;
#   - a stripped consumer without (or with dangling) Ninja evidence must
#     FAIL, never pass silently;
#   - clean consumers must PASS.
#
# Fixture-construction errors are fatal. Supplied consumer/provider paths
# are treated as READ-ONLY: generated fixtures are only ever written to
# fresh scratch paths, and supplied files are hash-verified unchanged at
# the end (SDK97-6 regression: an earlier harness compiled over a supplied
# consumer DSO).
#
# Usage: test_check_mlir_symbols.sh <llvm-install-prefix>
#        [consumer-dso [libcudaqMLIR [extra-provider...]]]
# The optional consumer arguments default to compiled fixtures; a real
# build-tree libcudaqMLIR and extension DSO give the strongest fixtures.

set -eu
export LC_ALL=C

if [ $# -lt 1 ] || [ ! -d "$1" ]; then
    echo "Usage: $0 <llvm-install-prefix> [consumer libcudaqMLIR extra-provider...]" >&2
    exit 2
fi
prefix="$(cd "$1" && pwd)"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
supplied_consumer="${2:-}"
supplied_cudaq_mlir="${3:-}"
shift 2 2>/dev/null || shift $#  # remaining args (if any) are extra providers

unset CUDAQ_BUILD_NINJA || true
sover="$("$prefix/bin/llvm-config" --version | sed 's/\.[0-9]*$//')"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
failures=0

die() { echo "FIXTURE ERROR: $*" >&2; exit 2; }

check() {   # check <name> <expect-pass|expect-fail> <expected-substring> <cmd...>
    local name="$1" expect="$2" want="$3"; shift 3
    local out rc
    out="$("$@" 2>&1)" && rc=0 || rc=$?
    if [ "$expect" = pass ] && [ "$rc" -ne 0 ]; then
        echo "NEG-TEST FAIL: $name: expected pass, exit $rc"; echo "$out" | head -3
        failures=$((failures + 1)); return
    fi
    if [ "$expect" = fail ]; then
        if [ "$rc" -eq 0 ]; then
            echo "NEG-TEST FAIL: $name: expected failure, audit passed"
            failures=$((failures + 1)); return
        fi
        if ! grep -qF "$want" <<< "$out"; then
            echo "NEG-TEST FAIL: $name: missing '$want' in:"; echo "$out" | head -3
            failures=$((failures + 1)); return
        fi
    fi
    echo "NEG-TEST OK: $name"
}

# Supplied inputs are read-only. Hash them now; the harness asserts the
# bytes are unchanged after every fixture ran (SDK97-6 regression).
declare -A supplied_hash
for f in "$supplied_consumer" "$supplied_cudaq_mlir" "$@"; do
    [ -z "$f" ] || supplied_hash["$(realpath "$f")"]="$(sha256sum "$(realpath "$f")" | awk '{print $1}')"
done

consumer="$supplied_consumer"
cudaq_mlir="$supplied_cudaq_mlir"
[ -n "$consumer" ] || consumer="$scratch/libclean.so"
[ -n "$cudaq_mlir" ] || cudaq_mlir="$scratch/libclean2.so"

# Generated fixtures live only in the scratch directory.
echo 'int fixture_anchor() { return 0; }' > "$scratch/anchor.cpp"
echo 'int provider_anchor() { return 0; }' > "$scratch/anchor2.cpp"
[ -n "$supplied_consumer" ] || \
    c++ -shared -fPIC -o "$consumer" "$scratch/anchor.cpp" || die "compiling consumer fixture"
[ -n "$supplied_cudaq_mlir" ] || \
    c++ -shared -fPIC -o "$cudaq_mlir" "$scratch/anchor2.cpp" || die "compiling provider fixture"

# A synthetic SDK provider tree: always built, so the provenance fixtures
# do not depend on the prefix's real providers. The synthetic provider
# exports one mlir:: name the "small component" archive also defines.
sdk="$scratch/sdk"
mkdir -p "$sdk/lib"
cat > "$sdk/provider_impl.cpp" <<'EOF'
namespace mlir { int openInputFile() { return 0; } }
namespace mlir { __attribute__((weak, visibility("default"))) int weakInlineHelper() { return 3; } }
extern "C" int mlirContextCreate() { return 0; }
extern "C" unsigned long adler32(unsigned long) { return 0; }
int sdk_anchor() { return 0; }
EOF
cat > "$sdk/weak_impl.cpp" <<'EOF'
namespace mlir { __attribute__((weak)) int weakInlineHelper() { return 9; } }
EOF
c++ -fPIC -c -o "$sdk/weak_impl.o" "$sdk/weak_impl.cpp" || die "compiling weak object"
ar rcs "$sdk/lib/libweakdup.a" "$sdk/weak_impl.o" || die "archiving weak archive"
c++ -shared -fPIC -o "$sdk/lib/libMLIR.so.$sover" "$sdk/provider_impl.cpp" \
    -Wl,-soname,"libMLIR.so.$sover" || die "compiling synthetic provider"
cat > "$sdk/capi_impl.cpp" <<'EOF'
extern "C" int mlirContextCreate() { return 7; }
EOF
c++ -fPIC -c -o "$sdk/capi_impl.o" "$sdk/capi_impl.cpp" || die "compiling capi object"
ar rcs "$sdk/lib/libMLIRCAPICopied.a" "$sdk/capi_impl.o" || die "archiving capi copy"
cat > "$sdk/z_impl.cpp" <<'EOF'
extern "C" unsigned long adler32(unsigned long) { return 3; }
EOF
c++ -fPIC -c -o "$sdk/z_impl.o" "$sdk/z_impl.cpp" || die "compiling z-style object"
ar rcs "$sdk/lib/libz-style.a" "$sdk/z_impl.o" || die "archiving z-style archive"
cat > "$sdk/support_impl.cpp" <<'EOF'
namespace mlir { int openInputFile() { return 7; } }
EOF
c++ -fPIC -c -o "$sdk/support_impl.o" "$sdk/support_impl.cpp" || die "compiling component object"
ar rcs "$sdk/lib/libMLIRSupport.a" "$sdk/support_impl.o" || die "archiving synthetic component"
cat > "$sdk/pass_impl.cpp" <<'EOF'
int mlir_pass_anchor() { return 0; }
EOF
c++ -fPIC -c -o "$sdk/pass_impl.o" "$sdk/pass_impl.cpp" || die "compiling pass object"
ar rcs "$sdk/lib/libMLIRPass.a" "$sdk/pass_impl.o" || die "archiving pass component"
cp "$sdk/pass_impl.o" "$sdk/lib/obj.MLIRIR.o" || die "placing direct object"

# 1. Clean consumer: passes. A stripped consumer (e.g. a real nanobind
#    extension) needs its clean provenance edge — missing evidence is a
#    failure by design, so supply the edge.
real_providers=("$prefix/lib/libMLIR.so.$sover" "$prefix/lib/libMLIR-C.so.$sover"
           "$prefix/lib/libMLIRExecutionEngineShared.so.$sover" "$prefix/lib/libLLVM.so.$sover")
if readelf -SW "$consumer" 2>/dev/null | grep -q '\.symtab'; then
    check "clean consumer" pass "" \
        bash "$script_dir/check_mlir_symbols.sh" "$consumer" "$cudaq_mlir" "${real_providers[@]}"
else
    mkdir -p "$scratch/clean-edge/obj"
    echo 'int clean_obj_anchor() { return 0; }' > "$scratch/clean-edge/o.cpp"
    c++ -fPIC -c -o "$scratch/clean-edge/obj/main.o" "$scratch/clean-edge/o.cpp" \
        || die "compiling clean-edge object"
    printf '%s\n' "build python/x/$(basename "$consumer"): LINKER obj/main.o" \
        > "$scratch/clean-edge/build.ninja"
    CUDAQ_BUILD_NINJA="$scratch/clean-edge/build.ninja" \
        check "clean consumer (stripped, with provenance edge)" pass "" \
        bash "$script_dir/check_mlir_symbols.sh" "$consumer" "$cudaq_mlir" "${real_providers[@]}"
fi

# 2. SMALL upstream component embedded and localized: the ownership proof
#    fails (duplicate provider-exported names), despite being far below any
#    population cap. Synthetic component: always runs. The prefix's real
#    libMLIRSupport.a, when present, is the exact reviewer-named case.
c++ -shared -fPIC -o "$scratch/libsmallhidden.so" "$scratch/anchor.cpp" \
    -Wl,--whole-archive "$sdk/lib/libMLIRSupport.a" -Wl,--no-whole-archive \
    -Wl,--exclude-libs,ALL -Wl,-z,undefs || die "compiling small-hidden fixture"
check "small hidden component (synthetic)" fail "duplicate names exported by the providers" \
    bash "$script_dir/check_mlir_symbols.sh" "$scratch/libsmallhidden.so" \
    "$sdk/lib/libMLIR.so.$sover"

# 2a-i. A localized VAGUE-LINKAGE duplicate (weak provider export) is not
#       a second implementation: phase 2 ignores weak duplicates in the
#       dynamic view; the ownership view does the same for LOCAL copies.
c++ -shared -fPIC -o "$scratch/libweakdup.so" "$scratch/anchor.cpp" \
    -Wl,--whole-archive "$sdk/lib/libweakdup.a" -Wl,--no-whole-archive \
    -Wl,--exclude-libs,ALL -Wl,-z,undefs || die "compiling weak-dup fixture"
check "localized weak duplicate tolerated" pass "" \
    bash "$script_dir/check_mlir_symbols.sh" "$scratch/libweakdup.so" \
    "$sdk/lib/libMLIR.so.$sover"

# 2b. A copied C-API component (plain mlirXxx names) is ownership too.
c++ -shared -fPIC -o "$scratch/libcapihidden.so" "$scratch/anchor.cpp" \
    -Wl,--whole-archive "$sdk/lib/libMLIRCAPICopied.a" -Wl,--no-whole-archive \
    -Wl,--exclude-libs,ALL -Wl,-z,undefs || die "compiling capi-hidden fixture"
check "hidden CAPI copy (plain mlirXxx names)" fail \
    "duplicate names exported by the providers" \
    bash "$script_dir/check_mlir_symbols.sh" "$scratch/libcapihidden.so" \
    "$sdk/lib/libMLIR.so.$sover"
if [ -f "$prefix/lib/libMLIRSupport.a" ]; then
    c++ -shared -fPIC -o "$scratch/libsmallhidden-real.so" "$scratch/anchor.cpp" \
        -Wl,--whole-archive "$prefix/lib/libMLIRSupport.a" -Wl,--no-whole-archive \
        -Wl,--exclude-libs,ALL -Wl,-z,undefs 2>/dev/null \
        || die "compiling real small-hidden fixture"
    check "small hidden component (real libMLIRSupport.a)" fail \
        "duplicate names exported by the providers" \
        bash "$script_dir/check_mlir_symbols.sh" "$scratch/libsmallhidden-real.so" \
        "$prefix/lib/libMLIR.so.$sover" "$prefix/lib/libLLVM.so.$sover"
else
    echo "NEG-TEST SKIP: no libMLIRSupport.a for the real small-component fixture"
fi

# 3. Link-input provenance fixtures. A stripped stand-in consumer plus
#    CUDAQ_BUILD_NINJA selects the provenance view; a supplied real
#    (stripped) consumer is used read-only instead when given.
strip "$consumer" -o "$scratch/stripped.so" 2>/dev/null \
    || cp "$consumer" "$scratch/stripped.so" || die "stripping stand-in consumer"
probe="$scratch/stripped.so"
[ -n "$supplied_consumer" ] && probe="$consumer"

build_dir="$scratch/build"
mkdir -p "$build_dir/obj"
edge="$build_dir/build.ninja"
edge_base="build python/x/$(basename "$probe"): LINKER obj/main.o"
cp "$sdk/pass_impl.o" "$build_dir/obj/main.o"

# A clean downstream-owned archive (defines no provider-exported names).
cat > "$build_dir/trivial.cpp" <<'EOF'
namespace acme { int trivial_dialect_anchor() { return 0; } }
EOF
c++ -fPIC -c -o "$build_dir/trivial.o" "$build_dir/trivial.cpp" || die "compiling downstream object"
mkdir -p "$build_dir/lib"
ar rcs "$build_dir/lib/libTrivialDialect.a" "$build_dir/trivial.o" || die "archiving downstream lib"
# A provider component copied into the consumer's own build tree.
cp "$sdk/lib/libMLIRSupport.a" "$build_dir/lib/copied-component.a" || die "copying component"
# A provider root nested below the consumer build tree.
mkdir -p "$build_dir/nested-sdk/lib"
cp "$sdk/lib/libMLIR.so.$sover" "$build_dir/nested-sdk/lib/"
cp "$sdk/lib/libMLIRPass.a" "$build_dir/nested-sdk/lib/"

sdk_providers=("$sdk/lib/libMLIR.so.$sover")
run_audit() {   # run_audit <extra-provider...> with $edge as CUDAQ_BUILD_NINJA
    CUDAQ_BUILD_NINJA="$edge" \
        bash "$script_dir/check_mlir_symbols.sh" "$probe" "$cudaq_mlir" "$@" \
        "${real_providers[@]}"
}
mk_edge() { printf '%s\n' "$1" > "$edge"; }
rel_to_build() { realpath --relative-to="$build_dir" "$1"; }

# 3a. Absolute upstream archive: rejected.
mk_edge "$edge_base $sdk/lib/libMLIRPass.a | x"
check "absolute upstream archive" fail "forbidden upstream component" \
    run_audit "${sdk_providers[@]}"

# 3b. RELATIVE path to the same upstream archive: resolved, then rejected.
mk_edge "$edge_base $(rel_to_build "$sdk/lib/libMLIRPass.a") | x"
check "relative upstream archive" fail "forbidden upstream component" \
    run_audit "${sdk_providers[@]}"

# 3c. Provider root nested below the build tree: upstream ownership is
#     checked before any build-tree exemption.
mk_edge "$edge_base nested-sdk/lib/libMLIRPass.a | x"
check "nested provider root under build tree" fail "forbidden upstream component" \
    run_audit "$build_dir/nested-sdk/lib/libMLIR.so.$sover"

# 3d. Direct upstream object file: rejected like an archive.
mk_edge "$edge_base $sdk/lib/obj.MLIRIR.o | x"
check "direct upstream object" fail "forbidden upstream component" \
    run_audit "${sdk_providers[@]}"

# 3e. Input only on the LINK_LIBRARIES variable line (the realistic CMake
#     location — not on the dependency edge): still scanned and rejected.
cp "$sdk/pass_impl.o" "$build_dir/obj/other.o"
printf '%s\n' "$edge_base obj/other.o" \
    "  LINK_LIBRARIES = $sdk/lib/libMLIRPass.a" \
    "  TARGET_FILE = python/x/$(basename "$probe")" > "$edge"
check "archive only in LINK_LIBRARIES variable" fail "forbidden upstream component" \
    run_audit "${sdk_providers[@]}"

# 3f. rspfile attribute: its CONTENTS are link inputs.
printf '%s\n' "$edge_base" "  rspfile = link.rsp" > "$edge"
printf '%s\n' "$sdk/lib/libMLIRPass.a" > "$build_dir/link.rsp"
check "upstream archive inside rspfile contents" fail "forbidden upstream component" \
    run_audit "${sdk_providers[@]}"

# 3g. Missing rspfile: missing evidence fails closed.
printf '%s\n' "$edge_base" "  rspfile = absent.rsp" > "$edge"
check "missing rspfile" fail "unresolved" run_audit "${sdk_providers[@]}"

# 3h. Response-file/variable tokens on the edge: unresolved evidence.
mk_edge "$edge_base out.rsp | x"
check "rsp token" fail "unresolved" run_audit "${sdk_providers[@]}"
mk_edge "$edge_base \$LINK_LIBRARIES | x"
check "variable token" fail "unresolved" run_audit "${sdk_providers[@]}"

# 3i. An input that does not exist cannot be inspected: missing evidence.
mk_edge "$edge_base lib/does-not-exist.a | x"
check "missing input file" fail "missing-input" run_audit "${sdk_providers[@]}"

# 3j. A provider component copied into the consumer's tree is caught by
#     content (strong definitions duplicate provider exports).
mk_edge "$edge_base lib/copied-component.a | x"
check "copied provider component in build tree" fail "embedded-definitions" \
    run_audit "${sdk_providers[@]}"

# 3k-i. Inputs conveyed through linker flags: -Wl, pieces are split and
#      classified like plain path tokens.
mk_edge "$edge_base -Wl,--whole-archive,$sdk/lib/libMLIRPass.a,--no-whole-archive"
check "archive inside -Wl whole-archive flags" fail "forbidden upstream component" \
    run_audit "${sdk_providers[@]}"

# 3k-ii. -L/-l search forms: exact and semantic names resolve through the
#      edge's library path and classify by what they resolve to.
mk_edge "$edge_base -L$sdk/lib -l:libMLIRPass.a"
check "exact library name (-l:) into provider dir" fail "forbidden upstream component" \
    run_audit "${sdk_providers[@]}"
mk_edge "$edge_base -L$sdk/lib -lMLIRPass"
check "semantic library name (-l) into provider dir" fail "forbidden upstream component" \
    run_audit "${sdk_providers[@]}"
mk_edge "$edge_base -lNoSuchSDKComponent"
check "unresolvable -l name" fail "unresolved" run_audit "${sdk_providers[@]}"

# 3k-iia. -l resolution matches GNU ld: directories in order, .so tried
#      first in each directory, .a as a per-directory fallback (the
#      -Bdynamic default does NOT skip archives); -Bstatic restricts the
#      whole search to archives. An early archive must not hide behind a
#      later DSO of the same name in EITHER mode.
mkdir -p "$build_dir/early/lib" "$build_dir/late/lib"
ar rcs "$build_dir/early/lib/libOwned.a" "$sdk/support_impl.o"
c++ -shared -fPIC -o "$build_dir/late/lib/libOwned.so" "$scratch/anchor.cpp"
mk_edge "$edge_base -L$build_dir/early/lib -L$build_dir/late/lib -lOwned"
check "early archive beats later DSO (ld order)" fail "embedded-definitions" \
    run_audit "${sdk_providers[@]}"
mk_edge "$edge_base -L$build_dir/early/lib -L$build_dir/late/lib -Bdynamic -lOwned"
check "-Bdynamic keeps the per-directory archive fallback" fail \
    "embedded-definitions" run_audit "${sdk_providers[@]}"
mk_edge "$edge_base -Bstatic -L$build_dir/late/lib -L$build_dir/early/lib -lOwned"
check "-Bstatic selects the archive despite order" fail "embedded-definitions" \
    run_audit "${sdk_providers[@]}"
mk_edge "$edge_base -Bstatic -L$build_dir/late/lib --push-state -L$build_dir/early/lib -lOwned --pop-state -lrt"
check "push/pop restores resolution state" fail "embedded-definitions" \
    run_audit "${sdk_providers[@]}"

# 3k-iib. GNU applies every -L to every -l regardless of textual position:
#      a -l in LINK_LIBRARIES must see a -L that only appears later, in
#      LINK_PATH. (edge as a multi-line block with the flag order swapped.)
printf '%s\n' "$edge_base" \
    "  LINK_LIBRARIES = -lOwned" \
    "  LINK_PATH = -L$build_dir/early/lib" > "$edge"
check "later -L applies to earlier -l (LINK_PATH after LINK_LIBRARIES)" \
    fail "embedded-definitions" run_audit "${sdk_providers[@]}"

# 3k-iid. The implicit search path comes from the build's toolchain data
#      (sibling CMakeCache), not a hardcoded host tuple: a custom
#      (non-Debian-tuple) implicit dir is honored, and a provider archive
#      there is caught. The env wiring overrides the cache.
custom_tc="$build_dir/custom-tc"
mkdir -p "$custom_tc/lib"
cat > "$build_dir/CMakeCache.txt" <<EOF
CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES:STATIC=$custom_tc/lib
EOF
cp "$build_dir/late/lib/libOwned.so" "$custom_tc/lib/"
printf '%s\n' "$edge_base" "  LINK_LIBRARIES = -lOwned" > "$edge"
check "implicit dirs from sibling CMakeCache honored" pass "" run_audit "${sdk_providers[@]}"
# Only the archive remains: -lOwned now resolves to it (ld's .so-first
# fallback is modeled, so the DSO must be removed for this case).
rm "$custom_tc/lib/libOwned.so"
ar rcs "$custom_tc/lib/libOwned.a" "$sdk/support_impl.o"
check "provider archive in custom implicit dir caught" fail "embedded-definitions" \
    run_audit "${sdk_providers[@]}"
rm "$custom_tc/lib/libOwned.a"
cp "$build_dir/late/lib/libOwned.so" "$custom_tc/lib/"
CUDAQ_IMPLICIT_LINK_DIRS="/definitely/not/used:$custom_tc/lib" run_audit "${sdk_providers[@]}" \
    > /dev/null 2>&1 && echo "NEG-TEST OK: env implicit-dir override wins" || {
        echo "NEG-TEST FAIL: env implicit-dir override wins"; failures=$((failures + 1)); }
rm -f "$build_dir/CMakeCache.txt"

# 3k-iie-i. Directory boundaries are preserved end-to-end, including a
#      directory containing a space, and the CMake gate env serialization
#      survives as ONE argument (configure-time probe of the real wiring).
spacey="$build_dir/sp ace"
mkdir -p "$spacey"
cp "$build_dir/late/lib/libOwned.so" "$spacey/"
printf '%s\n' "$edge_base" "  LINK_LIBRARIES = -lOwned" > "$edge"
CUDAQ_IMPLICIT_LINK_DIRS="/nope:$spacey" check "spaced implicit dir honored" pass "" \
    run_audit "${sdk_providers[@]}"

cmake_probe="$scratch/cmake-probe"
mkdir -p "$cmake_probe"
cat > "$cmake_probe/CMakeLists.txt" <<PROBEEOF
cmake_minimum_required(VERSION 3.26)
project(probe NONE)
set(CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES "$custom_tc/lib;/usr/lib")
string(REPLACE ";" ":" _dirs "\${CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES}")
add_custom_target(probe ALL
  COMMAND \${CMAKE_COMMAND} -E env "CUDAQ_IMPLICIT_LINK_DIRS=\${_dirs}"
    bash -c "printf '%s' \"\$CUDAQ_IMPLICIT_LINK_DIRS\" > \"\${CMAKE_BINARY_DIR}/env-value.txt\""
  VERBATIM)
PROBEEOF
cmake -S "$cmake_probe" -B "$cmake_probe/build" -G Ninja >/dev/null 2>&1 \
    || die "cmake probe configure failed"
cmake --build "$cmake_probe/build" >/dev/null 2>&1 \
    || die "cmake probe build failed (env arg list split)"
got="$(cat "$cmake_probe/build/env-value.txt")"
if [ "$got" = "$custom_tc/lib:/usr/lib" ]; then
    echo "NEG-TEST OK: CMake gate env serialization intact (2 dirs, one arg)"
else
    echo "NEG-TEST FAIL: CMake gate env serialization: got '$got'"
    failures=$((failures + 1))
fi

# 3k-iie. The configured toolchain record (CMakeFiles/<ver>/CMakeCXXCompiler.cmake)
#      is honored when no env/cache entry exists — the real LLVM build's shape.
rm -f "$build_dir/CMakeCache.txt"
mkdir -p "$build_dir/CMakeFiles/4.0.7"
cat > "$build_dir/CMakeFiles/4.0.7/CMakeCXXCompiler.cmake" <<EOF
set(CMAKE_CXX_COMPILER "/usr/bin/c++")
set(CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES "$custom_tc/lib;/usr/lib")
EOF
printf '%s\n' "$edge_base" "  LINK_LIBRARIES = -lOwned" > "$edge"
check "implicit dirs from CMakeCXXCompiler.cmake record" pass "" run_audit "${sdk_providers[@]}"
rm -rf "$build_dir/CMakeFiles"

# 3k-iic. Control: the same shape with the provider content only in a DSO
#      passes (resolution is genuinely what changes the verdict).
printf '%s\n' "$edge_base" \
    "  LINK_LIBRARIES = -lOwned" \
    "  LINK_PATH = -L$build_dir/late/lib" > "$edge"
check "DSO via late -L passes" pass "" run_audit "${sdk_providers[@]}"

# 3k-iib. A corrupt archive on the edge is unreadable evidence, not a pass.
echo "not-an-archive" > "$build_dir/lib/corrupt.a"
mk_edge "$edge_base lib/corrupt.a"
check "corrupt archive input" fail "unreadable-input" run_audit "${sdk_providers[@]}"

# 3k-iii. A copied direct OBJECT (not an archive) in the consumer's tree is
#      caught by content too.
cp "$sdk/support_impl.o" "$build_dir/lib/copied-object.o"
mk_edge "$edge_base lib/copied-object.o"
check "copied provider object in build tree" fail "embedded-definitions" \
    run_audit "${sdk_providers[@]}"

# 3k-iv. Plain-C third-party archives stay green even when a provider
#      exports the same C name (libLLVM/adler32 shape). Outside the
#      provider dirs, so only the content classification can fire.
mkdir -p "$build_dir/third-party/lib"
cp "$sdk/lib/libz-style.a" "$build_dir/third-party/lib/libz.a"
mk_edge "$edge_base obj/main.o third-party/lib/libz.a"
check "third-party C archive (adler32 shape) passes" pass "" run_audit "${sdk_providers[@]}"

# 3k. A stripped consumer with no Ninja evidence at all: fail closed.
#     (Only applicable to a stripped probe; an unstripped supplied consumer
#     legitimately passes on the symbol-table view alone.)
if readelf -SW "$probe" 2>/dev/null | grep -q '\.symtab'; then
    echo "NEG-TEST SKIP: probe is unstripped; no-evidence case N/A"
else
    check "stripped consumer, no CUDAQ_BUILD_NINJA" fail "stripped and no" \
        bash "$script_dir/check_mlir_symbols.sh" "$probe" "$cudaq_mlir" \
        "${sdk_providers[@]}" "${real_providers[@]}"
fi

# 3l. A dangling CUDAQ_BUILD_NINJA path is missing evidence: fail closed.
CUDAQ_BUILD_NINJA="$scratch/no-such-build.ninja" check \
    "stripped consumer, dangling ninja path" fail "does not exist" \
    bash "$script_dir/check_mlir_symbols.sh" "$probe" "$cudaq_mlir" \
        "${sdk_providers[@]}" "${real_providers[@]}"

# 3m. Positive: a clean edge (own objects, provider DSO, downstream
#     archive, system libraries; an order-only decoy after '||' is not a
#     link input).
mk_edge "$edge_base obj/main.o $sdk/lib/libMLIR.so.$sover lib/libTrivialDialect.a -lrt -ldl -lm | x || /nonexistent/order-only.a"
check "clean provenance edge" pass "" run_audit "${sdk_providers[@]}"

# 3n. The shared-provider layout has NO Support/Demangle exemption (the
#     substitution removed the need): those basenames under a provider dir
#     fail like every other component. The legacy static/monolithic layout
#     (provider set with no upstream DSOs) keeps the skip.
cp "$sdk/lib/libMLIRSupport.a" "$sdk/lib/libLLVMSupport.a"
cp "$sdk/lib/libMLIRSupport.a" "$sdk/lib/libLLVMDemangle.a"
mk_edge "$edge_base $sdk/lib/libLLVMSupport.a $sdk/lib/libLLVMDemangle.a | x"
check "no Support/Demangle exemption with shared providers" fail \
    "forbidden upstream component" run_audit "${sdk_providers[@]}"
# Static/monolithic layout: provider = cudaqMLIR (carries the LLVM impl
# itself, not an upstream basename, so no upstream dirs are derived).
c++ -shared -fPIC -o "$scratch/libcudaqMLIR.so" "$sdk/provider_impl.cpp" \
    -Wl,-soname,libcudaqMLIR.so || die "compiling static-mode provider"
CUDAQ_BUILD_NINJA="$edge" check "static-layout utility skip preserved" pass "" \
    bash "$script_dir/check_mlir_symbols.sh" "$probe" "$scratch/libcudaqMLIR.so"

# SDK97-6 regressions: supplied inputs must be byte-identical afterwards,
# and a consumer passed through a symlink is never written through.
if [ -z "$supplied_consumer" ]; then
    ln -s "$(realpath "$consumer")" "$scratch/linked-consumer.so"
    mk_edge "build python/x/linked-consumer.so: LINKER obj/main.o"
    before="$(sha256sum "$(realpath "$consumer")" | awk '{print $1}')"
    CUDAQ_BUILD_NINJA="$edge" check "symlinked consumer path" pass "" \
        bash "$script_dir/check_mlir_symbols.sh" "$scratch/linked-consumer.so" \
        "$cudaq_mlir" "${sdk_providers[@]}"
    after="$(sha256sum "$(realpath "$consumer")" | awk '{print $1}')"
    if [ "$before" != "$after" ]; then
        echo "NEG-TEST FAIL: consumer behind the symlink was modified"
        failures=$((failures + 1))
    fi
fi
for f in ${supplied_hash[@]+"${!supplied_hash[@]}"}; do
    now="$(sha256sum "$f" | awk '{print $1}')"
    if [ "$now" != "${supplied_hash[$f]}" ]; then
        echo "NEG-TEST FAIL: supplied input was modified: $f"
        failures=$((failures + 1))
    fi
done
[ -z "${supplied_hash[@]+set}" ] || echo "NEG-TEST OK: supplied inputs unchanged"

if [ "$failures" -ne 0 ]; then
    echo "NEG-TESTS: $failures failure(s)" >&2
    exit 1
fi
echo "NEG-TESTS: all passed"
