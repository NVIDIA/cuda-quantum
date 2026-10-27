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
#        [producer-build-tree]
# The build tree (unstripped providers + real build.ninja) enables the
# advanced pre-strip fixtures; without it they are skipped with a notice.

set -u
export LC_ALL=C

if [ $# -lt 1 ] || [ $# -gt 2 ] || [ ! -d "$1" ]; then
    echo "Usage: $0 <llvm-install-prefix-with-providers> [producer-build-tree]" >&2
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

# expect_fail_builddir <name> <expected-message-substring> <build.ninja content>
# Runs against a farm of the good prefix with a synthetic build directory.
expect_fail_builddir() {
    local name="$1" expect="$2" ninja="$3"
    local farm="$scratch/farm"
    rm -rf "$farm"
    cp -as "$prefix" "$farm"
    local fake="$scratch/fake-build"
    rm -rf "$fake" && mkdir -p "$fake/lib"
    printf '%s\n' "$ninja" > "$fake/build.ninja"
    local out rc
    out="$(bash "$script_dir/check_mlir_providers.sh" "$farm" --build-dir "$fake" 2>&1)" && rc=0 || rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "NEG-TEST FAIL: $name: audit passed, expected failure"
        failures=$((failures + 1)); return
    fi
    if ! grep -qF "$expect" <<< "$out"; then
        echo "NEG-TEST FAIL: $name: missing message '$expect' in:"; echo "$out" | head -4
        failures=$((failures + 1)); return
    fi
    echo "NEG-TEST OK: $name"
}

sover="$("$prefix/bin/llvm-config" --version | sed 's/\.[0-9]*$//')"

# 6. Build dir lacks the unstripped providers (clean ninja file).
expect_fail_builddir "build dir without unstripped providers" \
    "build tree lacks unstripped" ""

# 7. A component archive in an MLIR aggregate's explicit link inputs.
expect_fail_builddir "archive in libMLIR-C link inputs" \
    "link inputs contain component archives" \
    "build lib/libMLIR-C.so.$sover: LINKER foo.o lib/libMLIRPass.a | lib/libMLIR.so.$sover"

# 8. An MLIR archive embedded into the LLVM aggregate's link inputs.
expect_fail_builddir "MLIR archive in libLLVM link inputs" \
    "libLLVM's link inputs contain MLIR component archives" \
    "build lib/libLLVM.so.$sover: LINKER foo.o lib/libMLIRSupport.a"

# 8b. An archive in the implicit ('|') input section — the realistic Ninja
# layout (CMake puts LINK_LIBRARIES there); the scan must not stop at '|'.
expect_fail_builddir "archive in libMLIR-C implicit inputs" \
    "link inputs contain component archives" \
    "build lib/libMLIR-C.so.$sover: LINKER foo.o | lib/libMLIRPass.a lib/libMLIR.so.$sover"

# 8c. The provider's edge is missing from build.ninja: missing evidence fails.
expect_fail_builddir "missing link edge" "no link edge" \
    "build lib/libMLIR.so.$sover: LINKER foo.o"

# 8d. Unresolved evidence in the scanned inputs (rsp/variable) fails.
expect_fail_builddir "unresolved rsp link input" "unresolved link-input evidence" \
    "build lib/libMLIR.so.$sover: LINKER foo.o libLLVM.rsp | lib/libLLVM.so.$sover"

# 8d-ii. A small component archive RENAMED into a neutral location and
# linked by a non-owning provider: the basename rules cannot see it, but
# its strong definitions duplicate another provider's exports.
if [ -f "$prefix/lib/libLLVMSupport.a" ]; then
    fake_copy="$scratch/fake-copy-build"
    rm -rf "$fake_copy" && mkdir -p "$fake_copy/deps"
    cp "$prefix/lib/libLLVMSupport.a" "$fake_copy/deps/support-copy.a"
    for p in LLVM MLIR MLIRExecutionEngineShared; do
        printf 'build lib/lib%s.so.%s: LINKER foo.o\n' "$p" "$sover"
    done > "$fake_copy/build.ninja"
    printf 'build lib/libMLIR-C.so.%s: LINKER foo.o deps/support-copy.a\n' "$sover" \
        >> "$fake_copy/build.ninja"
    out="$(bash "$script_dir/check_mlir_providers.sh" "$prefix" --build-dir "$fake_copy" 2>&1)" && rc=0 || rc=$?
    if [ "$rc" -ne 0 ] && grep -qF "embedding another provider's implementation" <<< "$out"; then
        echo "NEG-TEST OK: renamed component archive caught by content"
    else
        echo "NEG-TEST FAIL: renamed component archive caught by content"; echo "$out" | head -3
        failures=$((failures + 1))
    fi
else
    echo "NEG-TEST SKIP: no libLLVMSupport.a for the renamed-copy fixture"
fi

# 8e-i. Referenced-but-missing and corrupt provider-edge inputs fail
#       closed (existence + successful inspection are required).
expect_fail_builddir "missing provider-edge input" "missing link-input evidence" \
    "build lib/libMLIR-C.so.$sover: LINKER foo.o lib/no-such.a | lib/libMLIR.so.$sover\nbuild lib/libMLIR.so.$sover: LINKER foo.o\nbuild lib/libMLIRExecutionEngineShared.so.$sover: LINKER foo.o\nbuild lib/libLLVM.so.$sover: LINKER foo.o"
_fake_corrupt="$scratch/fake-corrupt"
rm -rf "$_fake_corrupt" && mkdir -p "$_fake_corrupt/deps"
echo "not-an-archive" > "$_fake_corrupt/deps/corrupt.a"
printf 'build lib/libMLIR-C.so.%s: LINKER foo.o deps/corrupt.a\n' "$sover" > "$_fake_corrupt/build.ninja"
for p in MLIR MLIRExecutionEngineShared LLVM; do
    printf 'build lib/lib%s.so.%s: LINKER foo.o\n' "$p" "$sover" >> "$_fake_corrupt/build.ninja"
done
out="$(bash "$script_dir/check_mlir_providers.sh" "$prefix" --build-dir "$_fake_corrupt" 2>&1)" && rc=0 || rc=$?
if [ "$rc" -ne 0 ] && grep -qF "could not inspect" <<< "$out"; then
    echo "NEG-TEST OK: corrupt provider-edge input fails the inspection"
else
    echo "NEG-TEST FAIL: corrupt provider-edge input fails the inspection"; echo "$out" | head -3
    failures=$((failures + 1))
fi

# 8e. No build.ninja at all: missing evidence fails.
fake_noninja="$scratch/fake-no-ninja"
rm -rf "$fake_noninja" && mkdir -p "$fake_noninja"
out="$(bash "$script_dir/check_mlir_providers.sh" "$prefix" --build-dir "$fake_noninja" 2>&1)" && rc=0 || rc=$?
if [ "$rc" -ne 0 ] && grep -qF "no build.ninja under" <<< "$out"; then
    echo "NEG-TEST OK: missing build.ninja"
else
    echo "NEG-TEST FAIL: missing build.ninja"; failures=$((failures + 1))
fi

# 8f. Stripped providers offered as build-tree evidence: the pre-strip audit
# requires full symbol tables and fails closed on stripped input.
fake_stripped="$scratch/fake-stripped"
rm -rf "$fake_stripped" && mkdir -p "$fake_stripped/lib"
for p in LLVM MLIR MLIR-C MLIRExecutionEngineShared; do
    cp -L "$prefix/lib/lib${p}.so.$sover" "$fake_stripped/lib/" 2>/dev/null \
        || cp -L "$prefix/lib/lib${p}.so" "$fake_stripped/lib/lib${p}.so.$sover"
done
printf '%s\n' "build lib/libLLVM.so.$sover: LINKER foo.o" \
    "build lib/libMLIR.so.$sover: LINKER foo.o" \
    "build lib/libMLIR-C.so.$sover: LINKER foo.o" \
    "build lib/libMLIRExecutionEngineShared.so.$sover: LINKER foo.o" \
    > "$fake_stripped/build.ninja"
out="$(bash "$script_dir/check_mlir_providers.sh" "$prefix" --build-dir "$fake_stripped" 2>&1)" && rc=0 || rc=$?
if [ "$rc" -ne 0 ] && grep -qF "lacks a full symbol table" <<< "$out"; then
    echo "NEG-TEST OK: stripped provider evidence"
else
    echo "NEG-TEST FAIL: stripped provider evidence"; echo "$out" | head -3
    failures=$((failures + 1))
fi

# 12. Loadable-content digest (the build→install binding primitive): a
#     strip/install transformation leaves the digest unchanged, as does a
#     RUNPATH-only change (the one declared install transformation);
#     touching a loadable byte, a non-RUNPATH .dynstr string, or a NOBITS
#     descriptor changes it; non-loadable input fails closed.
digest_of() { bash "$script_dir/check_mlir_providers.sh" --load-digest "$1"; }
probe_lib="$scratch/probe.so"
cp -L "$(readlink -f "$prefix/lib/libMLIR-C.so.$sover")" "$probe_lib"
d_plain="$(digest_of "$probe_lib")"
d_plain2="$(digest_of "$probe_lib")"
strip "$probe_lib" -o "$scratch/probe-stripped.so"
d_stripped="$(digest_of "$scratch/probe-stripped.so")"
exec_off="$(readelf -lW "$scratch/probe-stripped.so" \
    | awk '$1 == "LOAD" && $0 ~ / E / {print $2; exit}')"
cp "$scratch/probe-stripped.so" "$scratch/probe-mutated.so"
printf 'X' | dd of="$scratch/probe-mutated.so" bs=1 \
    seek=$((16#${exec_off#0x} + 64)) conv=notrunc 2>/dev/null
d_mutated="$(digest_of "$scratch/probe-mutated.so")"
d_garbage="$(digest_of /dev/null || true)"
# 12b. Mutations exercised on scratch copies only: a non-RUNPATH .dynstr
#      string flip must change the digest; a RUNPATH-value flip must not
#      (that exact transformation is what the binding tolerates); a .bss
#      size-descriptor flip must change it.
python3 - "$scratch/probe-stripped.so" <<'PYEOF'
import shutil, struct, sys
src = sys.argv[1]
def secs_of(data):
    e_shoff = struct.unpack_from("<Q", data, 0x28)[0]
    entsz, num, strndx = struct.unpack_from("<HHH", data, 0x3A)
    secs = [struct.unpack_from("<IIQQQQ", data, e_shoff + i * entsz)
            for i in range(num)]
    _, _, _, _, sht_off, sht_size = secs[strndx]
    shstr = bytes(data[sht_off:sht_off + sht_size])
    def name(s):
        n = s[0]
        return shstr[n:shstr.index(b"\x00", n)].decode()
    return secs, name
base = bytearray(open(src, "rb").read())
secs, name = secs_of(base)
dynstr = next(s for s in secs if name(s) == ".dynstr")
soname_off = bytes(base).find(b"libMLIR-C.so", dynstr[4], dynstr[4] + dynstr[5])
runpath_off = bytes(base).find(b"$ORIGIN", dynstr[4], dynstr[4] + dynstr[5])
assert soname_off > 0 and runpath_off > 0
bss_i = next(i for i, s in enumerate(secs) if name(s) == ".bss")
m = bytearray(base); m[soname_off + 2] ^= 0x01
open(src + ".dynstr-mut", "wb").write(bytes(m))
m = bytearray(base); m[runpath_off + 1] = ord("X")
open(src + ".runpath-mut", "wb").write(bytes(m))
m = bytearray(base)
e_shoff = struct.unpack_from("<Q", m, 0x28)[0]
entsz = struct.unpack_from("<H", m, 0x3A)[0]
struct.pack_into("<Q", m, e_shoff + bss_i * entsz + 0x20, secs[bss_i][5] + 16)
open(src + ".bss-mut", "wb").write(bytes(m))
# A PT_LOAD permission flip (executable mapping made non-executable) must
# change the digest even though section bytes are untouched.
m = bytearray(base)
e_phoff = struct.unpack_from("<Q", m, 0x20)[0]
phentsz, phnum = struct.unpack_from("<HH", m, 0x36)
for i in range(phnum):
    off = e_phoff + i * phentsz
    p_type, p_flags = struct.unpack_from("<II", m, off)
    if p_type == 1 and p_flags & 0x1:
        struct.pack_into("<I", m, off + 4, p_flags & ~0x1)
        break
open(src + ".phdr-mut", "wb").write(bytes(m))
# A PT_LOAD file-offset flip (remapping which bytes the loader maps) must
# change the digest too.
m = bytearray(base)
e_phoff = struct.unpack_from("<Q", m, 0x20)[0]
phentsz, phnum = struct.unpack_from("<HH", m, 0x36)
for i in range(phnum):
    off = e_phoff + i * phentsz
    p_type, p_flags, p_off = struct.unpack_from("<IIQ", m, off)
    if p_type == 1 and p_off:
        struct.pack_into("<Q", m, off + 8, p_off + 0x100)
        break
open(src + ".off-mut", "wb").write(bytes(m))
PYEOF
d_dynstr="$(digest_of "$scratch/probe-stripped.so.dynstr-mut")"
d_runpath="$(digest_of "$scratch/probe-stripped.so.runpath-mut")"
d_bss="$(digest_of "$scratch/probe-stripped.so.bss-mut")"
d_phdr="$(digest_of "$scratch/probe-stripped.so.phdr-mut")"
d_off="$(digest_of "$scratch/probe-stripped.so.off-mut")"
if [ -n "$d_plain" ] && [ "$d_plain" = "$d_plain2" ] \
    && [ "$d_plain" = "$d_stripped" ] && [ -n "$d_mutated" ] \
    && [ "$d_mutated" != "$d_plain" ] && [ -z "$d_garbage" ] \
    && [ -n "$d_dynstr" ] && [ "$d_dynstr" != "$d_stripped" ] \
    && [ "$d_runpath" = "$d_stripped" ] \
    && [ -n "$d_bss" ] && [ "$d_bss" != "$d_stripped" ] \
    && [ -n "$d_phdr" ] && [ "$d_phdr" != "$d_stripped" ] \
    && [ -n "$d_off" ] && [ "$d_off" != "$d_stripped" ]; then
    echo "NEG-TEST OK: load digest is strip-invariant, deterministic, content-sensitive"
else
    echo "NEG-TEST FAIL: load digest semantics (plain=$d_plain stripped=$d_stripped" \
         "mutated=$d_mutated dynstr=$d_dynstr runpath=$d_runpath bss=$d_bss)"
    failures=$((failures + 1))
fi

# Advanced fixtures: real unstripped providers from the producer build tree
# and a compiled embedded-copy DSO exercising the pre-strip population gates.
producer_build="${2:-}"
if [ -n "$producer_build" ] && [ -d "$producer_build/lib" ] && command -v c++ >/dev/null 2>&1 \
        && [ -f "$prefix/lib/libMLIRIR.a" ]; then
    real_unstripped() { find "$producer_build" -name "lib$1.so.$sover" | head -1; }

    # 9. A static implementation localized by --exclude-libs,ALL: invisible
    # to every dynamic scan; the full symbol table exposes the population.
    fake_hidden="$scratch/fake-hidden"
    rm -rf "$fake_hidden" && mkdir -p "$fake_hidden/lib"
    for p in LLVM MLIR MLIRExecutionEngineShared; do
        cp -L "$(real_unstripped "$p")" "$fake_hidden/lib/" || true
done
    cp -L "$(real_unstripped MLIR)" "$fake_hidden/lib/"
    cat > "$fake_hidden/fixture.cpp" <<'EOF'
// Anchor TU: --whole-archive pulls every libMLIRIR.a member into the DSO.
namespace mlir { namespace { int anchor() { return 0; } } }
int fixture_anchor() { return 0; }
EOF
    c++ -shared -fPIC -o "$fake_hidden/lib/libMLIR-C.so.$sover" \
        "$fake_hidden/fixture.cpp" \
        -Wl,--whole-archive "$prefix/lib/libMLIRIR.a" -Wl,--no-whole-archive \
        -Wl,--exclude-libs,ALL
    printf '%s\n' "build lib/libLLVM.so.$sover: LINKER foo.o" \
        "build lib/libMLIR.so.$sover: LINKER foo.o" \
        "build lib/libMLIR-C.so.$sover: LINKER foo.o" \
        "build lib/libMLIRExecutionEngineShared.so.$sover: LINKER foo.o" \
        > "$fake_hidden/build.ninja"
    out="$(bash "$script_dir/check_mlir_providers.sh" "$prefix" --build-dir "$fake_hidden" 2>&1)" && rc=0 || rc=$?
    if [ "$rc" -ne 0 ] && grep -qF "embedded static implementation is localized" <<< "$out"; then
        echo "NEG-TEST OK: hidden localized copy population"
    else
        echo "NEG-TEST FAIL: hidden localized copy population"; echo "$out" | head -3
        failures=$((failures + 1))
    fi

    # 10. The same embedded copy WITHOUT localization: interposable
    # GLOBAL/DEFAULT definitions shared with the owning provider.
    fake_dup="$scratch/fake-dup"
    rm -rf "$fake_dup" && mkdir -p "$fake_dup/lib"
    for p in LLVM MLIR MLIRExecutionEngineShared; do
        cp -L "$(real_unstripped "$p")" "$fake_dup/lib/" || true
    done
    cp -L "$(real_unstripped MLIR)" "$fake_dup/lib/"
    c++ -shared -fPIC -o "$fake_dup/lib/libMLIR-C.so.$sover" \
        "$fake_hidden/fixture.cpp" \
        -Wl,--whole-archive "$prefix/lib/libMLIRIR.a" -Wl,--no-whole-archive
    printf '%s\n' "build lib/libLLVM.so.$sover: LINKER foo.o" \
        "build lib/libMLIR.so.$sover: LINKER foo.o" \
        "build lib/libMLIR-C.so.$sover: LINKER foo.o" \
        "build lib/libMLIRExecutionEngineShared.so.$sover: LINKER foo.o" \
        > "$fake_dup/build.ninja"
    out="$(bash "$script_dir/check_mlir_providers.sh" "$prefix" --build-dir "$fake_dup" 2>&1)" && rc=0 || rc=$?
    if [ "$rc" -ne 0 ] && grep -qF "pre-strip interposable definitions shared by" <<< "$out"; then
        echo "NEG-TEST OK: interposable duplicate provider definitions"
    else
        echo "NEG-TEST FAIL: interposable duplicate provider definitions"; echo "$out" | head -3
        failures=$((failures + 1))
    fi

    # 11. Positive control with the real build tree: the full audit passes.
    if bash "$script_dir/check_mlir_providers.sh" "$prefix" --build-dir "$producer_build" \
            > /dev/null 2>&1; then
        echo "NEG-TEST OK: real build tree passes the full audit"
    else
        echo "NEG-TEST FAIL: real build tree must pass the full audit"
        failures=$((failures + 1))
    fi
else
    echo "NEG-TEST SKIP: advanced pre-strip fixtures need <producer-build-tree> and a C++ compiler"
fi

if [ "$failures" -ne 0 ]; then
    echo "NEG-TESTS: $failures failure(s)" >&2
    exit 1
fi
echo "NEG-TESTS: all passed"
