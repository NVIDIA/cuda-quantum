#!/bin/bash

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Negative/positive tests for scripts/check_provider_residency.sh: SONAME
# parsing, fail-closed error handling, canonical containment, and the
# expected-provider evidence requirement. A real compiled binary + provider
# DSOs exercise ldd itself; a PATH-shimmed ldd supplies canned outputs for
# the synthetic negatives.
#
# Usage: test_check_provider_residency.sh   (self-contained; no arguments)

set -eu
export LC_ALL=C

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
failures=0
die() { echo "FIXTURE ERROR: $*" >&2; exit 2; }

sover=99.0   # derived from the fixture llvm-config's 99.0.1 (major.minor)

# Fixture prefix with fake providers and an llvm-config.
prefix="$scratch/prefix"
mkdir -p "$prefix/bin" "$prefix/lib"
cat > "$prefix/bin/llvm-config" <<EOF
#!/bin/bash
[ "\$1" = "--version" ] && echo 99.0.1
EOF
chmod +x "$prefix/bin/llvm-config"
providers=(libMLIR.so.$sover libMLIR-C.so.$sover
           libMLIRExecutionEngineShared.so.$sover libLLVM.so.$sover)
for l in "${providers[@]}"; do
    echo "int ${l//[.-]/_}_anchor() { return 0; }" | tr -d '\n' > "$scratch/impl.c"
    echo >> "$scratch/impl.c"
    c++ -shared -fPIC -o "$prefix/lib/$l" "$scratch/impl.c" -Wl,-soname,"$l" \
        || die "compiling $l"
done

# A real consumer binary with DT_NEEDED on all four providers, RPATH into
# the prefix.
cat > "$scratch/main.c" <<'EOF'
int main(void) { return 0; }
EOF
c++ -o "$scratch/consumer" "$scratch/main.c" \
    -Wl,--no-as-needed -Wl,-rpath,"$prefix/lib" \
    -L"$prefix/lib" "${providers[@]/#/-l:}" || die "compiling consumer"

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

# 1. Real ldd, real providers inside the prefix: passes.
check "resident providers (real ldd)" pass "" \
    bash "$script_dir/check_provider_residency.sh" "$scratch/consumer" "$prefix"

# A PATH-shimmed ldd for canned outputs.
shim="$scratch/shim"
mkdir -p "$shim"
write_shim() { printf '#!/bin/bash\ncat <<'"'"'EOF'"'"'\n%s\nEOF\n' "$1" > "$shim/ldd"; chmod +x "$shim/ldd"; }
run_shimmed() {   # run_shimmed <ldd-output-file> <prefix>
    PATH="$shim:$PATH" bash "$script_dir/check_provider_residency.sh" \
        "$scratch/consumer" "$2" < /dev/null
}
ldd_line() { printf '\t%s => %s (0x00007f000000)\n' "$1" "$2"; }

# 2. Foreign resolution: provider path outside the prefix must fail.
mkdir -p "$scratch/foreign/lib"
cp "$prefix/lib/${providers[3]}" "$scratch/foreign/lib/"
{
    for l in "${providers[@]:0:3}"; do ldd_line "$l" "$prefix/lib/$l"; done
    ldd_line "${providers[3]}" "$scratch/foreign/lib/${providers[3]}"
} > "$scratch/foreign.txt"
write_shim "$(cat "$scratch/foreign.txt")"
check "foreign provider path" fail "resolves outside the prefix" \
    run_shimmed "$scratch/foreign.txt" "$prefix"

# 3. A 'not found' provider is an error, not a pass.
{
    for l in "${providers[@]:0:3}"; do ldd_line "$l" "$prefix/lib/$l"; done
    printf '\t%s => not found\n' "${providers[3]}"
} > "$scratch/notfound.txt"
write_shim "$(cat "$scratch/notfound.txt")"
check "provider not found" fail "unresolved" run_shimmed "$scratch/notfound.txt" "$prefix"

# 4. Missing expected-provider evidence (libLLVM absent from the output).
{
    for l in "${providers[@]:0:3}"; do ldd_line "$l" "$prefix/lib/$l"; done
} > "$scratch/missing.txt"
write_shim "$(cat "$scratch/missing.txt")"
check "missing expected provider" fail "missing evidence" \
    run_shimmed "$scratch/missing.txt" "$prefix"

# 5. ldd itself failing must fail the gate (no silent pass).
printf '#!/bin/bash\nexit 1\n' > "$shim/ldd"; chmod +x "$shim/ldd"
check "ldd failure" fail "ldd failed" run_shimmed /dev/null "$prefix"

# 6. Empty ldd output: no evidence at all.
printf '#!/bin/bash\nexit 0\n' > "$shim/ldd"; chmod +x "$shim/ldd"
check "empty ldd output" fail "no output" run_shimmed /dev/null "$prefix"

# 7. Loader/vdso lines without SONAMEs are not provider evidence; the
#    residency verdict must come from the => lines.
{
    printf '\tlinux-vdso.so.1 (0x00007ffe00000)\n'
    for l in "${providers[@]}"; do ldd_line "$l" "$prefix/lib/$l"; done
    printf '\t/lib64/ld-linux-x86-64.so.2 (0x00007f100000)\n'
} > "$scratch/loaderlines.txt"
write_shim "$(cat "$scratch/loaderlines.txt")"
check "loader/vdso lines skipped, providers pass" pass "" \
    run_shimmed "$scratch/loaderlines.txt" "$prefix"

# 8. Symlink escape: an in-prefix-looking path that canonicalizes outside
#    must fail (canonical containment, not textual).
mkdir -p "$scratch/foreign-real"
cp "$prefix/lib/${providers[3]}" "$scratch/foreign-real/"
rm "$prefix/lib/${providers[3]}"
ln -s "$scratch/foreign-real/${providers[3]}" "$prefix/lib/${providers[3]}"
check "symlink escaping the prefix" fail "resolves outside the prefix" \
    bash "$script_dir/check_provider_residency.sh" "$scratch/consumer" "$prefix"
mv "$scratch/foreign-real/${providers[3]}" "$prefix/lib/${providers[3]}"

# 9. Textual-prefix sibling trap: P-other must not count as inside P.
#    $prefix-other shares a textual prefix with $prefix.
other="${prefix}-other"
mkdir -p "$other/lib"
cp "$prefix/lib/${providers[3]}" "$other/lib/"
{
    for l in "${providers[@]:0:3}"; do ldd_line "$l" "$prefix/lib/$l"; done
    ldd_line "${providers[3]}" "$other/lib/${providers[3]}"
} > "$scratch/sibling.txt"
write_shim "$(cat "$scratch/sibling.txt")"
check "textual-prefix sibling (P-other)" fail "resolves outside the prefix" \
    run_shimmed "$scratch/sibling.txt" "$prefix"

if [ "$failures" -ne 0 ]; then
    echo "NEG-TESTS: $failures failure(s)" >&2
    exit 1
fi
echo "NEG-TESTS: all passed"
