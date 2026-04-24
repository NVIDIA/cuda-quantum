#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Smoke test for an installed CUDA-Q deb package.
#
# Run inside a fresh Debian/Ubuntu container after the deb has been installed
# via `apt install ./<file>.deb`. The script verifies:
#
#   1. dpkg sees the package as installed
#   2. Files landed under the expected install prefix
#   3. The /etc/profile.d drop-in exports CUDAQ_INSTALL_PATH and friends
#   4. nvq++ runs (--version)
#   5. (core only) nvq++ compiles and runs a CPU-only Bell-state sample
#   6. apt remove cleans the prefix + profile.d drop-in
#
# The realtime flavor skips the compile-and-run step because libcudaq-realtime
# requires GPU + Holoscan/DOCA runtimes that aren't available on CI validation
# runners. Install/uninstall correctness is still covered.
#
# Usage:
#   bash scripts/validate_deb.sh -c 12 -f core
#   bash scripts/validate_deb.sh -c 13 -f realtime
#
# Options:
#   -c <cuda_major>   12 or 13. Required.
#   -f <flavor>       core (default) or realtime.

set -euo pipefail

cuda_major=""
flavor="core"

__optind__=$OPTIND
OPTIND=1
while getopts ":c:f:" opt; do
  case $opt in
    c) cuda_major="$OPTARG" ;;
    f) flavor="$OPTARG" ;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
    :)  echo "Option -$OPTARG requires an argument" >&2; exit 1 ;;
  esac
done
OPTIND=$__optind__

[ -n "$cuda_major" ] || { echo "Error: -c <cuda_major> required" >&2; exit 1; }
[ "$cuda_major" = "12" ] || [ "$cuda_major" = "13" ] || {
  echo "Error: cuda_major must be 12 or 13" >&2; exit 1; }

case "$flavor" in
  core)
    pkg_name="cuda-quantum-cu${cuda_major}"
    prefix="/opt/nvidia/cudaq"
    profile="/etc/profile.d/cuda-quantum-cu${cuda_major}.sh"
    ;;
  realtime)
    pkg_name="cuda-quantum-realtime-cu${cuda_major}"
    prefix="/opt/nvidia/cudaq-realtime"
    profile="/etc/profile.d/cuda-quantum-realtime-cu${cuda_major}.sh"
    ;;
  *)
    echo "Error: -f must be 'core' or 'realtime'" >&2; exit 1 ;;
esac

fail() { echo "FAIL: $*" >&2; exit 1; }
pass() { echo "PASS: $*"; }

# ---------------------------------------------------------------------------
# 1. dpkg sees the package
# ---------------------------------------------------------------------------
dpkg -l "$pkg_name" 2>/dev/null | awk 'NR>5 {print $1}' | grep -q '^ii$' \
  || fail "$pkg_name is not in dpkg state ii"
pass "$pkg_name installed (dpkg -l)"

# ---------------------------------------------------------------------------
# 2. Files landed at the right prefix
# ---------------------------------------------------------------------------
[ -x "$prefix/bin/nvq++" ]  || fail "missing $prefix/bin/nvq++"
[ -f "$prefix/set_env.sh" ] || fail "missing $prefix/set_env.sh"
[ -f "$profile" ]           || fail "missing $profile"
pass "install prefix $prefix populated"

# ---------------------------------------------------------------------------
# 3. profile.d drop-in exports CUDAQ_INSTALL_PATH (in a fresh subshell so we
#    don't leak env into the rest of the script)
# ---------------------------------------------------------------------------
exported=$(bash -c ". $profile && echo \$CUDAQ_INSTALL_PATH")
[ "$exported" = "$prefix" ] || fail "profile.d did not export CUDAQ_INSTALL_PATH=$prefix (got '$exported')"
pass "profile.d drop-in exports CUDAQ_INSTALL_PATH"

# ---------------------------------------------------------------------------
# 4. nvq++ --version
# ---------------------------------------------------------------------------
. "$profile"
"$prefix/bin/nvq++" --version || fail "nvq++ --version failed"
pass "nvq++ --version succeeded"

# ---------------------------------------------------------------------------
# 5. Compile + run a CPU-only sample (core flavor only)
# ---------------------------------------------------------------------------
if [ "$flavor" = "core" ]; then
    tmpdir=$(mktemp -d)
    trap 'rm -rf "$tmpdir"' EXIT

    cat > "$tmpdir/bell.cpp" <<'EOF'
#include <cudaq.h>
#include <iostream>

__qpu__ void bell() {
    cudaq::qubit q, r;
    h(q);
    x<cudaq::ctrl>(q, r);
    mz(q);
    mz(r);
}

int main() {
    auto result = cudaq::sample(bell);
    std::cout << "bell counts: " << result.size() << " distinct outcomes\n";
    result.dump();
    return 0;
}
EOF

    "$prefix/bin/nvq++" --target qpp-cpu "$tmpdir/bell.cpp" -o "$tmpdir/bell" \
        || fail "nvq++ failed to compile bell.cpp"
    pass "nvq++ compiled CPU-only Bell sample"

    "$tmpdir/bell" || fail "compiled bell binary failed at runtime"
    pass "compiled binary executed successfully"
fi

# ---------------------------------------------------------------------------
# 6. apt remove cleans up
# ---------------------------------------------------------------------------
apt-get remove -y "$pkg_name" >/dev/null
[ ! -e "$prefix/bin/nvq++" ] || fail "$prefix/bin/nvq++ remained after apt remove"
[ ! -e "$profile" ]          || fail "$profile remained after apt remove"
pass "$pkg_name removed cleanly"

echo "OK: $pkg_name validated end-to-end"
