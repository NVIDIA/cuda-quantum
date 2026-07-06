#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Verifies that CUDA-Q distributions comply with the LGPL v3 requirements for
# the redistributed GMP and MPFR shared libraries:
#   1. The full LGPL v3 and GPL v3 license texts and the NOTICE file with the
#      GMP/MPFR copyright notices are shipped with the distribution.
#   2. GMP and MPFR are shipped as shared libraries only (no static archives).
#   3. CUDA-Q binaries reference GMP/MPFR exclusively via dynamic linking: at
#      least one shipped binary declares a runtime dependency on them, and no
#      shipped binary exports GMP/MPFR symbols itself (which would indicate an
#      accidentally statically linked copy).
#   4. The libraries can be replaced by substituting the shared library files:
#      Clifford+T rotation synthesis works, stops working when the shipped
#      libgmp is replaced with an invalid file (proving the file is really
#      loaded at runtime rather than a hidden static copy being used), and
#      works again once a valid replacement is put in place.
#
# Usage:
#   validate_license_compliance.sh [install_root]
#     Validates a CUDA-Q C++ installation (container or installer).
#     install_root defaults to $CUDA_QUANTUM_PATH.
#   validate_license_compliance.sh --wheel
#     Validates an installed CUDA-Q Python wheel; the installation is located
#     via the python3 on the current PATH.

failures=0
report() {
    if [ "$1" -eq 0 ]; then
        echo "  [OK] $2"
    else
        echo -e "  \e[01;31m[FAILED] $2\e[0m" >&2
        failures=$((failures + 1))
    fi
}

case "$(uname -s)" in
    Darwin) is_darwin=true ;;
    *) is_darwin=false ;;
esac

wheel_mode=false
if [ "${1:-}" == "--wheel" ]; then
    wheel_mode=true
fi

echo "Validating LGPL compliance for the redistributed GMP/MPFR libraries."

# Locate the license texts, the bundled libraries, and the CUDA-Q binaries
# that need to be scanned for references to them.
if $wheel_mode; then
    site_packages=$(python3 -c "import cudaq, os; print(os.path.dirname(os.path.dirname(os.path.abspath(cudaq.__file__))))" 2>/dev/null)
    if [ -z "$site_packages" ]; then
        echo -e "\e[01;31mError: failed to locate the installed cudaq Python package.\e[0m" >&2
        exit 10
    fi
    echo "Validating Python wheel installation in $site_packages."
    lib_dir="$site_packages/lib"
    # PEP 639 places the entries of license-files in dist-info/licenses/,
    # preserving their relative paths.
    dist_info_licenses=$(ls -d "$site_packages"/cuda_quantum*.dist-info/licenses 2>/dev/null | head -1)
    licenses_dir="$dist_info_licenses/LICENSES"
    notice_file="$dist_info_licenses/NOTICE"
    scan_dirs="$lib_dir $site_packages/cudaq"
else
    install_root="${1:-$CUDA_QUANTUM_PATH}"
    if [ -z "$install_root" ] || [ ! -d "$install_root" ]; then
        echo -e "\e[01;31mError: no CUDA-Q installation found; pass the install root or set CUDA_QUANTUM_PATH.\e[0m" >&2
        exit 10
    fi
    echo "Validating CUDA-Q installation in $install_root."
    lib_dir="$install_root/lib"
    licenses_dir="$install_root/LICENSES"
    notice_file="$install_root/NOTICE"
    scan_dirs="$lib_dir $install_root/bin"
fi

echo "Checking that the license texts are distributed..."
[ -s "$licenses_dir/LICENSE.LGPLv3" ]
report $? "LGPL v3 text at $licenses_dir/LICENSE.LGPLv3"
[ -s "$licenses_dir/LICENSE.GPLv3" ]
report $? "GPL v3 text at $licenses_dir/LICENSE.GPLv3"
[ -f "$notice_file" ] && grep -q "GMP" "$notice_file" && grep -q "MPFR" "$notice_file"
report $? "NOTICE with GMP and MPFR copyright notices at $notice_file"

echo "Checking that GMP and MPFR are shipped as shared libraries..."
if $is_darwin; then
    gmp_libs=$(ls "$lib_dir"/libgmp*.dylib 2>/dev/null)
    mpfr_libs=$(ls "$lib_dir"/libmpfr*.dylib 2>/dev/null)
else
    gmp_libs=$(ls "$lib_dir"/libgmp.so* 2>/dev/null)
    mpfr_libs=$(ls "$lib_dir"/libmpfr.so* 2>/dev/null)
fi
[ -n "$gmp_libs" ]
report $? "GMP shared library in $lib_dir"
[ -n "$mpfr_libs" ]
report $? "MPFR shared library in $lib_dir"
static_copies=$(find $scan_dirs -name 'libgmp*.a' -o -name 'libmpfr*.a' 2>/dev/null)
[ -z "$static_copies" ]
report $? "no static GMP/MPFR archives are shipped${static_copies:+ (found: $static_copies)}"

echo "Checking that GMP/MPFR are referenced via dynamic linking only..."
# List the runtime dependencies of a binary.
runtime_deps() {
    if $is_darwin; then
        otool -L "$1" 2>/dev/null
    else
        ldd "$1" 2>/dev/null
    fi
}
# List the dynamic symbols a binary defines (not the ones it imports).
defined_dynamic_syms() {
    if $is_darwin; then
        nm -gU "$1" 2>/dev/null
    else
        nm -D --defined-only "$1" 2>/dev/null
    fi
}
# All GMP exports start with __gmp (per-type prefixes __gmpz_, __gmpn_, ...)
# or gmp_; all MPFR exports start with mpfr_. macOS prepends an underscore.
sym_pattern='__gmp|_?mpfr_'
# The set of shipped CUDA-Q binaries to scan. maxdepth 1 keeps the bundled
# LLVM subtree (lib/llvm in the installer) out of scope; the Python
# extension modules live deeper inside the cudaq package, so scan that
# subtree fully.
if $wheel_mode; then
    scan_files=$(find "$lib_dir" -maxdepth 1 \( -name '*.so*' -o -name '*.dylib' \) 2>/dev/null; \
                 find "$site_packages/cudaq" \( -name '*.so*' -o -name '*.dylib' \) 2>/dev/null)
else
    scan_files=$(find "$lib_dir" -maxdepth 1 \( -name '*.so*' -o -name '*.dylib' \) 2>/dev/null; \
                 find "$install_root/bin" -maxdepth 1 -type f 2>/dev/null)
fi
dynamic_ref_found=false
static_link_found=false
if command -v nm >/dev/null 2>&1; then
    for f in $scan_files; do
        case "$(basename "$f")" in
            libgmp*|libmpfr*) continue ;;
        esac
        if runtime_deps "$f" | grep -qE 'libgmp|libmpfr'; then
            dynamic_ref_found=true
        fi
        if defined_dynamic_syms "$f" | grep -qE " ($sym_pattern)"; then
            echo "  Found GMP/MPFR symbols statically linked into $f." >&2
            static_link_found=true
        fi
    done
    $dynamic_ref_found
    report $? "at least one CUDA-Q binary depends on GMP/MPFR dynamically"
    ! $static_link_found
    report $? "no CUDA-Q binary contains a statically linked GMP/MPFR copy"
else
    echo -e "\e[01;31mError: nm not available; cannot verify dynamic linking.\e[0m" >&2
    report 1 "dynamic linking verification (nm not available)"
fi

echo "Checking that the GMP/MPFR libraries can be replaced..."
if [ -z "$gmp_libs" ]; then
    # The missing library was already reported as a failure above.
    echo "  Skipping the library replacement checks (no shipped libgmp found)." >&2
    echo "License compliance check finished with $failures failure(s)."
    exit 10
fi
# Run a Clifford+T rotation synthesis, which exercises GMP and MPFR.
# C++ installations ship cudaq-opt; wheels expose cudaq.synth.gridsynth.
if ! $wheel_mode && [ -x "$install_root/bin/cudaq-opt" ]; then
    smoke_dir=$(mktemp -d)
    cat > "$smoke_dir/rz.qke" << 'EOF'
func.func @rz() {
  %q = quake.alloca !quake.ref
  %a = arith.constant 0.7853981633974483 : f64
  quake.rz (%a) %q : (f64, !quake.ref) -> ()
  return
}
EOF
    synthesis_smoke_test() {
        "$install_root/bin/cudaq-opt" --clifford-t-synthesis='epsilon=1e-3' \
            "$smoke_dir/rz.qke" 2>/dev/null | grep -q "__cliffordt_rz_"
    }
else
    synthesis_smoke_test() {
        python3 -c "import cudaq; gates = cudaq.synth.gridsynth(0.7853981633974483, 1e-3); assert 'T' in gates, gates" 2>/dev/null
    }
fi

# Back up the shipped libgmp files and guarantee they are restored.
backup_dir=$(mktemp -d)
restore_gmp() {
    for f in $gmp_libs; do
        cp -f "$backup_dir/$(basename "$f")" "$f" 2>/dev/null
    done
}
trap restore_gmp EXIT
for f in $gmp_libs; do
    cp -f "$f" "$backup_dir/$(basename "$f")"
done

if synthesis_smoke_test; then
    report 0 "Clifford+T rotation synthesis works with the shipped libraries"

    # Replacing libgmp with an invalid file must break synthesis: this proves
    # the shipped file is what is loaded at runtime. A hidden statically
    # linked copy would keep working here.
    for f in $gmp_libs; do
        echo "not a shared library" > "$f"
    done
    ! synthesis_smoke_test
    report $? "synthesis stops working when libgmp is replaced with an invalid file (the shipped library is really used)"

    # Substituting a valid library must make synthesis work again. The
    # pristine copy stands in for a user-provided compatible build; this
    # exercises the replacement mechanism the LGPL requires us to support.
    restore_gmp
    synthesis_smoke_test
    report $? "synthesis works again after substituting the libgmp files"

    # Informational only: if the system provides its own libgmp with the same
    # soname, try that as a genuinely different replacement build.
    if ! $is_darwin && command -v ldconfig >/dev/null 2>&1; then
        soname_file=$(basename "$(echo "$gmp_libs" | grep -E '\.so\.[0-9]+$' | head -1)")
        system_gmp=$(ldconfig -p 2>/dev/null | grep -F "$soname_file " | grep -oE '/[^ ]+$' | grep -v "^$lib_dir" | head -1)
        if [ -n "$soname_file" ] && [ -n "$system_gmp" ]; then
            cp -f "$system_gmp" "$lib_dir/$soname_file"
            if synthesis_smoke_test; then
                echo "  [OK] synthesis also works with the system-provided $system_gmp"
            else
                echo "  [INFO] synthesis did not work with the system-provided $system_gmp (not counted as a failure)"
            fi
            restore_gmp
        fi
    fi
else
    report 1 "Clifford+T rotation synthesis works with the shipped libraries"
    echo "  Skipping the library replacement checks." >&2
fi

echo "License compliance check finished with $failures failure(s)."
if [ "$failures" -eq 0 ]; then exit 0; else exit 10; fi
