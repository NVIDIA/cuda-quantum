#!/bin/bash

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Validate an installed CUDA-Q SDK prefix as a Melior consumer sees it.
#
# Checks, against the shared private prefix P:
#  1. required tool presence (llvm-config, mlir-tblgen, FileCheck, not, count),
#  2. stock llvm-config coherence (version, shared/static libnames, paths),
#  3. the upstream provider audit (scripts/check_mlir_providers.sh): exact
#     library names/SONAMEs, dependency direction, no duplicate strong MLIR
#     implementations, public CAPI export closure,
#  4. CUDA-Q's own shared libraries resolve the providers and keep their C API
#     additions in libcudaqMLIRCAPI (never in libMLIR-C),
#  5. CMake consumer discovery: a C consumer against the upstream MLIR package,
#     and find_package(CUDAQ) when the devel configuration is installed,
#  6. optionally (--wheelhouse): wheel RECORD path ownership between the
#     matching cudaq-core/cudaq-devel wheels and their pinned versions,
#  7. a standalone unmodified Melior smoke crate (scripts/melior_sdk_smoke),
#     built against this prefix with its pinned toolchain and Cargo.lock.
#
# Usage: validate_melior_sdk.sh --prefix <sdk-prefix> [--wheelhouse <dir>]

set -euo pipefail
export LC_ALL=C
# The point of this script is validating the installed SDK standalone: never
# inherit producer paths from the ambient environment.
# PYTHONPATH too: producer environments (the dev container bakes
# /usr/local/cudaq) would otherwise shadow the wheel-installed packages.
unset LD_LIBRARY_PATH LIBRARY_PATH CPATH CPLUS_INCLUDE_PATH PYTHONPATH LD_PRELOAD || true

prefix=""
wheelhouse=""
while [ $# -gt 0 ]; do
  case "$1" in
    --prefix) prefix="$2"; shift 2 ;;
    --wheelhouse) wheelhouse="$2"; shift 2 ;;
    *) echo "Usage: $0 --prefix <sdk-prefix> [--wheelhouse <dir>]" >&2; exit 2 ;;
  esac
done
if [ -z "$prefix" ] || [ ! -d "$prefix" ]; then
    echo "Error: --prefix <sdk-prefix> required (got '$prefix')" >&2
    exit 2
fi
prefix="$(cd "$prefix" && pwd)"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

failures=0
fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
note() { echo "validate_melior_sdk: $*"; }

sover_for() { "$1/bin/llvm-config" --version | sed 's/\.[0-9]*$//'; }
soname_of() { readelf -d "$1" | sed -n 's/.*SONAME.*\[\(.*\)\]/\1/p'; }
needed_of() { readelf -d "$1" | sed -n 's/.*NEEDED.*Shared library: \[\(.*\)\]/\1/p'; }

# ---- 1./2. Tools and stock llvm-config ---------------------------------- #
llvm_config="$prefix/bin/llvm-config"
for tool in llvm-config mlir-tblgen FileCheck not count; do
  [ -x "$prefix/bin/$tool" ] || fail "missing tool bin/$tool"
done
[ -x "$llvm_config" ] || fail "llvm-config not executable"
if [ -x "$llvm_config" ]; then
  version="$("$llvm_config" --version)"
  sover="${version%.*}"
  note "llvm-config --version: $version"
  [ "$version" = "22.1.4" ] || fail "llvm-config --version is $version; the SDK contract pins 22.1.4"
  # Feature manifest: the pinned producer flags must be reflected in the
  # installed MLIR configuration.
  grep -q 'set(MLIR_ENABLE_EXECUTION_ENGINE "ON")' \
    "$prefix/lib/cmake/mlir/MLIRConfig.cmake" \
    || fail "installed MLIRConfig does not enable the execution engine"
  [ "$("$llvm_config" --includedir)" = "$prefix/include" ] \
    || fail "llvm-config --includedir != $prefix/include"
  [ "$("$llvm_config" --libdir)" = "$prefix/lib" ] \
    || fail "llvm-config --libdir != $prefix/lib"
  [ "$("$llvm_config" --link-shared --libnames)" = "libLLVM-${sover%%.*}.so" ] \
    || fail "llvm-config --link-shared --libnames is not libLLVM-${sover%%.*}.so"
  "$llvm_config" --link-static --libnames | tr ' ' '\n' | grep -q '^libLLVMSupport\.a$' \
    || fail "llvm-config --link-static lost static archives"
fi

# ---- 3. Upstream provider audit ------------------------------------------ #
if ! bash "$script_dir/check_mlir_providers.sh" "$prefix"; then
  fail "check_mlir_providers.sh failed"
fi

# ---- 4. CUDA-Q libraries resolve providers; CAPI additions stay separate - #
sover="${sover:-$(sover_for "$llvm_config")}"
for lib in libcudaqMLIR.so libcudaqMLIRCAPI.so; do
  [ -f "$prefix/lib/$lib" ] || fail "missing $lib"
done
if [ -f "$prefix/lib/libcudaqMLIR.so" ]; then
  needed_of "$prefix/lib/libcudaqMLIR.so" | grep -qx "libMLIR.so.$sover" \
    || fail "libcudaqMLIR does not depend on libMLIR.so.$sover"
fi
if [ -f "$prefix/lib/libcudaqMLIRCAPI.so" ]; then
  needed_of "$prefix/lib/libcudaqMLIRCAPI.so" | grep -qx "libMLIR-C.so.$sover" \
    || fail "libcudaqMLIRCAPI does not depend on libMLIR-C.so.$sover"
  nm -D --defined-only "$prefix/lib/libcudaqMLIRCAPI.so" \
    | grep -q ' mlirGetDialectHandle__quake__$' \
    || fail "libcudaqMLIRCAPI lost mlirGetDialectHandle__quake__"
  if nm -D --defined-only "$(readlink -f "$prefix/lib/libMLIR-C.so.$sover")" \
    | grep -q 'quake'; then
    fail "libMLIR-C unexpectedly exports quake symbols"
  fi
fi

# ---- 5. CMake consumer discovery ------------------------------------------ #
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT

# 5a. Upstream MLIR package: a C consumer of the C API shared provider.
mkdir -p "$scratch/c-consumer"
cat > "$scratch/c-consumer/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.26)
project(sdk_c_consumer C)
find_package(MLIR REQUIRED CONFIG)
add_executable(c_consumer main.c)
target_include_directories(c_consumer SYSTEM PRIVATE ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
target_link_libraries(c_consumer PRIVATE MLIR-C)
EOF
cat > "$scratch/c-consumer/main.c" <<'EOF'
#include <mlir-c/IR.h>
#include <mlir-c/RegisterEverything.h>
int main(void) {
  MlirContext context = mlirContextCreate();
  MlirDialectRegistry registry = mlirDialectRegistryCreate();
  mlirRegisterAllDialects(registry);
  mlirContextAppendDialectRegistry(context, registry);
  mlirDialectRegistryDestroy(registry);
  mlirContextDestroy(context);
  return 0;
}
EOF
if ! cmake -S "$scratch/c-consumer" -B "$scratch/c-consumer/build" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DMLIR_DIR="$prefix/lib/cmake/mlir" -DLLVM_DIR="$prefix/lib/cmake/llvm" \
      > "$scratch/c-consumer/configure.log" 2>&1 \
   || ! cmake --build "$scratch/c-consumer/build" >> "$scratch/c-consumer/configure.log" 2>&1; then
  cat "$scratch/c-consumer/configure.log" >&2
  fail "upstream MLIR C consumer build failed"
elif ! "$scratch/c-consumer/build/c_consumer"; then
  fail "upstream MLIR C consumer did not run"
fi

# 5b. CUDA-Q package discovery (when the devel configuration is installed).
site_packages="$(cd "$prefix/../../.." 2>/dev/null && pwd || true)"
if [ -n "$site_packages" ] && [ -f "$site_packages/lib/cmake/cudaq/CUDAQConfig.cmake" ]; then
  mkdir -p "$scratch/cudaq-consumer"
  cat > "$scratch/cudaq-consumer/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.26)
project(sdk_cudaq_consumer NONE)
set(CUDAQ_ENABLE_LANGUAGE OFF)
find_package(CUDAQ REQUIRED)
foreach(target IN ITEMS cudaq::cudaqMLIR cudaq::cudaqMLIRCAPI
    cudaq::UpstreamMLIR cudaq::UpstreamMLIRCAPI
    cudaq::UpstreamMLIRExecutionEngine cudaq::UpstreamLLVM)
  if(NOT TARGET ${target})
    message(FATAL_ERROR "missing expected target ${target}")
  endif()
endforeach()
EOF
  if ! cmake -S "$scratch/cudaq-consumer" -B "$scratch/cudaq-consumer/build" \
        -DCMAKE_PREFIX_PATH="$site_packages" \
        > "$scratch/cudaq-consumer/configure.log" 2>&1; then
    cat "$scratch/cudaq-consumer/configure.log" >&2
    fail "find_package(CUDAQ) consumer discovery failed"
  fi
else
  note "no devel CUDAQConfig.cmake under $site_packages; skipping 5b"
fi

# 4c. Producer signals (sanity only): the shipped libcudaqMLIR's .comment
# records the pinned LLVM commit and the GCC family. These strings are NOT
# an identity proof: they pin the linker commit and the presence of any GCC
# record — GCC CRT entries appear in Clang-built DSOs too (the existing
# Logical compiler audit documents this distinction). The artifact/ABI
# identity binding is the wheel-RECORD/prefix byte-identity chain in §6
# (requires --wheelhouse).
if [ "$(uname)" != "Darwin" ] && [ -f "$prefix/lib/libcudaqMLIR.so" ]; then
  comment="$(readelf -p .comment "$prefix/lib/libcudaqMLIR.so" 2>/dev/null || true)"
  case "$comment" in
  *def143a6c624*|*def143a6c624dc9b991ebfdfec5c36a7084171eb*)
    : ;;
  *)
    fail "libcudaqMLIR .comment lacks the pinned LLVM producer commit (def143a6c624)"
    ;;
  esac
  case "$comment" in
  *"GCC:"*) : ;;
  *) fail "libcudaqMLIR .comment lacks a GCC producer-family record" ;;
  esac
fi

# 5c. Contract-path consumer (the rulings in notes_79022983 + 79497982):
# bounded Option B — CUDA-Q development config discovery at the
# same-environment flat site-packages path D; the LLVM/MLIR development
# packages resolve at P via the CUDAQ config's guarded seeding. A search
# hint is not a constraint: the consumer asserts the EFFECTIVE roots —
# CUDAQ_DIR and the AddCUDAQ helper home are canonical-exact D, LLVM_DIR /
# MLIR_DIR are canonical-exact P, and the imported tools/providers resolve
# inside canonical P (a textual-prefix sibling or a symlink escaping P is
# rejected). The umbrella must yield a working cudaq::MLIR, and a foreign
# (forwarding) CUDAQ config must fail the configure.
canon_prefix="$(realpath "$prefix")"
canon_site="$(realpath "$site_packages")"
expected_cudaq_dir="$canon_site/lib/cmake/cudaq"
expected_llvm_dir="$canon_prefix/lib/cmake/llvm"
expected_mlir_dir="$canon_prefix/lib/cmake/mlir"
expected_args=(
  "-DEXPECTED_CUDAQ_DIR=$expected_cudaq_dir"
  "-DEXPECTED_LLVM_DIR=$expected_llvm_dir"
  "-DEXPECTED_MLIR_DIR=$expected_mlir_dir"
  "-DEXPECTED_PREFIX=$canon_prefix" "-DEXPECTED_SITE=$canon_site")
mkdir -p "$scratch/contract-consumer"
cat > "$scratch/contract-consumer/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.26)
project(sdk_contract_consumer CXX)
set(CUDAQ_ENABLE_LANGUAGE OFF)
# CUDA-Q first: the config seeds MLIR_DIR/LLVM_DIR to P's packages.
find_package(CUDAQ REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}" "${MLIR_CMAKE_DIR}" "${CUDAQ_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(AddCUDAQ RESULT_VARIABLE _cudaq_addcudaq_path)
# Exact-root assertions: the effective configuration is the contract one,
# not a foreign or sibling root reachable through the search hint — and not
# an in-prefix symlink that resolves outside it. Canonicalize both sides,
# require equality with the expected roots AND containment inside them, and
# check the helper file actually included, not only its directory.
file(REAL_PATH "${CUDAQ_DIR}" _eff_cudaq_dir)
file(REAL_PATH "${CUDAQ_CMAKE_DIR}" _eff_cudaq_cmake_dir)
file(REAL_PATH "${LLVM_DIR}" _eff_llvm_dir)
file(REAL_PATH "${MLIR_DIR}" _eff_mlir_dir)
file(REAL_PATH "${EXPECTED_CUDAQ_DIR}" _exp_cudaq_dir)
file(REAL_PATH "${EXPECTED_LLVM_DIR}" _exp_llvm_dir)
file(REAL_PATH "${EXPECTED_MLIR_DIR}" _exp_mlir_dir)
file(REAL_PATH "${EXPECTED_PREFIX}" _exp_prefix)
file(REAL_PATH "${EXPECTED_SITE}" _exp_site)
if(NOT _eff_cudaq_dir STREQUAL _exp_cudaq_dir)
  message(FATAL_ERROR "contract-root-mismatch: effective CUDAQ_DIR (${CUDAQ_DIR}) is not the contract root D (${EXPECTED_CUDAQ_DIR})")
endif()
# The effective config FILES, not only their directories: a symlinked
# config file resolving outside its intended domain fails even when the
# directory variables compare equal. (Checked before the helper dir, so a
# file-level escape cannot hide behind it.)
file(REAL_PATH "${CUDAQ_DIR}/CUDAQConfig.cmake" _eff_cudaq_config_file)
if(NOT _eff_cudaq_config_file STREQUAL "${_exp_cudaq_dir}/CUDAQConfig.cmake")
  message(FATAL_ERROR "contract-root-mismatch: effective CUDAQConfig (${_eff_cudaq_config_file}) is not D's file (${_exp_cudaq_dir}/CUDAQConfig.cmake)")
endif()
if(NOT _eff_cudaq_cmake_dir STREQUAL _exp_cudaq_dir)
  message(FATAL_ERROR "contract-root-mismatch: AddCUDAQ home CUDAQ_CMAKE_DIR (${CUDAQ_CMAKE_DIR}) is not the contract root D (${EXPECTED_CUDAQ_DIR})")
endif()
if(NOT _eff_llvm_dir STREQUAL _exp_llvm_dir)
  message(FATAL_ERROR "contract-root-mismatch: effective LLVM_DIR (${LLVM_DIR}) is not P's LLVM package (${EXPECTED_LLVM_DIR})")
endif()
if(NOT _eff_mlir_dir STREQUAL _exp_mlir_dir)
  message(FATAL_ERROR "contract-root-mismatch: effective MLIR_DIR (${MLIR_DIR}) is not P's MLIR package (${EXPECTED_MLIR_DIR})")
endif()
file(REAL_PATH "${LLVM_DIR}/LLVMConfig.cmake" _eff_llvm_config_file)
file(REAL_PATH "${MLIR_DIR}/MLIRConfig.cmake" _eff_mlir_config_file)
if(NOT _eff_llvm_config_file STREQUAL "${_exp_llvm_dir}/LLVMConfig.cmake")
  message(FATAL_ERROR "contract-root-mismatch: effective LLVMConfig (${_eff_llvm_config_file}) is not P's file (${_exp_llvm_dir}/LLVMConfig.cmake)")
endif()
if(NOT _eff_mlir_config_file STREQUAL "${_exp_mlir_dir}/MLIRConfig.cmake")
  message(FATAL_ERROR "contract-root-mismatch: effective MLIRConfig (${_eff_mlir_config_file}) is not P's file (${_exp_mlir_dir}/MLIRConfig.cmake)")
endif()
# Containment is literal and component-bounded, never regex: a root
# containing '+' must not reject itself, and '.' must not match foreign
# characters.
string(FIND "${_eff_cudaq_dir}/" "${_exp_site}/" _pos)
if(NOT _pos EQUAL 0)
  message(FATAL_ERROR "contract-escape-mismatch: CUDA-Q config root (${_eff_cudaq_dir}) escapes the intended site (${_exp_site})")
endif()
string(FIND "${_eff_llvm_dir}/" "${_exp_prefix}/" _pos)
if(NOT _pos EQUAL 0)
  message(FATAL_ERROR "contract-escape-mismatch: effective LLVM config root (${_eff_llvm_dir}) escapes the intended prefix (${_exp_prefix})")
endif()
string(FIND "${_eff_mlir_dir}/" "${_exp_prefix}/" _pos)
if(NOT _pos EQUAL 0)
  message(FATAL_ERROR "contract-escape-mismatch: effective MLIR config root (${_eff_mlir_dir}) escapes the intended prefix (${_exp_prefix})")
endif()
file(REAL_PATH "${_cudaq_addcudaq_path}" _eff_addcudaq_path)
if(NOT _eff_addcudaq_path STREQUAL "${_exp_cudaq_dir}/AddCUDAQ.cmake")
  message(FATAL_ERROR "contract-root-mismatch: included AddCUDAQ (${_cudaq_addcudaq_path}) is not D's helper (${_exp_cudaq_dir}/AddCUDAQ.cmake)")
endif()
foreach(target IN ITEMS cudaq::MLIR cudaq::cudaqMLIR cudaq::cudaqMLIRCAPI
    cudaq::UpstreamMLIR cudaq::UpstreamMLIRCAPI
    cudaq::UpstreamMLIRExecutionEngine cudaq::UpstreamLLVM mlir-tblgen)
  if(NOT TARGET ${target})
    message(FATAL_ERROR "missing expected target ${target}")
  endif()
endforeach()
file(GENERATE OUTPUT tblgen-path.txt CONTENT "$<TARGET_FILE:mlir-tblgen>")
# Record every provider location so the harness can assert each resolves
# inside the canonical installed prefix, never a foreign or sibling root.
file(GENERATE OUTPUT provider-paths.txt CONTENT
  "$<TARGET_FILE:cudaq::UpstreamMLIR>\n$<TARGET_FILE:cudaq::UpstreamMLIRCAPI>\n$<TARGET_FILE:cudaq::UpstreamMLIRExecutionEngine>\n$<TARGET_FILE:cudaq::UpstreamLLVM>\n")
add_executable(contract_consumer main.cpp)
target_include_directories(contract_consumer SYSTEM PRIVATE ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
# Umbrella-only acceptance (SDK97-5): cudaq::MLIR must deliver the whole
# provider set transitively — libLLVM included via cudaqMLIR's PUBLIC
# cudaq::UpstreamLLVM edge — with no consumer-side LLVM link workaround.
target_link_libraries(contract_consumer PRIVATE cudaq::MLIR)
EOF
cat > "$scratch/contract-consumer/main.cpp" <<'EOF'
#include <mlir/IR/MLIRContext.h>
int main() { mlir::MLIRContext ctx; (void)ctx; return 0; }
EOF
if ! cmake -S "$scratch/contract-consumer" -B "$scratch/contract-consumer/build" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="$site_packages" \
      "${expected_args[@]}" \
      > "$scratch/contract-consumer/configure.log" 2>&1 \
   || ! cmake --build "$scratch/contract-consumer/build" >> "$scratch/contract-consumer/configure.log" 2>&1; then
  cat "$scratch/contract-consumer/configure.log" >&2
  fail "contract-path consumer (\$prefix discovery) failed"
elif ! "$scratch/contract-consumer/build/contract_consumer"; then
  fail "contract-path consumer did not run"
else
  # Effective-tool and provider assertions (canonical; a runnable tool is
  # not evidence it lives at P, and a symlinked tool that escapes P fails
  # containment even when both sides resolve alike).
  if ! "$(cat "$scratch/contract-consumer/build/tblgen-path.txt")" --version | grep -q "LLVM"; then
    fail "contract-path consumer: imported mlir-tblgen target does not resolve to a runnable tool"
  fi
  tblgen_path="$(realpath "$(cat "$scratch/contract-consumer/build/tblgen-path.txt")")"
  if [ "$tblgen_path" != "$(realpath "$canon_prefix/bin/mlir-tblgen")" ]; then
    fail "mlir-tblgen resolves to $tblgen_path, not P's $canon_prefix/bin/mlir-tblgen"
  fi
  case "$tblgen_path" in
  "$canon_prefix/bin"/*) ;;
  *) fail "mlir-tblgen escapes the canonical prefix: $tblgen_path" ;;
  esac
  while IFS= read -r provider_path; do
    [ -n "$provider_path" ] || continue
    canon_provider="$(realpath "$provider_path")"
    case "$canon_provider" in
    "$canon_prefix/lib"/*) ;;
    *) fail "provider resolves outside the canonical prefix: $canon_provider" ;;
    esac
  done < "$scratch/contract-consumer/build/provider-paths.txt"
fi

# Foreign-config negative: a foreign CUDA-Q config that forwards to the
# legitimate installation creates P-local providers and passes naive
# provider checks — the exact-D assertion must reject the root anyway.
foreign="$scratch/foreign-root"
mkdir -p "$foreign/lib/cmake/cudaq"
cat > "$foreign/lib/cmake/cudaq/CUDAQConfig.cmake" <<EOF
# Foreign forwarding config: discovery must reject any root but D.
include("$expected_cudaq_dir/CUDAQConfig.cmake")
EOF
if cmake -S "$scratch/contract-consumer" -B "$scratch/contract-consumer/build-foreign" \
    -DCMAKE_PREFIX_PATH="$foreign;$site_packages" \
    "${expected_args[@]}" \
    > "$scratch/contract-consumer/foreign-configure.log" 2>&1; then
  fail "foreign CUDAQ_DIR override was not rejected (exact-D rule broken)"
else
  # CMake wraps long messages; match the short stable tag.
  if grep -q "contract-root-mismatch" \
      "$scratch/contract-consumer/foreign-configure.log"; then
    note "foreign-config override rejected at configure, as required"
  else
    cat "$scratch/contract-consumer/foreign-configure.log" >&2
    fail "foreign override rejected for the wrong reason"
  fi
fi

# Symlink-escape negative: a prefix mirroring the real content through
# symlinks is a foreign root — equality of resolved paths must not
# suffice; containment inside the intended root is required.
escape="$scratch/escape-prefix"
mkdir -p "$escape"
ln -s "$canon_site/lib" "$escape/lib"
ln -s "$canon_site/cudaq_core" "$escape/cudaq_core"
escape_args=(
  "-DEXPECTED_CUDAQ_DIR=$escape/lib/cmake/cudaq"
  "-DEXPECTED_LLVM_DIR=$escape/cudaq_core/cudaq/core/lib/cmake/llvm"
  "-DEXPECTED_MLIR_DIR=$escape/cudaq_core/cudaq/core/lib/cmake/mlir"
  "-DEXPECTED_PREFIX=$escape/cudaq_core/cudaq/core"
  "-DEXPECTED_SITE=$escape")
if cmake -S "$scratch/contract-consumer" -B "$scratch/contract-consumer/build-escape" \
    -DCMAKE_PREFIX_PATH="$escape" \
    "${escape_args[@]}" \
    > "$scratch/contract-consumer/escape-configure.log" 2>&1; then
  fail "symlink-escape prefix was not rejected (containment rule broken)"
else
  if grep -q "contract-escape-mismatch" \
      "$scratch/contract-consumer/escape-configure.log"; then
    note "symlink-escape prefix rejected at configure, as required"
  else
    cat "$scratch/contract-consumer/escape-configure.log" >&2
    fail "symlink-escape rejected for the wrong reason"
  fi
fi

# Config-file symlink negative: a REAL config directory whose
# CUDAQConfig.cmake file alone is a symlink to the real D's file — the
# directory compares equal, but the effective config FILE resolves outside
# it and must be rejected.
filelink="$scratch/filelink-site"
mkdir -p "$filelink/lib/cmake/cudaq"
for entry in "$canon_site"/lib/cmake/*; do
    name="$(basename "$entry")"
    [ "$name" = cudaq ] || ln -s "$entry" "$filelink/lib/cmake/$name"
done
ln -s "$canon_site/cudaq_core" "$filelink/cudaq_core"
for f in "$canon_site"/lib/cmake/cudaq/*; do
    ln -s "$f" "$filelink/lib/cmake/cudaq/"
done
filelink_args=(
  "-DEXPECTED_CUDAQ_DIR=$filelink/lib/cmake/cudaq"
  "-DEXPECTED_LLVM_DIR=$expected_llvm_dir"
  "-DEXPECTED_MLIR_DIR=$expected_mlir_dir"
  "-DEXPECTED_PREFIX=$canon_prefix"
  "-DEXPECTED_SITE=$filelink")
if cmake -S "$scratch/contract-consumer" -B "$scratch/contract-consumer/build-filelink" \
    -DCMAKE_PREFIX_PATH="$filelink;$site_packages" \
    "${filelink_args[@]}" \
    > "$scratch/contract-consumer/filelink-configure.log" 2>&1; then
  fail "config-file symlink was not rejected (file-domain rule broken)"
else
  if grep -q "contract-root-mismatch" \
      "$scratch/contract-consumer/filelink-configure.log"; then
    note "config-file symlink rejected at configure, as required"
  else
    cat "$scratch/contract-consumer/filelink-configure.log" >&2
    fail "config-file symlink rejected for the wrong reason"
  fi
fi

# ---- 6. Wheel RECORD ownership (optional) --------------------------------- #
if [ -n "$wheelhouse" ]; then
  python3 - "$wheelhouse" <<'PY' || fail "wheel RECORD ownership check failed"
import glob, sys, zipfile
house = sys.argv[1]
def wheels(pattern):
    found = sorted(glob.glob(f"{house}/{pattern}"))
    if len(found) > 1:
        manylinux = [f for f in found if "manylinux" in f]
        if manylinux:
            found = manylinux
    if len(found) != 1:
        print(f"FAIL: expected exactly one {pattern} in {house}, got {found}")
        sys.exit(1)
    return found[0]
core, devel = wheels("cudaq_core-*.whl"), wheels("cudaq_devel-*.whl")
def metadata(wheel):
    with zipfile.ZipFile(wheel) as z:
        meta = [n for n in z.namelist() if n.endswith(".dist-info/METADATA")][0]
        text = z.read(meta).decode()
        name = next(l.split(": ",1)[1] for l in text.splitlines() if l.startswith("Name: "))
        version = next(l.split(": ",1)[1] for l in text.splitlines() if l.startswith("Version: "))
        requires = [l for l in text.splitlines() if l.startswith("Requires-Dist: ")]
        return name, version, requires
core_name, core_v, _ = metadata(core)
dev_name, dev_v, dev_req = metadata(devel)
if core_v != dev_v:
    print(f"FAIL: wheel versions differ: {core_name} {core_v} vs {dev_name} {dev_v}")
    sys.exit(1)
dev_pins = [r.split(": ", 1)[1].strip() for r in dev_req]
# Luca-approved metadata fallback (onsen #6 note_79265934): devel keeps the
# full cudaq==V closure; the delegated cudaq-core==V pin is a staged change.
if not any(r == f"cudaq=={core_v}" for r in dev_pins):
    print(f"FAIL: {dev_name} does not pin cudaq=={core_v} (fallback metadata): {dev_pins[:3]}")
    sys.exit(1)

def record_paths(wheel):
    with zipfile.ZipFile(wheel) as z:
        record = [n for n in z.namelist() if n.endswith(".dist-info/RECORD")][0]
        return {line.split(",")[0] for line in
                z.read(record).decode().splitlines() if line}
core_paths, devel_paths = record_paths(core), record_paths(devel)
overlap = {p for p in core_paths & devel_paths
           if not p.startswith(("cudaq_core-", "cudaq_devel-"))
           and ".dist-info/" not in p}
if overlap:
    print(f"FAIL: paths owned by both wheels: {sorted(overlap)[:10]}")
    sys.exit(1)
need_core = "cudaq_core/cudaq/core/lib/libMLIR.so"
need_devel = "cudaq_core/cudaq/core/bin/llvm-config"
for paths, need, name in ((core_paths, need_core, "core"),
                          (devel_paths, need_devel, "devel")):
    if not any(p == need or p.startswith(need + ".") for p in paths):
        print(f"FAIL: {name} wheel missing {need}")
        sys.exit(1)
if any("lib/libMLIR.so" in p or "lib/libLLVM.so" in p for p in devel_paths):
    print("FAIL: devel wheel owns provider libraries")
    sys.exit(1)
print("OK: wheel RECORD ownership is disjoint; providers with core, tools with devel")
PY

  # 6b. Artifact identity binding (the SDK99-4 contract): for BOTH wheels
  # (core and devel), every RECORD-hashed member's installed bytes must
  # equal the RECORD-verified wheel bytes, unhashed non-signature members
  # fail closed, and the installed dist-info METADATA must equal the
  # wheel's. .comment strings (see 4c) cannot bind implementation content;
  # byte identity between the installed prefix and the recorded wheel
  # artifacts can.
  shopt -s nullglob
  core_candidates=("$wheelhouse"/cudaq_core-*.whl)
  devel_candidates=("$wheelhouse"/cudaq_devel-*.whl)
  shopt -u nullglob
  pick_one() {   # pick_one <candidates...>: echo the single wheel (manylinux preferred)
    local arr=(${@+"$@"}) many=()
    if [ "${#arr[@]}" -gt 1 ]; then
      local w
      for w in "${arr[@]}"; do case "$w" in *manylinux*) many+=("$w");; esac; done
      [ "${#many[@]}" -gt 0 ] && arr=("${many[@]}")
    fi
    [ "${#arr[@]}" -eq 1 ] && echo "${arr[0]}"
  }
  core_wheel="$(pick_one ${core_candidates[@]+"${core_candidates[@]}"})"
  devel_wheel="$(pick_one ${devel_candidates[@]+"${devel_candidates[@]}"})"
  [ -n "$core_wheel" ] || fail "expected exactly one cudaq_core wheel in $wheelhouse"
  [ -n "$devel_wheel" ] || fail "expected exactly one cudaq_devel wheel in $wheelhouse"
  for w in "$core_wheel" "$devel_wheel"; do
    [ -z "$w" ] && continue
    if ! bash "$script_dir/check_wheel_install_identity.sh" "$w" "$site_packages"; then
      fail "wheel↔install identity binding failed for $(basename "$w")"
    fi
  done
fi

# ---- 7. Standalone unmodified Melior smoke crate --------------------------- #
if ! command -v cargo >/dev/null 2>&1; then
  fail "cargo not found; the pinned smoke crate requires a Rust toolchain"
else
  crate="$scratch/melior_sdk_smoke"
  cp -r "$script_dir/melior_sdk_smoke" "$crate"
  export CARGO_BUILD_JOBS=2
  export CUDAQ_SDK_PREFIX="$prefix"
  export MLIR_SYS_220_PREFIX="$prefix"
  export MLIR_SYS_LINK_SHARED=1
  export TABLEGEN_220_PREFIX="$prefix"
  export LIBCLANG_PATH="$prefix/lib"
  export PATH="$prefix/bin:$PATH"
  # Link-time discovery must not rely on LD_LIBRARY_PATH.
  export RUSTFLAGS="-C link-arg=-Wl,-rpath,$prefix/lib ${RUSTFLAGS:-}"
  note "fetching pinned crates (download time reported separately)"
  fetch_start=$SECONDS
  if ! (cd "$crate" && cargo fetch --locked) > "$scratch/cargo.log" 2>&1; then
    cat "$scratch/cargo.log" >&2
    fail "cargo fetch --locked failed"
  fi
  fetch_elapsed=$((SECONDS - fetch_start))
  # The cold build runs ONCE, inside the measurement wrapper, so both the
  # elapsed time and the peak RSS describe the genuine cold compile (a copy
  # of an already-built fixture would measure a warm no-op).
  read -r build_elapsed peak_kb < <(python3 - "$crate" <<'PYEOF'
import resource, subprocess, sys, time
start = time.monotonic()
subprocess.run(["cargo", "build", "--locked", "-j", "2", "--offline"],
               cwd=sys.argv[1], check=True)
print(round(time.monotonic() - start), resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
PYEOF
) || fail "cold build (measured) failed"
  # Cheap-build evidence: incremental rebuilds neither recompile nor relink.
  no_op_start=$SECONDS
  (cd "$crate" && cargo build --locked -j 2) >> "$scratch/cargo.log" 2>&1 || fail "incremental rebuild failed"
  no_op_elapsed=$((SECONDS - no_op_start))
  touch "$crate/src/main.rs"
  inc_start=$SECONDS
  (cd "$crate" && cargo build --locked -j 2) >> "$scratch/cargo.log" 2>&1 || fail "touched-source rebuild failed"
  inc_elapsed=$((SECONDS - inc_start))
  note "cargo fetch: ${fetch_elapsed}s, cold build: ${build_elapsed}s, no-op: ${no_op_elapsed}s, touched main.rs rebuild: ${inc_elapsed}s, peak RSS: $((peak_kb / 1024)) MiB (2 jobs)"
  if [ "$(uname)" != "Darwin" ]; then
    # Loaded-provider residency: the smoke binary must resolve every shared
    # provider from the selected installation, never from a preloaded or
    # foreign system copy. Fail-closed on ldd errors, unresolved providers,
    # and missing expected-provider evidence; canonical paths (a symlink
    # escaping the prefix is a foreign resolution).
    if ! bash "$script_dir/check_provider_residency.sh" \
        "$crate/target/debug/cudaq-sdk-melior-smoke" "$prefix"; then
      fail "smoke binary resolves providers outside the prefix"
    fi
  fi
  if ! "$crate/target/debug/cudaq-sdk-melior-smoke"; then
    fail "Melior smoke binary failed"
  fi
fi

if [ "$failures" -ne 0 ]; then
  echo "validate_melior_sdk: $failures check(s) failed for $prefix" >&2
  exit 1
fi
note "all checks passed for $prefix"
