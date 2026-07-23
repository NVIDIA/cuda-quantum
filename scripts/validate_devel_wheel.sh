#!/bin/bash

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Validate a cudaq-devel wheel: contents check + optional out-of-tree smoke build.
#
# Usage:
#   bash scripts/validate_devel_wheel.sh -i dist -r dist -v 0.0.0
#
# Options:
#   -i <dir>: Directory containing cudaq_devel*.whl (required)
#   -r <dir>: Directory containing cuda-quantum-cu*.whl for co-install (optional)
#   -q: Skip the out-of-tree CMake smoke build (contents check only)

set -euo pipefail

devel_dir=""
runtime_dir=""
quick=false

while getopts ":i:r:q" opt; do
  case $opt in
  i) devel_dir="$OPTARG" ;;
  r) runtime_dir="$OPTARG" ;;
  q) quick=true ;;
  *) echo "Usage: $0 -i <devel_wheel_dir> [-r <runtime_wheel_dir>] [-q]" >&2; exit 1 ;;
  esac
done

if [ -z "$devel_dir" ]; then
  echo "Error: -i <devel_wheel_dir> is required" >&2
  exit 1
fi

devel_wheel=$(ls "$devel_dir"/cudaq_devel*.whl 2>/dev/null | head -1)
if [ -z "$devel_wheel" ]; then
  echo "Error: no cudaq_devel*.whl found in $devel_dir" >&2
  exit 1
fi
echo "Validating devel wheel: $devel_wheel"

# Contents: expect dev artifacts, not runtime-only paths stripped from runtime wheel.
for pattern in 'include/cudaq/' 'lib/cmake/cudaq/CUDAQConfig.cmake' 'bin/mlir-tblgen'; do
  if ! unzip -l "$devel_wheel" | grep -q "$pattern"; then
    echo "Error: expected path matching '$pattern' in devel wheel" >&2
    exit 1
  fi
  echo "  OK: found $pattern"
done

if unzip -l "$devel_wheel" | grep -q 'lib/libcudaq\.so'; then
  echo "Error: devel wheel should not ship libcudaq.so (provided by cudaq runtime)" >&2
  exit 1
fi
echo "  OK: libcudaq.so not bundled in devel wheel"

if $quick; then
  echo "Skipping smoke build (-q)"
  exit 0
fi

python="${PYTHON:-python3}"
venv_dir=$(mktemp -d)
smoke_dir=$(mktemp -d)
trap 'rm -rf "$venv_dir" "$smoke_dir"' EXIT

"$python" -m venv "$venv_dir"
# shellcheck source=/dev/null
source "$venv_dir/bin/activate"
pip install -q pip wheel

if [ -n "$runtime_dir" ]; then
  runtime_wheel=$(ls "$runtime_dir"/cuda_quantum_cu*.whl 2>/dev/null | head -1)
  if [ -n "$runtime_wheel" ]; then
    echo "Installing runtime wheel: $runtime_wheel"
    pip install -q "$runtime_wheel"
  fi
fi

echo "Installing devel wheel (no-deps; runtime co-installed above if provided)"
pip install -q --no-deps "$devel_wheel"

site_packages=$("$python" -c 'import site; print(site.getsitepackages()[0])')
cudaq_cmake_dir="$site_packages/lib/cmake/cudaq"
mlir_cmake_dir="$site_packages/lib/cmake/mlir"
llvm_cmake_dir="$site_packages/lib/cmake/llvm"

cat > "$smoke_dir/CMakeLists.txt" << 'EOF'
cmake_minimum_required(VERSION 3.30)
project(cudaq_devel_smoke LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAQ REQUIRED)
find_package(MLIR REQUIRED CONFIG)

cudaq_add_mlir_extension(cudaq_devel_smoke
  SOURCES noop_extension.cpp)
EOF

cat > "$smoke_dir/noop_extension.cpp" << 'EOF'
// Minimal shared library used to verify cudaq-devel + cudaq wheel overlay linking.

extern "C" void cudaq_devel_smoke_anchor() {}
EOF

echo "=== CMake configure ==="
if ! cmake -S "$smoke_dir" -B "$smoke_dir/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCUDAQ_DIR="$cudaq_cmake_dir" \
        -DMLIR_DIR="$mlir_cmake_dir" \
        -DLLVM_DIR="$llvm_cmake_dir" \
        -G "Unix Makefiles" 2>&1; then
  echo ""
  echo "FAIL: CMake configure failed." >&2
  exit 1
fi

echo ""
echo "=== CMake build ==="
if ! cmake --build "$smoke_dir/build" 2>&1; then
  echo ""
  echo "FAIL: Build failed." >&2
  exit 1
fi

echo ""
echo "PASS: devel wheel validation succeeded."
