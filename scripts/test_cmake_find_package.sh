#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Quick standalone test for the CMake find_package(CUDAQ) workflow.
#
# Usage:
#   source /path/to/cudaq/set_env.sh   # sets CUDA_QUANTUM_PATH
#   bash scripts/test_cmake_find_package.sh
#
# Or point directly at an install prefix:
#   bash scripts/test_cmake_find_package.sh /tmp/cudaq_test
#
# Exit codes:  0 = passed, 1 = failed/missing prereqs

set -euo pipefail

cudaq_path="${1:-${CUDA_QUANTUM_PATH:-}}"
if [ -z "$cudaq_path" ]; then
    echo "Usage: $0 [CUDA_QUANTUM_PATH]" >&2
    echo "  Or set CUDA_QUANTUM_PATH in your environment." >&2
    exit 1
fi

cudaq_cmake_dir="$cudaq_path/lib/cmake/cudaq"
if [ ! -f "$cudaq_cmake_dir/CUDAQConfig.cmake" ]; then
    echo "FAIL: CUDAQConfig.cmake not found in $cudaq_cmake_dir" >&2
    exit 1
fi

for tool in cmake make; do
    if ! command -v $tool &>/dev/null; then
        echo "FAIL: $tool not found" >&2
        exit 1
    fi
done

test_dir=$(mktemp -d)
trap 'rm -rf "$test_dir"' EXIT

cat > "$test_dir/CMakeLists.txt" << 'EOF'
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(cudaq_cmake_test LANGUAGES CXX)
find_package(CUDAQ REQUIRED)
add_executable(ghz_test ghz_test.cpp)
EOF

cat > "$test_dir/ghz_test.cpp" << 'EOF'
#include <cudaq.h>
struct ghz {
  auto operator()() __qpu__ {
    cudaq::qarray<3> q;
    h(q[0]);
    for (int i = 0; i < 2; i++)
      x<cudaq::ctrl>(q[i], q[i + 1]);
    mz(q);
  }
};
int main() {
  auto counts = cudaq::sample(ghz{});
  counts.dump();
  return 0;
}
EOF

echo "=== CMake configure ==="
if ! cmake -S "$test_dir" -B "$test_dir/build" \
        -DCUDAQ_DIR="$cudaq_cmake_dir" \
        -G "Unix Makefiles" 2>&1; then
    echo ""
    echo "FAIL: CMake configure failed." >&2
    exit 1
fi

echo ""
echo "=== CMake build ==="
if ! cmake --build "$test_dir/build" 2>&1; then
    echo ""
    echo "FAIL: Build failed." >&2
    exit 1
fi

echo ""
echo "=== Run ==="
if ! "$test_dir/build/ghz_test" 2>&1; then
    echo ""
    echo "FAIL: Execution failed." >&2
    exit 1
fi

echo ""
echo "PASS: find_package(CUDAQ) configure + build + run all succeeded."
