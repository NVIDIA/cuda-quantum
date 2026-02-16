#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Run CUDA-Q test suite (ctest + llvm-lit)
# Used by both Linux and macOS CI, and for local development.
#
# Usage: bash scripts/run_tests.sh [-v] [-B build_dir]
#
# Note: GPU tests will fail gracefully on macOS (no CUDA available)

this_file_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$this_file_dir/set_env_defaults.sh"

build_dir="build"
verbose=""

while getopts ":vB:" opt; do
  case $opt in
    v) verbose="-v" ;;
    B) build_dir="$OPTARG" ;;
  esac
done

# Verify required environment variables
if [ -z "${LLVM_INSTALL_PREFIX:-}" ]; then
  echo "::error::LLVM_INSTALL_PREFIX is not set"
  exit 1
fi

status_sum=0

# Set PYTHONPATH to find the built cudaq module
export PYTHONPATH="$build_dir/python:${PYTHONPATH:-}"

# Determine number of parallel jobs (use all available CPUs)
if [ "$(uname)" = "Darwin" ]; then
  num_jobs=$(sysctl -n hw.ncpu)
else
  num_jobs=$(nproc)
fi
echo "Running tests with $num_jobs parallel jobs"

# 1. CTest
echo "=== Running ctest ==="
ctest --output-on-failure --test-dir "$build_dir" \
  -E "ctest-nvqpp|ctest-targettests"
ctest_status=$?
if [ $ctest_status -ne 0 ]; then
  echo "::error::ctest failed with status $ctest_status"
  status_sum=$((status_sum + 1))
fi

# 2. Main lit tests
echo "=== Running llvm-lit (build/test) ==="
"$LLVM_INSTALL_PREFIX/bin/llvm-lit" $verbose --time-tests -j "$num_jobs" \
  --param nvqpp_site_config="$build_dir/test/lit.site.cfg.py" \
  "$build_dir/test"
lit_status=$?
if [ $lit_status -ne 0 ]; then
  echo "::error::llvm-lit (build/test) failed with status $lit_status"
  status_sum=$((status_sum + 1))
fi

# 3. Target tests
echo "=== Running llvm-lit (build/targettests) ==="
"$LLVM_INSTALL_PREFIX/bin/llvm-lit" $verbose --time-tests -j "$num_jobs" \
  --param nvqpp_site_config="$build_dir/targettests/lit.site.cfg.py" \
  "$build_dir/targettests"
targ_status=$?
if [ $targ_status -ne 0 ]; then
  echo "::error::llvm-lit (targettests) failed with status $targ_status"
  status_sum=$((status_sum + 1))
fi

# 4. Python MLIR tests
echo "=== Running llvm-lit (python/tests/mlir) ==="
"$LLVM_INSTALL_PREFIX/bin/llvm-lit" $verbose --time-tests -j "$num_jobs" \
  --param nvqpp_site_config="$build_dir/python/tests/mlir/lit.site.cfg.py" \
  "$build_dir/python/tests/mlir"
pymlir_status=$?
if [ $pymlir_status -ne 0 ]; then
  echo "::error::llvm-lit (python/tests/mlir) failed with status $pymlir_status"
  status_sum=$((status_sum + 1))
fi

# 5. Python interop tests
echo "=== Running pytest (interop tests) ==="
python3 -m pytest $verbose --durations=0 "$build_dir/python/tests/interop/"
pytest_status=$?
if [ $pytest_status -ne 0 ]; then
  echo "::error::pytest (interop tests) failed with status $pytest_status"
  status_sum=$((status_sum + 1))
fi

exit $((status_sum > 0 ? 1 : 0))
