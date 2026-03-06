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

# Thread budget to avoid OpenMP oversubscription.
# OpenMP-parallel tests (qpp, dm simulators) each use OMP_NUM_THREADS cores.
# For ctest, the PROCESSORS property handles scheduling for a given -j $num_jobs.
if [ -z "${OMP_NUM_THREADS:-}" ]; then
  if [ "$num_jobs" -le 4 ]; then
    omp_threads=1
    parallel_jobs=$num_jobs
  else
    omp_threads=2
    parallel_jobs=$((num_jobs / omp_threads))
  fi
  export OMP_NUM_THREADS=$omp_threads
else
  omp_threads=$OMP_NUM_THREADS
  parallel_jobs=$((num_jobs / omp_threads))
  if [ "$parallel_jobs" -lt 1 ]; then parallel_jobs=1; fi
fi
echo "Thread budget: $parallel_jobs parallel jobs x $omp_threads OMP threads (${num_jobs} cores)"

# Detect GPU availability for ctest label filtering
gpu_excludes=""
if [ "$(uname)" = "Darwin" ]; then
  gpu_excludes="--label-exclude gpu_required"
elif [ ! -x "$(command -v nvidia-smi)" ] || \
     [ -z "$(nvidia-smi | egrep -o "CUDA Version: ([0-9]{1,}\.)+[0-9]{1,}")" ]; then
  gpu_excludes="--label-exclude gpu_required"
fi

# 1a. CTest: CPU tests in parallel (PROCESSORS property handles scheduling)
echo "=== Running ctest ==="
ctest --output-on-failure --test-dir "$build_dir" -j "$num_jobs" \
  -E "ctest-nvqpp|ctest-targettests" $gpu_excludes
ctest_status=$?
if [ $ctest_status -ne 0 ]; then
  echo "::error::ctest failed with status $ctest_status"
  status_sum=$((status_sum + 1))
fi

# 1b. GPU tests: run serially to avoid GPU memory contention
if [ -z "$gpu_excludes" ]; then
  echo "=== Running GPU ctest (serial) ==="
  ctest --output-on-failure --test-dir "$build_dir" -j 1 \
    -E "ctest-nvqpp|ctest-targettests" \
    -L "gpu_required" --label-exclude "mgpus_required"
  gpu_ctest_status=$?
  if [ $gpu_ctest_status -ne 0 ]; then
    echo "::error::GPU ctest failed with status $gpu_ctest_status"
    status_sum=$((status_sum + 1))
  fi
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
"$LLVM_INSTALL_PREFIX/bin/llvm-lit" $verbose --time-tests -j "$parallel_jobs" \
  --param nvqpp_site_config="$build_dir/targettests/lit.site.cfg.py" \
  "$build_dir/targettests"
targ_status=$?
if [ $targ_status -ne 0 ]; then
  echo "::error::llvm-lit (targettests) failed with status $targ_status"
  status_sum=$((status_sum + 1))
fi

# 4. Python MLIR tests
echo "=== Running llvm-lit (python/tests/mlir) ==="
"$LLVM_INSTALL_PREFIX/bin/llvm-lit" $verbose --time-tests -j "$parallel_jobs" \
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
