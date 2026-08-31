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
# Usage: bash scripts/run_tests.sh [-v] [-B build_dir] [-j num_jobs]
#
# Note: GPU tests will fail gracefully on macOS (no CUDA available)

this_file_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$this_file_dir/set_env_defaults.sh"

build_dir="build"
verbose=""
num_jobs=""

while getopts ":vB:j:" opt; do
  case $opt in
    v) verbose="-v" ;;
    B) build_dir="$OPTARG" ;;
    j) num_jobs="$OPTARG" ;;
    :)
      echo "::error::Option -$OPTARG requires an argument"
      exit 1
      ;;
    \?)
      echo "::error::Invalid option: -$OPTARG"
      exit 1
      ;;
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

# Determine number of parallel jobs (use all available CPUs unless -j is set)
if [ "$(uname)" = "Darwin" ]; then
  max_jobs=$(sysctl -n hw.ncpu)
else
  # nproc would return OMP_NUM_THREADS if set, which defeats our purpose.
  # Unset OMP_NUM_THREADS and OMP_THREAD_LIMIT to get the real number of cores.
  max_jobs=$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc --all)
fi
if [ -n "$num_jobs" ]; then
  if ! [[ "$num_jobs" =~ ^[1-9][0-9]*$ ]]; then
    echo "::error::-j requires a positive integer, got: $num_jobs"
    exit 1
  fi
  # Allow some oversubscription to 2 * max_jobs
  if [ "$num_jobs" -gt $((2 * max_jobs)) ]; then
    num_jobs=$((2 * max_jobs))
  fi
else
  num_jobs=$max_jobs
fi
if [ "$num_jobs" -lt 1 ]; then num_jobs=1; fi

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

# 1. CTest: all gtest tests in parallel.
# Exclude lit test suites from ctest -- they are run individually below.
# Keep ctest-cudaq-unit in this pass; it wraps gtest-based cudaq/unittests.
# GPU tests serialize automatically via RESOURCE_LOCK "gpu" in CMakeLists.txt.
# On machines without a GPU, $gpu_excludes skips gpu_required tests.
echo "=== Running ctest ==="
ctest --output-on-failure --test-dir "$build_dir" -j "$num_jobs" \
  -E "^(ctest-cudaq|ctest-targettests|pycudaq-mlir)$" $gpu_excludes
ctest_status=$?
if [ $ctest_status -ne 0 ]; then
  echo "::error::ctest failed with status $ctest_status"
  status_sum=$((status_sum + 1))
fi

# 2. Main lit tests
echo "=== Running llvm-lit (build/cudaq/test) ==="
"$LLVM_INSTALL_PREFIX/bin/llvm-lit" $verbose --time-tests -j "$num_jobs" \
  --param cudaq_site_config="$build_dir/cudaq/test/lit.site.cfg.py" \
  "$build_dir/cudaq/test"
lit_status=$?
if [ $lit_status -ne 0 ]; then
  echo "::error::llvm-lit (build/cudaq/test) failed with status $lit_status"
  status_sum=$((status_sum + 1))
fi

# 3. Target tests
echo "=== Running llvm-lit (build/targettests) ==="
"$LLVM_INSTALL_PREFIX/bin/llvm-lit" $verbose --time-tests -j "$parallel_jobs" \
  --param cudaq_site_config="$build_dir/targettests/lit.site.cfg.py" \
  "$build_dir/targettests"
targ_status=$?
if [ $targ_status -ne 0 ]; then
  echo "::error::llvm-lit (targettests) failed with status $targ_status"
  status_sum=$((status_sum + 1))
fi

# 4. Python MLIR tests
echo "=== Running llvm-lit (python/tests/mlir) ==="
"$LLVM_INSTALL_PREFIX/bin/llvm-lit" $verbose --time-tests -j "$parallel_jobs" \
  --param cudaq_site_config="$build_dir/python/tests/mlir/lit.site.cfg.py" \
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
