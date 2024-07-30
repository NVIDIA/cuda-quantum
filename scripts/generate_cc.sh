#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Usage:
# bash scripts/generate_cc.sh -c
# -or-
# bash scripts/generate_cc.sh -p
# -or-
# bash scripts/generate_cc.sh -c -p
# -c flag generates coverage information for C and C++ codes.
# -p flag generates coverage information for Python codes.
# C and C++ coverage reports are generated in the directory 'build/ccoverage'
# Python coverage reports are generated in the directory 'build/pycoverage'
#
# Note:
# The script should be run in the cuda-quantum-devdeps container environment.
# Currently, Python code cannot display the coverage of kernel functions.

if [ $# -lt 1 ]; then
  echo "Please provide at least one parameter"
  exit 1
fi

gen_cpp_coverage=false
gen_py_coverage=false

# Process command line arguments
__optind__=$OPTIND
OPTIND=1
while getopts ":cp" opt; do
  case $opt in
  c)
    gen_cpp_coverage=true
    ;;
  p)
    gen_py_coverage=true
    ;;
  \?)
    echo "Invalid command line option -$OPTARG" >&2
    exit 1
    ;;
  esac
done
OPTIND=$__optind__

# Repo root
this_file_dir=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
repo_root=$(cd "$this_file_dir" && git rev-parse --show-toplevel)

# Set envs
if $gen_cpp_coverage; then
  export CUDAQ_ENABLE_CC=ON
  export LLVM_PROFILE_FILE=${repo_root}/build/tmp/cudaq-cc/profile-%9m.profraw
fi

# Build project
bash ${repo_root}/scripts/build_cudaq.sh
if [ $? -ne 0 ]; then
    echo "Build cudaq failure: $?" >&2
    exit 1
fi

if $gen_cpp_coverage; then
  # Detect toolchain
  use_llvm_cov=false
  toolchain_contents=$(cat "$LLVM_INSTALL_PREFIX/bootstrap/cc")
  if [[ $toolchain_contents == *"$LLVM_INSTALL_PREFIX/bin/clang"* ]]; then
    use_llvm_cov=true
  else
    echo "Currently not supported, running tests using llvm-lit fails"
    exit 1
  fi

  # Run tests (C++ Unittests)
  python3 -m pip install iqm-client==16.1
  ctest --output-on-failure --test-dir ${repo_root}/build -E ctest-nvqpp -E ctest-targettests
  ctest_status=$?
  $LLVM_INSTALL_PREFIX/bin/llvm-lit -v --param nvqpp_site_config=${repo_root}/build/test/lit.site.cfg.py ${repo_root}/build/test
  lit_status=$?
  $LLVM_INSTALL_PREFIX/bin/llvm-lit -v --param nvqpp_site_config=${repo_root}/build/targettests/lit.site.cfg.py ${repo_root}/build/targettests
  targ_status=$?
  $LLVM_INSTALL_PREFIX/bin/llvm-lit -v --param nvqpp_site_config=${repo_root}/build/python/tests/mlir/lit.site.cfg.py ${repo_root}/build/python/tests/mlir
  pymlir_status=$?
  if [ ! $ctest_status -eq 0 ] || [ ! $lit_status -eq 0 ] || [ $targ_status -ne 0 ] || [ $pymlir_status -ne 0 ]; then
    echo "::error C++ tests failed (ctest status $ctest_status, llvm-lit status $lit_status, \
    target tests status $targ_status, Python MLIR status $pymlir_status)."
    exit 1
  fi

  # Run tests (Python tests)
  rm -rf ${repo_root}/_skbuild
  pip install ${repo_root} --user -vvv
  python3 -m pytest -v ${repo_root}/python/tests/ --ignore ${repo_root}/python/tests/backends
  for backendTest in ${repo_root}/python/tests/backends/*.py; do
    python3 -m pytest -v $backendTest
    pytest_status=$?
    if [ ! $pytest_status -eq 0 ] && [ ! $pytest_status -eq 5 ]; then
      echo "::error $backendTest tests failed with status $pytest_status."
      exit 1
    fi
  done

  # Generate report
  if $use_llvm_cov; then
    llvm-profdata merge -sparse ${repo_root}/build/tmp/cudaq-cc/profile-*.profraw -o ${repo_root}/build/coverage.profdata
    binarys=($(sed -n -e '/Linking CXX shared library/s/^.*Linking CXX shared library //p' \
      -e '/Linking CXX static library/s/^.*Linking CXX static library //p' \
      -e '/Linking CXX shared module/s/^.*Linking CXX shared module //p' \
      -e '/Linking CXX executable/s/^.*Linking CXX executable //p' ${repo_root}/build/logs/ninja_output.txt))
    objects=""
    for item in "${binarys[@]}"; do
      if [ "$item" != "lib/libCUDAQuantumMLIRCAPI.a" ] && [ "$item" != "lib/libOptTransforms.a" ]; then
        objects+="-object ${repo_root}/build/$item "
      fi
    done
    llvm-cov show -format=html ${objects} -instr-profile=${repo_root}/build/coverage.profdata --ignore-filename-regex="${repo_root}/tpls/*" \
      --ignore-filename-regex="${repo_root}/build/*" --ignore-filename-regex="${repo_root}/unittests/*" -o ${repo_root}/build/ccoverage
  else
    # Use gcov
    echo "Currently not supported, running tests using llvm-lit fails"
    exit 1
  fi
fi

if $gen_py_coverage; then
  pip install pytest-cov
  pip install iqm_client==16.1 --user -vvv
  pip install . --user -vvv
  python3 -m pytest -v python/tests/ --ignore python/tests/backends --cov=./python --cov-report=html:${repo_root}/build/pycoverage --cov-append
  for backendTest in python/tests/backends/*.py; do
    python3 -m pytest -v $backendTest --cov=./python --cov-report=html:${repo_root}/build/pycoverage --cov-append
    pytest_status=$?
    if [ ! $pytest_status -eq 0 ] && [ ! $pytest_status -eq 5 ]; then
      echo "::error $backendTest tests failed with status $pytest_status."
      exit 1
    fi
  done
fi
