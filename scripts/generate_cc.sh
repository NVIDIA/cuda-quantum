#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
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
# -v flag generates data format for uploading to codecov
# C and C++ coverage reports are generated in the directory 'build/ccoverage'
# Python coverage reports are generated in the directory 'build/pycoverage'
#
# Note:
# Run in a CUDA-Q development container:
# docker build -t cuda-quantum-devdeps:13.0-llvm-ext \
#   --build-arg cuda_version=13.0 \
#   --build-arg base_image=ghcr.io/nvidia/cuda-quantum-devdeps:amd64-llvm-main \
#   --build-arg ompidev_image=ghcr.io/nvidia/cuda-quantum-devdeps:ompi-amd64-cu13.0-main \
#   -f docker/build/devcontainer.Dockerfile .
#
# From the repository root, run with GPU access (requires an NVIDIA GPU,
# compatible driver, and NVIDIA Container Toolkit):
# docker run --rm --gpus all -v "$PWD:/workspace/cuda-quantum" \
#   -w /workspace/cuda-quantum cuda-quantum-devdeps:13.0-llvm-ext \
#   bash scripts/generate_cc.sh -c -p
#
# To run without GPU access, omit --gpus all:
# docker run --rm -v "$PWD:/workspace/cuda-quantum" \
#   -w /workspace/cuda-quantum cuda-quantum-devdeps:13.0-llvm-ext \
#   bash scripts/generate_cc.sh -c -p

if [ $# -lt 1 ]; then
    echo "Please provide at least one parameter"
    exit 1
fi

gen_cpp_coverage=false
gen_py_coverage=false
is_codecov_format=false

# Process command line arguments
__optind__=$OPTIND
OPTIND=1
while getopts ":cpv" opt; do
    case $opt in
    c)
        gen_cpp_coverage=true
        ;;
    p)
        gen_py_coverage=true
        ;;
    v)
        is_codecov_format=true
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

# Runs the source-tree pytest suites using $pytest as the launcher
# (mirrors the test_in_devenv.yml CI job). Pytest exit code 5 (no tests
# collected) is tolerated for the per-backend files.
# PYTHONPATH points at build/python (as in run_tests.sh); without it the
# C++-only cudaq/ directory in the repo root shadows the real package.
run_pytest_suites() {
    local s=0 rc
    export PYTHONPATH=${repo_root}/build/python${PYTHONPATH:+:$PYTHONPATH}
    OMP_NUM_THREADS=1 $pytest $pytest_parallel -v ${repo_root}/python/tests \
        --ignore ${repo_root}/python/tests/backends \
        --ignore ${repo_root}/python/tests/contrib || s=1
    for f in ${repo_root}/python/tests/backends/*.py; do
        OMP_NUM_THREADS=1 $pytest -v -s --durations=0 "$f"; rc=$?
        [ $rc -ne 0 ] && [ $rc -ne 5 ] && s=1
    done
    OMP_NUM_THREADS=1 $pytest -v ${repo_root}/python/tests/contrib || s=1
    return $s
}

# Runs a build step and exits immediately if it fails: a broken build would
# produce coverage data that looks valid but covers almost nothing.
run_build_step() {
    "$@"
    local status=$?
    if [ $status -ne 0 ]; then
        echo "::error Build step '$*' failed with status $status." >&2
        exit 1
    fi
}

# Set envs
# Test failures are collected here and reported at the end: they do not stop the
# run, but the script exits non-zero once the reports have been written.
test_failures=0
build_cmake_args=(-DCUDAQ_TEST_OMP_SLOTS=2)

# The development container presets CUDAHOSTCXX to clang, so build_cudaq.sh
# leaves CUDAFLAGS untouched. Allow nvcc to use that compiler for this build.
if [ -z "${CUDAFLAGS:-}" ] && [ -n "${CUDAHOSTCXX:-}" ] && \
   "$CUDAHOSTCXX" --version 2>/dev/null | grep -qi clang; then
    export CUDAFLAGS="-allow-unsupported-compiler --compiler-options --stdlib=libstdc++"
    if [ -d "${GCC_TOOLCHAIN:-}" ]; then
        CUDAFLAGS+=" --compiler-options --gcc-toolchain=\"$GCC_TOOLCHAIN\""
    fi
fi

# CMakeLists.txt sets CMAKE_CUDA_FLAGS before enabling CUDA, so CMake does not
# initialize it from CUDAFLAGS during the scikit-build-core package build.
if [ -n "${CUDAFLAGS:-}" ]; then
    cuda_cmake_arg=$(python3 -c 'import os, shlex; print(shlex.quote("-DCMAKE_CUDA_FLAGS=" + os.environ["CUDAFLAGS"]))')
    export CMAKE_ARGS="$cuda_cmake_arg${CMAKE_ARGS:+ $CMAKE_ARGS}"
fi

if $gen_cpp_coverage; then
    export CUDAQ_ENABLE_CC=ON
    # Work around llvm/llvm-project#179316: clang 22.1.x asserts when a
    # coverage region maps into a -isystem header (e.g. MLIR .inc files).
    # Enable system-headers coverage collection instead; fixed in clang
    # 22.1.7, drop this flag once the toolchain is updated.
    # Instrumented libraries all export the __llvm_profile_* runtime
    # symbols, which would trip the MLIR symbol-closure check; coverage
    # build artifacts are not shipped, so the check is disabled here.
    # CXXFLAGS reaches both builds. Pass the CMake option directly to
    # build_cudaq.sh and through CMAKE_ARGS to scikit-build-core. -Wno-error
    # keeps clang's unused-argument warning at link time from failing configure
    # in tpls/cpr's try_compile (it adds its own -Werror).
    export CUDAQ_WERROR=OFF
    export CXXFLAGS="${CXXFLAGS:+$CXXFLAGS }-Wno-error=unused-command-line-argument -mllvm -system-headers-coverage"
    export CMAKE_ARGS="${CMAKE_ARGS:+$CMAKE_ARGS }-DCUDAQ_CHECK_MLIR_SYMBOL_CLOSURE=OFF"
    build_cmake_args+=(-DCUDAQ_CHECK_MLIR_SYMBOL_CLOSURE=OFF)
fi

# Build project
# Profile counters are written per-phase so that data from build, ctest/lit,
# and Python test runs can be merged (or re-collected) separately.
export LLVM_PROFILE_FILE=${repo_root}/build/tmp/cudaq-cc/profile-build-%9m.profraw
mkdir -p ${repo_root}/build/tmp/cudaq-cc
run_build_step bash ${repo_root}/scripts/build_cudaq.sh -v -- "${build_cmake_args[@]}"

if $gen_cpp_coverage; then
    # Install the cudaq Python package and test dependencies needed by the
    # source-tree pytest suites below. The source tree's cudaq/ directory has no
    # __init__.py, so without an installed package the tests silently import it
    # as a namespace package and every test errors out: fail fast instead.
    run_build_step python3 -m pip install -r ${repo_root}/requirements-tests-backend.txt --break-system-packages
    rm -rf ${repo_root}/_skbuild
    run_build_step pip install ${repo_root} --user -vvv
    # Enables the contrib tests (self-skipping when qiskit is missing).
    python3 -m pip install qiskit --user

    export LLVM_PROFILE_FILE=${repo_root}/build/tmp/cudaq-cc/profile-run-tests-%9m.profraw

    # ctest, lit suites, and interop pytest through the same wrapper as CI.
    # Test failures do not stop the run: the reports below are still generated
    # from whatever the passing tests exercised, and the script exits non-zero
    # once everything has been collected.
    bash ${repo_root}/scripts/run_tests.sh -v -B ${repo_root}/build
    run_tests_status=$?
    if [ ! $run_tests_status -eq 0 ]; then
      echo "::error run_tests.sh failed with status $run_tests_status."
      test_failures=$((test_failures + 1))
    fi

    # Source-tree pytest suites.
    pytest="python3 -m pytest"
    pytest_parallel=""
    python3 -c "import xdist" 2>/dev/null && \
        pytest_parallel="-n $(( ($(nproc) + 1) / 2 ))"
    run_pytest_suites || {
        echo "::error Python test suites failed."
        test_failures=$((test_failures + 1))
    }

    # Generate report: merge all profraw fragments, then scan the build tree
    # for instrumented binaries (those carrying __llvm_covmap); non-ELF files
    # like the nvq++ driver script and artifacts without coverage data are
    # skipped automatically.
    llvm-profdata merge -sparse ${repo_root}/build/tmp/cudaq-cc/profile-*.profraw \
        -o ${repo_root}/build/coverage.profdata
    objects=()
    while IFS= read -r f; do
        objects+=(-object "$f")
    done < <(find ${repo_root}/build -type f \( -name "*.so*" -o -perm /111 \) \
        -exec sh -c 'readelf -S "$1" 2>/dev/null | grep -q __llvm_covmap' _ {} \; -print)

    if [ ${#objects[@]} -eq 0 ]; then
        echo "::error No instrumented objects found under ${repo_root}/build."
        exit 1
    fi
    echo "Using ${#objects[@]} instrumented objects."

    # Exclude non-product code from the report: third-party sources, build
    # artifacts, test code, and system/vendor headers.
    coverage_ignore_args=(
        --ignore-filename-regex="${repo_root}/tpls/*"
        --ignore-filename-regex="${repo_root}/build/*"
        --ignore-filename-regex="(^|/)unittests/"
        --ignore-filename-regex="^/?usr/(include|local)/"
    )

    mkdir -p ${repo_root}/build/ccoverage
    common_args=("${objects[@]}" -instr-profile=${repo_root}/build/coverage.profdata
        "${coverage_ignore_args[@]}")
    if $is_codecov_format; then
        llvm-cov show "${common_args[@]}" > ${repo_root}/build/ccoverage/coverage.txt 2>&1 || {
            echo "::error llvm-cov show failed."; exit 1; }
    else
        llvm-cov show -format=html "${common_args[@]}" -o ${repo_root}/build/ccoverage 2>&1 || {
            echo "::error llvm-cov show failed."; exit 1; }
    fi
    llvm-cov export "${common_args[@]}" > ${repo_root}/build/ccoverage/coverage.json || {
        echo "::error llvm-cov export failed."; exit 1; }
fi

if $gen_py_coverage; then
    # Needs to be installed outside of venv
    run_build_step python3 -m pip install -r ${repo_root}/requirements-tests-backend.txt --break-system-packages

    venv_dir=${repo_root}/build/venv-coverage
    if python3 -c 'import ensurepip' 2>/dev/null; then
        run_build_step python3 -m venv --system-site-packages "$venv_dir"
    elif command -v uv >/dev/null 2>&1; then
        run_build_step uv venv --seed --allow-existing --system-site-packages \
            --python "$(command -v python3)" "$venv_dir"
    else
        echo "::error Install the Python venv package or uv to create the coverage environment." >&2
        exit 1
    fi
    . "${venv_dir}/bin/activate"
    run_build_step pip install coverage
    rm -rf ${repo_root}/_skbuild
    run_build_step pip install -e . -vvv
    # Enables the contrib tests (self-skipping when qiskit is missing).
    pip install qiskit
    mkdir -p ${repo_root}/build/pycoverage
    # Run the source-tree pytest suites under coverage.py. Tests run
    # serially (no xdist) so every test executes inside the `coverage run`
    # process and is actually traced.
    pytest="coverage run -a -m pytest"
    pytest_parallel=""
    run_pytest_suites || {
        echo "::error Python coverage tests failed."
        test_failures=$((test_failures + 1))
    }

    # generate report
    if $is_codecov_format; then
        coverage xml -o ${repo_root}/build/pycoverage/coverage.xml --omit=${repo_root}/python/tests/*,/usr/lib/*
    else
        coverage html -d ${repo_root}/build/pycoverage --omit=${repo_root}/python/tests/*,/usr/lib/*
    fi
fi

if [ ${test_failures:-0} -ne 0 ]; then
    echo "::error Coverage was generated, but $test_failures test stage(s) failed; the report is incomplete." >&2
    exit 1
fi
