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
# -v flag generates data format for uploading to codecov
# C and C++ coverage reports are generated in the directory 'build/ccoverage'
# Python coverage reports are generated in the directory 'build/pycoverage'
#
# Note:
# The script should be run in the cuda-quantum-devdeps container environment.
# current tested image: ghcr.io/nvidia/cuda-quantum-devdeps:clang16-main
# Don't enable GPU
# C/C++ coverage is located in the ./build/ccoverage directory
# Python coverage is located in the ./build/pycoverage directory

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

# Set envs
if $gen_cpp_coverage; then
    export CUDAQ_ENABLE_CC=ON
    mkdir -p /usr/lib/llvm-16/lib/clang/16/lib/linux
    ln -s /usr/local/llvm/lib/clang/16/lib/x86_64-unknown-linux-gnu/libclang_rt.profile.a /usr/lib/llvm-16/lib/clang/16/lib/linux/libclang_rt.profile-x86_64.a
    export LLVM_PROFILE_FILE=${repo_root}/build/tmp/cudaq-cc/profile-%9m.profraw
fi

# Build project
bash ${repo_root}/scripts/build_cudaq.sh
if [ $? -ne 0 ]; then
    echo "Build cudaq failure: $?" >&2
    exit 1
fi

# Function to run the llvm-cov command
gen_cplusplus_report() {
    if $is_codecov_format; then
        mkdir -p ${repo_root}/build/ccoverage
        llvm-cov show ${objects} -instr-profile=${repo_root}/build/coverage.profdata --ignore-filename-regex="${repo_root}/tpls/*" \
            --ignore-filename-regex="${repo_root}/build/*" --ignore-filename-regex="${repo_root}/unittests/*" 2>&1 > ${repo_root}/build/ccoverage/coverage.txt
    else
        llvm-cov show -format=html ${objects} -instr-profile=${repo_root}/build/coverage.profdata --ignore-filename-regex="${repo_root}/tpls/*" \
            --ignore-filename-regex="${repo_root}/build/*" --ignore-filename-regex="${repo_root}/unittests/*" -o ${repo_root}/build/ccoverage 2>&1
    fi
}

if $gen_cpp_coverage; then
    # Detect toolchain use_llvm_cov=false
    toolchain_contents=$(cat /usr/local/llvm/bootstrap/cc)
    if [[ $toolchain_contents == *"/usr/bin/clang-16"* ]]; then
        use_llvm_cov=true
    else
        echo "Currently not supported, running tests using llvm-lit fails"
        exit 1
    fi

    # Run tests (C++ Unittests)
    python3 -m pip install iqm-client==16.1
    ctest --output-on-failure --test-dir ${repo_root}/build -E ctest-nvqpp -E ctest-targettests
    ctest_status=$?
    /usr/local/llvm/bin/llvm-lit -v --param nvqpp_site_config=${repo_root}/build/test/lit.site.cfg.py ${repo_root}/build/test
    lit_status=$?
    /usr/local/llvm/bin/llvm-lit -v --param nvqpp_site_config=${repo_root}/build/targettests/lit.site.cfg.py ${repo_root}/build/targettests
    targ_status=$?
    /usr/local/llvm/bin/llvm-lit -v --param nvqpp_site_config=${repo_root}/build/python/tests/mlir/lit.site.cfg.py ${repo_root}/build/python/tests/mlir
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
            objects+="-object ${repo_root}/build/$item "
        done

        # The purpose of adding this code is to avoid the llvm-cov show command
        # from being unable to generate a report due to a malformed format error of an object.
        # This is mainly an error caused by a static library, but it has little impact on the coverage rate.
        # Loop until the command succeeds
        while true; do
            output=$(gen_cplusplus_report ${objects})
            status=$?

            # Check if the command failed due to malformed coverage data
            if [ $status -ne 0 ]; then
                echo "Error detected. Attempting to remove problematic object and retry."
                echo "$output"

                # Extract the problematic object from the error message
                problematic_object=$(echo "$output" | grep -oP "error: Failed to load coverage: '\K[^']+")
                echo $problematic_object

                if [ -n "$problematic_object" ]; then
                    # Remove the problematic object from the objects variable
                    objects=$(echo $objects | sed "s|-object $problematic_object||")

                    # Check if the problematic object was successfully removed
                    if [[ $objects != *"-object $problematic_object"* ]]; then
                        echo "Problematic object '$problematic_object' removed. Retrying..."
                    else
                        echo "Failed to remove problematic object '$problematic_object'. Exiting..."
                        exit 1
                    fi
                else
                    echo "No problematic object found in the error message. Exiting..."
                    exit 1
                fi
            else
                echo "Command succeeded."
                break
            fi
        done
    else
        # Use gcov
        echo "Currently not supported, running tests using llvm-lit fails"
        exit 1
    fi
fi

if $gen_py_coverage; then
    pip install pytest-cov
    pip install iqm_client==16.1 --user -vvv
    rm -rf ${repo_root}/_skbuild
    pip install . --user -vvv
    if $is_codecov_format; then
        python3 -m pytest -v python/tests/ --ignore python/tests/backends --cov=cudaq --cov-report=xml:${repo_root}/build/pycoverage/coverage.xml --cov-append
    else
        python3 -m pytest -v python/tests/ --ignore python/tests/backends --cov=cudaq --cov-report=html:${repo_root}/build/pycoverage --cov-append
    fi
    for backendTest in python/tests/backends/*.py; do
        if $is_codecov_format; then
            python3 -m pytest -v $backendTest --cov=cudaq --cov-report=xml:${repo_root}/build/pycoverage/coverage.xml --cov-append
        else
            python3 -m pytest -v $backendTest --cov=cudaq --cov-report=html:${repo_root}/build/pycoverage --cov-append
        fi
        pytest_status=$?
        if [ ! $pytest_status -eq 0 ] && [ ! $pytest_status -eq 5 ]; then
            echo "::error $backendTest tests failed with status $pytest_status."
            exit 1
        fi
    done
fi
