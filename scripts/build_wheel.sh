#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Usage: `bash build_wheel.sh`

LLVM_DIR=${LLVM_DIR:-/opt/llvm/lib/cmake/llvm}
CPR_DIR=${CPR_DIR:-/cpr/install}
BLAS_PATH=${BLAS_PATH:-/usr/lib64/libblas.a}
# Get the CC and CXX directories and export so scikit-build can find them.
# CC=${CC:-/opt/llvm/clang-16/bin/clang}
# CXX=${CXX:-/opt/llvm/clang-16/bin/clang++}
# export CC=$CC
# export CXX=$CXX

# Run the script from the top-level of the repo
working_dir=`pwd`
this_file_dir=`dirname "$(readlink -f "${BASH_SOURCE[0]}")"`
repo_root=$(cd "$this_file_dir" && git rev-parse --show-toplevel)
cd "$repo_root"

# Remove any old builds and dist's
rm -rf _skbuild
rm -rf cuda_quantum.egg-info
rm -rf python/cuda_quantum.egg-info
rm -rf dist
rm -rf wheelhouse

# TODO: Set the srcdir here to dump the wheels and all extra files into a contained folder.
# Build the wheel only (no sdist). 
# The setup.py file will call `wheel_dependencies.sh` and 
# handle all dependency installation for someone using our
# Docker image.
python3.10 -m build --wheel

cd "$working_dir"