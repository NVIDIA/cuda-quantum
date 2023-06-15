#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This script is intended to be run by the CI or the local developer to
# build a wheel for the CUDA Quantum python bindings.
# It will handle calling `/cuda-quantum/scripts/wheel_dependencies.sh`
# for any missing dependencies.
#
# Usage: 
# `bash build_wheel.sh`
# -or-
# `LLVM_DIR=/path/to/llvm/lib/cmake/llvm  CPR_DIR=/path/to/cpr/install BLAS_PATH=/path/to/libblas.a  bash build_wheel.sh`
#
# Prerequisites:
# All tools required to run this script are installed when using the dev container definition
# in this repository. Any remaining dependencies are automatically installed here through a
# call to `scripts/wheel_dependencies.sh`. 

LLVM_DIR=${LLVM_DIR:-/opt/llvm}
CPR_DIR=${CPR_DIR:-/cpr/install}
BLAS_PATH=${BLAS_PATH:-/usr/lib64/libblas.a}

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

# Call the dependency script.
# bash /cuda-quantum/scripts/wheel_dependencies.sh

# Return to the outer level of CUDA Quantum to build the wheel off of setup.py
cd "$repo_root"

# TODO: Set the srcdir here to dump the wheels and all extra files into a contained folder.
# Build the wheel only (no sdist). 
# The setup.py file will call `wheel_dependencies.sh` and 
# handle all dependency installation for someone using our
# Docker image.
python3.10 -m build --wheel

cd "$working_dir"