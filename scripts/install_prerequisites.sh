#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Usage: 
# This script installs all dependencies needed to build pip install python 
# bindings from source. 

# Usage: 
# `bash install_wheel_dependencies.sh`
# -or-
# `LLVM_INSTALL_PREFIX=/path/to/llvm BLAS_PATH=/path/to/libblas.a bash install_wheel_dependencies.sh`

LLVM_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX:-/opt/llvm}
BLAS_PATH=${BLAS_PATH:-/usr/lib64/libblas.a}

llvm_config="$LLVM_INSTALL_PREFIX/bin/llvm-config"
llvm_lib_dir=`"$llvm_config" --libdir 2>/dev/null`
if [ ! -d "$llvm_lib_dir" ]; then
  echo "Could not find llvm libraries."

  # Build llvm libraries from source and install them in the install directory
  source "$(git rev-parse --show-toplevel)/scripts/build_llvm.sh"
  (return 0 2>/dev/null) && is_sourced=true || is_sourced=false

  llvm_lib_dir=`"$llvm_config" --libdir 2>/dev/null`
  if [ ! -d "$llvm_lib_dir" ]; then
    echo "Failed to find llvm libraries directory $llvm_lib_dir."
    if $is_sourced; then return 1; else exit 1; fi
  fi
else 
  echo "Configured C compiler: $CC"
  echo "Configured C++ compiler: $CXX"
fi

# TODO: 
#wget -q https://github.com/xianyi/OpenBLAS/releases/download/v0.3.23/OpenBLAS-0.3.23.tar.gz
#tar -xf OpenBLAS-0.3.23.tar.gz && cd OpenBLAS-0.3.23
#... && make USE_OPENMP=1 && make install
# mv blas_LINUX.a "$BLAS_PATH"

# TODO: what to do about the compiler prerequisites?
# apt-get update && apt-get install -y --no-install-recommends \
#    wget build-essential python3-venv gfortran 
