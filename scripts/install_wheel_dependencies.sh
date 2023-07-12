#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Usage: 
# This script is called in the file `/cuda-quantum/setup.py` and is responsible
# for installing all dependencies needed to build a python wheel or to pip install
# the python bindings from source. 
# This script is also called by `build_wheel.sh`, which is the script for generating
# and distributing our python wheels.

# Usage: 
# `bash install_wheel_dependencies.sh`
# -or-
# `LLVM_INSTALL_PREFIX=/path/to/llvm BLAS_PATH=/path/to/libblas.a bash install_wheel_dependencies.sh`

# Prerequisites:
# It expects to be run on the Ubuntu Dev Image, otherwise refer to the installation 
# steps in `build_cudaq.sh` for any further missing dependencies.

LLVM_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX:-/opt/llvm}
BLAS_PATH=${BLAS_PATH:-/usr/lib64/libblas.a}

echo "Installing required packages for python wheel..."

# Run the script from the top-level of the repo
working_dir=`pwd`
this_file_dir=`dirname "$(readlink -f "${BASH_SOURCE[0]}")"`
repo_root=$(cd "$this_file_dir" && git rev-parse --show-toplevel)
cd "$repo_root"

# Check if apt-get exists on the system. If not, at this time,
# we assume we're in the manylinux image. 
apt-get update 
if [ $? -eq 0 ]; then
    echo "Found apt-get in: "
    echo which apt-get 
    echo "Updating apt and installing needed packages..."
    apt-get install -y --no-install-recommends build-essential python3.10-venv gfortran
else
    echo "Could not find apt-get. Doing nothing..."
    # Maybe need to install wget here. TBD.
    # yes | dnf check-update && dnf upgrade
fi

if [ ! -d "$LLVM_INSTALL_PREFIX" ]; then
  echo "Could not find llvm libraries."
  # Build llvm libraries from source and install them in the install directory
  llvm_build_script=`pwd`/scripts/build_llvm.sh
  cd "$working_dir" && source "$llvm_build_script" -c $build_configuration && cd "$repo_root"
  (return 0 2>/dev/null) && is_sourced=true || is_sourced=false
else 
  echo "Configured C compiler: $CC"
  echo "Configured C++ compiler: $CXX"
  echo "LLVM directory: $LLVM_INSTALL_PREFIX"
fi

cd "$repo_root"

if [ ! -f "$BLAS_PATH" ]; then
  echo "Did not find libblas.a\n Installing libblas"
  # Install BLAS 
  cd "$repo_root" && cd /
  wget http://www.netlib.org/blas/blas-3.11.0.tgz
  tar -xzvf blas-3.11.0.tgz && cd BLAS-3.11.0
  make && mv blas_LINUX.a /usr/lib64/libblas.a
else 
  echo "Found libblas.a"
fi
cd "$repo_root"

# FIXME: Hard-coded on python3.10
# Install the python build package via pip
LLVM_INSTALL_PREFIX="$LLVM_INSTALL_PREFIX" python3.10 -m pip install build pytest scikit-build
