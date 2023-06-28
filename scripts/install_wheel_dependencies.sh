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
# the python bindings from source. It expects to be run on the Ubuntu Dev Image, 
# otherwise refer to the installation steps in `build_cudaq.sh` for any further 
# missing dependencies.

LLVM_DIR=${LLVM_DIR:-/opt/llvm}
CPR_DIR=${CPR_DIR:-/cpr/install}
BLAS_PATH=${BLAS_PATH:-/usr/lib64/libblas.a}

echo "Installing required packages for python wheel..."

# Run the script from the top-level of the repo
working_dir=`pwd`
this_file_dir=`dirname "$(readlink -f "${BASH_SOURCE[0]}")"`
repo_root=$(cd "$this_file_dir" && git rev-parse --show-toplevel)
cd "$repo_root"

# Clone the submodules (skipping llvm)
echo "Cloning submodules..."
git -c submodule.tpls/llvm.update=none submodule update --init --recursive

if [ ! -d "$LLVM_DIR" ]; then
  echo "Could not find llvm libraries."
  # Build llvm libraries from source and install them in the install directory
  llvm_build_script=`pwd`/scripts/build_llvm.sh
  cd "$working_dir" && source "$llvm_build_script" -c $build_configuration && cd "$repo_root"
  (return 0 2>/dev/null) && is_sourced=true || is_sourced=false
else 
  echo "Configured C compiler: $CC"
  echo "Configured C++ compiler: $CXX"
  echo "LLVM directory: $LLVM_DIR"
fi
export LLVM_DIR=$LLVM_DIR

# Get necessary pacakges from apt.
apt-get update && apt-get install -y --no-install-recommends build-essential wget gfortran python3 python3-pip python3.10-venv 
if [ ! -d "$CPR_DIR" ]; then
  echo "Did not find a libcpr install dir"
  echo "Installing libcpr"
  
  # Install in same parent directory as cuda-quantum.
  cd "$repo_root" && cd /
  echo `pwd`

  echo "Cloning libcpr"
  git clone https://github.com/libcpr/cpr
  echo "Building libcpr"
  cd cpr && rm -rf build && mkdir build && cd build
  cmake .. -G Ninja -DCPR_FORCE_USE_SYSTEM_CURL=FALSE 
                    -DCMAKE_INSTALL_LIBDIR=lib 
                    -DOPENSSL_USE_STATIC_LIBS=TRUE 
                    -DBUILD_SHARED_LIBS=FALSE 
                    -DOPENSSL_ROOT_DIR=/usr/local/ssl 
                    -DCMAKE_POSITION_INDEPENDENT_CODE=TRUE 
                    -DCMAKE_INSTALL_PREFIX=$CPR_DIR
  
  ninja install
else 
  echo "libcpr directory: $CPR_DIR"
fi
export CPR_DIR=$CPR_DIR
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

# Check for pip and install
# wget https://bootstrap.pypa.io/get-pip.py && python3.10 ./get-pip.py

# FIXME: Hard-coded on python3.10
# Install the python build package via pip
python3.10 -m pip install build