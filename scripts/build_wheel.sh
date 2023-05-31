#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Usage:


# The idea is to run this from the docker image that we've already spit up for the CI.
# This allows you to build the wheels without doing an entirely new build of CUDA Quantum.
# It also allows someone to run this shell script from their machine and handle all of the
# extra dependencies for them.


# TODO: Handle all of the downloading of LLVM and whatnot here the same way
# we do in the outer build script. This should make sure all of the proper 
# dependencies are in place on the machine for anyone who wants to build off
# of setup.py

#  apt-get install python3.10-venv
#  python3 -m pip install build

LLVM_DIR=${LLVM_DIR:-/opt/llvm/clang-16/lib/cmake/llvm}

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
  export LLVM_DIR=$LLVM_DIR
  echo "Configured C compiler: $CC"
  echo "Configured C++ compiler: $CXX"
  echo "LLVM directory: $LLVM_DIR"
fi

# Find or configure SSL
REQUIRED_PKG="some-package"
PKG_OK=$(dpkg-query -W --showformat='${Status}\n' $REQUIRED_PKG|grep "install ok installed")
echo Checking for $REQUIRED_PKG: $PKG_OK
if [ "" = "$PKG_OK" ]; then
  echo "Could not find $REQUIRED_PKG. Installing $REQUIRED_PKG."
  apt-get --yes install $REQUIRED_PKG
fi

# Find or configure CPR


# Find or configure BLAS






python3 -m build --wheel 

cd working_dir