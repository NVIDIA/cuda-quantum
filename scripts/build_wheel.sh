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
# of setup.py, but also allow it to be built within a manylinux container.

# These are the following packages needed on top of the base docker image:
#  apt-get install build-essential
#  apt-get install gfortran
#  apt-get install python3.10-venv
#  python3 -m pip install build


LLVM_DIR=${LLVM_DIR:-/opt/llvm/clang-16/lib/cmake/llvm}
CPR_DIR=${CPR_DIR:-/cpr/install}
# SSL_DIR
# BLAS_DIR

# REMOVE ME ONCE THE CONDITIONAL BELOW IS SORTED
export LLVM_DIR=$LLVM_DIR
export CPR_DIR=$LLVM_DIR

# Run the script from the top-level of the repo
working_dir=`pwd`
this_file_dir=`dirname "$(readlink -f "${BASH_SOURCE[0]}")"`
repo_root=$(cd "$this_file_dir" && git rev-parse --show-toplevel)
cd "$repo_root"

# Clone the submodules (skipping llvm)
# echo "Cloning submodules..."
# git -c submodule.tpls/llvm.update=none submodule update --init --recursive

# if [ ! -d "$LLVM_DIR" ]; then
#   echo "Could not find llvm libraries."
#   # Build llvm libraries from source and install them in the install directory
#   llvm_build_script=`pwd`/scripts/build_llvm.sh
#   cd "$working_dir" && source "$llvm_build_script" -c $build_configuration && cd "$repo_root"
#   (return 0 2>/dev/null) && is_sourced=true || is_sourced=false
# else 
#   export LLVM_DIR=$LLVM_DIR
#   echo "Configured C compiler: $CC"
#   echo "Configured C++ compiler: $CXX"
#   echo "LLVM directory: $LLVM_DIR"
# fi

# Find or configure package
# REQUIRED_PKG="<package-name>"
# PKG_OK=$(dpkg-query -W --showformat='${Status}\n' $REQUIRED_PKG|grep "install ok installed")
# echo Checking for $REQUIRED_PKG: $PKG_OK
# if [ "" = "$PKG_OK" ]; then
#   echo "Could not find $REQUIRED_PKG. Installing $REQUIRED_PKG."
#   apt-get --yes install $REQUIRED_PKG
# fi

# # SSL
# # Move to a layer outside of the repository
# cd "$repo_root" && cd /
# dnf install -y openssh-clients wget glibc-static zlib-static perl-core --nobest
# git clone https://github.com/openssl/openssl
# cd openssl && ./config --prefix=/usr/local/ssl --openssldir=/usr/local/ssl -static zlib
# make install && cd 

# # Find or configure CPR
# cd / && git clone https://github.com/libcpr/cpr
# cd cpr && mkdir build && cd build
# cmake .. -G Ninja -DCPR_FORCE_USE_SYSTEM_CURL=FALSE 
#                   -DCMAKE_INSTALL_LIBDIR=lib 
#                   -DOPENSSL_USE_STATIC_LIBS=TRUE 
#                   -DBUILD_SHARED_LIBS=FALSE 
#                   -DOPENSSL_ROOT_DIR=/usr/local/ssl 
#                   -DCMAKE_POSITION_INDEPENDENT_CODE=TRUE 
#                   -DCMAKE_INSTALL_PREFIX=/cpr/install
# ninja install
# # Find or configure BLAS within CPR dir
# wget http://www.netlib.org/blas/blas-3.11.0.tgz
# tar -xzvf blas-3.11.0.tgz && cd BLAS-3.11.0
# make && mv blas_LINUX.a /usr/lib64/libblas.a
# # cd ../..

# Return to the outer level of CUDA Quantum to build the wheel off of setup.py
cd "$repo_root"

# Build the wheel only (no sdist).
# Note: Using `--no-isolation` to prevent this from running in a pyenv. 
# pybind doesn't like the pyenv and returns an error on line 50 of
# `tpls/pybind11/tools/pybind11Tools.cmake`
# python3 -m build --wheel #--no-isolation 
python3 setup.py bdist_wheel

cd "$working_dir"