#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Usage: 
# This script builds and installs a minimal set of dependencies needed to build 
# CUDA Quantum from source. 
#
# Usage: 
# bash install_prerequisites.sh
#
# The necessary LLVM components will be installed in the location defined by the
# LLVM_INSTALL_PREFIX if they do not already exist in that location.
# If OpenBLAS is not found, it will be built from source and installed the location
# defined by the OPENBLAS_INSTALL_PREFIX.
# If OpenSSL is not found, it will be built from source and installed the location
# defined by the OPENSSL_INSTALL_PREFIX.

LLVM_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX:-/opt/llvm}
OPENBLAS_INSTALL_PREFIX=${OPENBLAS_INSTALL_PREFIX:-/usr/local}
OPENSSL_INSTALL_PREFIX=${OPENSSL_INSTALL_PREFIX:-/usr/local}

function temp_install_if_command_unknown {
    if [ ! -x "$(command -v $1)" ]; then
        apt-get install -y --no-install-recommends $2
        APT_UNINSTALL="$APT_UNINSTALL $2"
    fi
}

function remove_temp_installs {
  if [ "$APT_UNINSTALL" != "" ]; then
      echo "Uninstalling packages used for bootstrapping: $APT_UNINSTALL"
      apt-get remove -y $APT_UNINSTALL && apt-get autoremove -y
      unset APT_UNINSTALL
  fi
}

trap remove_temp_installs EXIT
this_file_dir=`dirname "$(readlink -f "${BASH_SOURCE[0]}")"`

if [ ! -x "$(command -v cmake)" ]; then
    apt-get update && apt-get install -y --no-install-recommends cmake
    APT_UNINSTALL="$APT_UNINSTALL $2"
fi
if [ "$CC" == "" ] && [ "$CXX" == "" ]; then
  source "$this_file_dir/install_tool.sh" -t gcc12
fi

llvm_config="$LLVM_INSTALL_PREFIX/bin/llvm-config"
llvm_lib_dir=`"$llvm_config" --libdir 2>/dev/null`
if [ ! -d "$llvm_lib_dir" ]; then
  echo "Could not find llvm libraries."

  # Build llvm libraries from source and install them in the install directory
  source "$this_file_dir/build_llvm.sh"
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

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$OPENBLAS_INSTALL_PREFIX/lib"
openblas_found=`cmake --find-package -DNAME=OpenBLAS -DCOMPILER_ID=GNU -DLANGUAGE=C -DMODE=EXIST | grep -i "OpenBLAS found"`
if [ -z "$openblas_found" ]; then
  temp_install_if_command_unknown wget wget
  temp_install_if_command_unknown make make

  wget -q https://github.com/xianyi/OpenBLAS/releases/download/v0.3.23/OpenBLAS-0.3.23.tar.gz
  tar -xf OpenBLAS-0.3.23.tar.gz && cd OpenBLAS-0.3.23
  # FIXME: set USE_OPENMP to 1 after enabling it in the llvm build.
  make USE_OPENMP=0 && make install PREFIX="$OPENBLAS_INSTALL_PREFIX"
  cd .. && rm -rf OpenBLAS-0.3.23*
fi

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$OPENSSL_INSTALL_PREFIX/lib"
openssl_found=`cmake --find-package -DNAME=OpenSSL -DCOMPILER_ID=GNU -DLANGUAGE=C -DMODE=EXIST | grep -i "OpenBLAS found"`
if [ -z "$openssl_found" ]; then
  temp_install_if_command_unknown git git
  temp_install_if_command_unknown make make

  wget -q https://www.openssl.org/source/openssl-3.1.1.tar.gz
  tar -xf openssl-3.1.1.tar.gz && cd openssl-3.1.1
  ./config --prefix="$OPENSSL_INSTALL_PREFIX" --openssldir="$OPENSSL_INSTALL_PREFIX" -static zlib
  make install && cd .. && rm -rf openssl-3.1.1*
fi
