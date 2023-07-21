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
# If BLAS is not found, it will be built from source and installed the location
# defined by the BLAS_INSTALL_PREFIX.
# If OpenSSL is not found, it will be built from source and installed the location
# defined by the OPENSSL_INSTALL_PREFIX.

LLVM_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX:-/opt/llvm}
BLAS_INSTALL_PREFIX=${BLAS_INSTALL_PREFIX:-/usr/local/blas}
OPENSSL_INSTALL_PREFIX=${OPENSSL_INSTALL_PREFIX:-/usr/lib/ssl}

function temp_install_if_command_unknown {
    if [ ! -x "$(command -v $1)" ]; then
        apt-get install -y --no-install-recommends $2
        APT_UNINSTALL="$APT_UNINSTALL $2"
    fi
}

function remove_temp_installs {
  if [ "$APT_UNINSTALL" != "" ]; then
      echo "Uninstalling packages used for bootstrapping: $APT_UNINSTALL"
      apt-get remove -y $APT_UNINSTALL && apt-get autoremove -y --purge
      unset APT_UNINSTALL
  fi
}

set -e
trap remove_temp_installs EXIT
this_file_dir=`dirname "$(readlink -f "${BASH_SOURCE[0]}")"`

if [ ! -x "$(command -v cmake)" ]; then
    apt-get update && apt-get install -y --no-install-recommends cmake
    APT_UNINSTALL="$APT_UNINSTALL $2"
fi
if [ "$CC" == "" ] && [ "$CXX" == "" ]; then
  source "$this_file_dir/install_toolchain.sh" -t gcc12
fi

llvm_dir="$LLVM_INSTALL_PREFIX/lib/cmake/llvm"
if [ ! -d "$llvm_dir" ]; then
  echo "Could not find llvm libraries."

  # Build llvm libraries from source and install them in the install directory
  source "$this_file_dir/build_llvm.sh"
  (return 0 2>/dev/null) && is_sourced=true || is_sourced=false

  if [ ! -d "$llvm_dir" ]; then
    echo "Failed to find directory $llvm_dir."
    if $is_sourced; then return 1; else exit 1; fi
  fi
else 
  echo "Configured C compiler: $CC"
  echo "Configured C++ compiler: $CXX"
fi

if [ ! -x "$(command -v ar)" ] && [ -x "$(command -v "$LLVM_INSTALL_PREFIX/bin/llvm-ar")" ]; then
    ln -s "$LLVM_INSTALL_PREFIX/bin/llvm-ar" /usr/bin/ar
    created_ld_sym_link=$?
    if [ "$created_ld_sym_link" = "" ] || [ ! "$created_ld_sym_link" -eq "0" ]; then
        echo "Failed to find ar or llvm-ar."
    else 
        echo "Setting llvm-ar as the default ar."
    fi
fi

if [ ! -f "$BLAS_INSTALL_PREFIX/libblas.a" ] && [ ! -f "$BLAS_INSTALL_PREFIX/lib/libblas.a" ]; then
  if [ -x "$(command -v apt-get)" ]; then
    apt-get update
  fi

  temp_install_if_command_unknown wget wget
  temp_install_if_command_unknown make make
  temp_install_if_command_unknown gcc gcc
  temp_install_if_command_unknown g++ g++
  temp_install_if_command_unknown gfortran gfortran

  # See also: https://github.com/NVIDIA/cuda-quantum/issues/452
  wget http://www.netlib.org/blas/blas-3.11.0.tgz
  tar -xzvf blas-3.11.0.tgz && cd BLAS-3.11.0
  make && mkdir -p "$BLAS_INSTALL_PREFIX" && mv blas_LINUX.a "$BLAS_INSTALL_PREFIX/libblas.a"
  cd .. && rm -rf blas-3.11.0.tgz BLAS-3.11.0
  remove_temp_installs
fi

if [ ! -d "$OPENSSL_INSTALL_PREFIX" ] || [ -z "$(ls -A "$OPENSSL_INSTALL_PREFIX"/openssl*)" ]; then
  if [ -x "$(command -v apt-get)" ]; then
    apt-get update && apt-get install -y --no-install-recommends perl
  fi

  temp_install_if_command_unknown wget wget
  temp_install_if_command_unknown make make

  wget https://www.openssl.org/source/openssl-3.1.1.tar.gz
  tar -xf openssl-3.1.1.tar.gz && cd openssl-3.1.1
  ./config no-zlib --prefix="$OPENSSL_INSTALL_PREFIX" --openssldir="$OPENSSL_INSTALL_PREFIX"
  make install && cd .. && rm -rf openssl-3.1.1*
  remove_temp_installs
fi
