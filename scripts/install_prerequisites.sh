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
# If LLVM components need to be built from source, pybind11 will be built and 
# installed in the location defined by PYBIND11_INSTALL_PREFIX unless that folder 
# already exists.
# If BLAS is not found, it will be built from source and installed the location
# defined by the BLAS_INSTALL_PREFIX.
# If OpenSSL is not found, it will be built from source and installed the location
# defined by the OPENSSL_INSTALL_PREFIX.

LLVM_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX:-/opt/llvm}
PYBIND11_INSTALL_PREFIX=${PYBIND11_INSTALL_PREFIX:-/usr/local/pybind11}
BLAS_INSTALL_PREFIX=${BLAS_INSTALL_PREFIX:-/usr/local/blas}
OPENSSL_INSTALL_PREFIX=${OPENSSL_INSTALL_PREFIX:-/usr/lib/ssl}

function create_llvm_symlinks {
  if [ ! -x "$(command -v ld)" ] && [ -x "$(command -v "$LLVM_INSTALL_PREFIX/bin/ld.lld")" ]; then
    ln -s "$LLVM_INSTALL_PREFIX/bin/ld.lld" /usr/bin/ld
    echo "Setting lld linker as the default linker."
  fi
  if [ ! -x "$(command -v ar)" ] && [ -x "$(command -v "$LLVM_INSTALL_PREFIX/bin/llvm-ar")" ]; then
    ln -s "$LLVM_INSTALL_PREFIX/bin/llvm-ar" /usr/bin/ar
    echo "Setting llvm-ar as the default ar."
  fi
}

function temp_install_if_command_unknown {
    if [ ! -x "$(command -v $1)" ]; then
        apt-get install -y --no-install-recommends $2
        APT_UNINSTALL="$APT_UNINSTALL $2"
    fi
}

function remove_temp_installs {
  if [ -n "$APT_UNINSTALL" ]; then
      echo "Uninstalling packages used for bootstrapping: $APT_UNINSTALL"
      apt-get remove -y $APT_UNINSTALL && apt-get autoremove -y --purge
      unset APT_UNINSTALL
      create_llvm_symlinks # uninstalling other compiler tools may have removed the symlinks
  fi
}

read __errexit__ < <(echo $SHELLOPTS | egrep -o '(^|:)errexit(:|$)' || echo)
function exit_gracefully {
  remove_temp_installs
  if [ -z "$__errexit__" ]; then set +e; fi
}

set -e && trap exit_gracefully EXIT
this_file_dir=`dirname "$(readlink -f "${BASH_SOURCE[0]}")"`

if [ ! -x "$(command -v cmake)" ]; then
  temp_install_if_command_unknown wget wget
  wget https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4-linux-$(uname -m).sh -O cmake-install.sh
  bash cmake-install.sh --skip-licence --exclude-subdir --prefix=/usr/local
fi
if [ ! -x "$(command -v ninja)" ]; then
  temp_install_if_command_unknown unzip unzip
  wget https://github.com/ninja-build/ninja/releases/download/v1.11.1/ninja-linux.zip
  unzip ninja-linux.zip && mv ninja /usr/local/bin/ && rm -rf ninja-linux.zip
fi
if [ "$CC" == "" ] && [ "$CXX" == "" ]; then
  source "$this_file_dir/install_toolchain.sh" -t gcc12
fi

llvm_dir="$LLVM_INSTALL_PREFIX/lib/cmake/llvm"
if [ ! -d "$llvm_dir" ]; then
  echo "Could not find llvm libraries."

  if [ ! -d "$PYBIND11_INSTALL_PREFIX" ]; then
    echo "Building PyBind11..."
    repo_root="$(git rev-parse --show-toplevel)" && cd "$repo_root"
    git submodule update --init --recursive --recommend-shallow --single-branch tpls/pybind11 && cd -
    mkdir "$repo_root/tpls/pybind11/build" && cd "$repo_root/tpls/pybind11/build"
    cmake -G Ninja ../ -DCMAKE_INSTALL_PREFIX="$PYBIND11_INSTALL_PREFIX"
    cmake --build . --target install --config Release && cd -
  fi

  # Build llvm libraries from source and install them in the install directory
  set +e && source "$this_file_dir/build_llvm.sh" -v && set -e
  (return 0 2>/dev/null) && is_sourced=true || is_sourced=false

  if [ ! -d "$llvm_dir" ]; then
    echo "Failed to find directory $llvm_dir."
    if $is_sourced; then return 1; else exit 1; fi
  fi
else 
  echo "Configured C compiler: $CC"
  echo "Configured C++ compiler: $CXX"
fi

if [ ! -f "$BLAS_INSTALL_PREFIX/libblas.a" ] && [ ! -f "$BLAS_INSTALL_PREFIX/lib/libblas.a" ]; then
  if [ -x "$(command -v apt-get)" ]; then
    apt-get update
  fi

  temp_install_if_command_unknown wget wget
  temp_install_if_command_unknown make make
  if [ ! -x "$(command -v "$FC")" ]; then
    temp_install_if_command_unknown gcc gcc
    temp_install_if_command_unknown g++ g++
    temp_install_if_command_unknown gfortran gfortran
  fi

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
