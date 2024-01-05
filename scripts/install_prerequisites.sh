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
ZLIB_INSTALL_PREFIX=${ZLIB_INSTALL_PREFIX:-/usr/local/zlib}
OPENSSL_INSTALL_PREFIX=${OPENSSL_INSTALL_PREFIX:-/usr/lib/ssl}
CURL_INSTALL_PREFIX=${CURL_INSTALL_PREFIX:-/usr/local/curl}

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
    if [ -x "$(command -v apt-get)" ]; then
      if [ -z "$PKG_UNINSTALL" ]; then apt-get update; fi
      apt-get install -y --no-install-recommends $2
    elif [ -x "$(command -v dnf)" ]; then
      dnf install -y --nobest --setopt=install_weak_deps=False $2
    else
      echo "No package manager was found to install $2." >&2
    fi
    PKG_UNINSTALL="$PKG_UNINSTALL $2"
  fi
}

function remove_temp_installs {
  if [ -n "$PKG_UNINSTALL" ]; then
    echo "Uninstalling packages used for bootstrapping: $PKG_UNINSTALL"
    if [ -x "$(command -v apt-get)" ]; then  
      apt-get remove -y $PKG_UNINSTALL
      apt-get autoremove -y --purge
    elif [ -x "$(command -v dnf)" ]; then
      dnf remove -y $PKG_UNINSTALL
      dnf clean all
    else
      echo "No package manager configured for clean up." >&2
    fi
    unset PKG_UNINSTALL
    create_llvm_symlinks # uninstalling other compiler tools may have removed the symlinks
  fi
}

working_dir=`pwd`
read __errexit__ < <(echo $SHELLOPTS | egrep -o '(^|:)errexit(:|$)' || echo)
function prepare_exit {
  cd "$working_dir" && remove_temp_installs
  if [ -z "$__errexit__" ]; then set +e; fi
}

set -e
trap 'prepare_exit && ((return 0 2>/dev/null) && return 1 || exit 1)' EXIT
this_file_dir=`dirname "$(readlink -f "${BASH_SOURCE[0]}")"`

# [Toolchain] CMake, ninja and C/C++ compiler
if [ ! -x "$(command -v "$CC")" ] || [ ! -x "$(command -v "$CXX")" ]; then
  source "$this_file_dir/install_toolchain.sh" -t gcc12
fi
if [ ! -x "$(command -v cmake)" ]; then
  echo "Installing CMake..."
  temp_install_if_command_unknown wget wget
  wget https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4-linux-$(uname -m).sh -O cmake-install.sh
  bash cmake-install.sh --skip-licence --exclude-subdir --prefix=/usr/local
fi
if [ ! -x "$(command -v ninja)" ]; then
  echo "Installing Ninja..."
  temp_install_if_command_unknown wget wget
  temp_install_if_command_unknown make make

  # The pre-built binary for Linux on GitHub is built for x86_64 only, 
  # see also https://github.com/ninja-build/ninja/issues/2284.
  wget https://github.com/ninja-build/ninja/archive/refs/tags/v1.11.1.tar.gz
  tar -xzvf v1.11.1.tar.gz && cd ninja-1.11.1
  cmake -B build && cmake --build build
  mv build/ninja /usr/local/bin/
  rm -rf v1.11.1.tar.gz ninja-1.11.1
fi

# [Blas] Needed for certain optimizers
if [ ! -f "$BLAS_INSTALL_PREFIX/libblas.a" ] && [ ! -f "$BLAS_INSTALL_PREFIX/lib/libblas.a" ]; then
  echo "Installing BLAS..."
  temp_install_if_command_unknown wget wget
  temp_install_if_command_unknown make make
  if [ ! -x "$(command -v "$FC")" ]; then
    temp_install_if_command_unknown gcc gcc
    temp_install_if_command_unknown g++ g++
    temp_install_if_command_unknown gfortran gfortran
  elif [ ! -x "gfortran" ]; then
    ln -s "$FC" /usr/bin/gfortran
  fi

  # See also: https://github.com/NVIDIA/cuda-quantum/issues/452
  wget http://www.netlib.org/blas/blas-3.11.0.tgz
  tar -xzvf blas-3.11.0.tgz 
  cd BLAS-3.11.0 && make 
  mkdir -p "$BLAS_INSTALL_PREFIX"
  mv blas_LINUX.a "$BLAS_INSTALL_PREFIX/libblas.a"
  cd .. && rm -rf blas-3.11.0.tgz BLAS-3.11.0
  remove_temp_installs
fi

# [Zlib] Needed to build LLVM with zlib support (used by linker)
if [ ! -f "$ZLIB_INSTALL_PREFIX/lib/libz.a" ]; then
  echo "Installing libz..."
  temp_install_if_command_unknown wget wget
  temp_install_if_command_unknown make make

  wget https://github.com/madler/zlib/releases/download/v1.3/zlib-1.3.tar.gz
  tar -xzvf zlib-1.3.tar.gz && cd zlib-1.3
  CFLAGS="-fPIC" CXXFLAGS="-fPIC" \
  ./configure --prefix="$ZLIB_INSTALL_PREFIX" --static
  make && make install
  cd .. && rm -rf zlib-1.3.tar.gz zlib-1.3
  remove_temp_installs
fi

# [OpenSSL] Needed for communication with external services
if [ ! -d "$OPENSSL_INSTALL_PREFIX" ] || [ -z "$(find "$OPENSSL_INSTALL_PREFIX" -name libssl.a)" ]; then
  echo "Installing OpenSSL..."
  temp_install_if_command_unknown wget wget
  temp_install_if_command_unknown make make

  # Not all perl installations include all necessary modules.
  # To facilitate a consistent build across platforms and to minimize dependencies, 
  # we just use our own perl version for the OpenSSL build.
  wget https://www.cpan.org/src/5.0/perl-5.38.2.tar.gz
  tar -xzf perl-5.38.2.tar.gz && cd perl-5.38.2
  ./Configure -des -Dcc="$CC" -Dprefix=~/.perl5
  make && make install
  cd .. && rm -rf perl-5.38.2.tar.gz perl-5.38.2
  # Additional perl modules can be installed with cpan, e.g.
  # PERL_MM_USE_DEFAULT=1 ~/.perl5/bin/cpan App::cpanminus

  wget https://www.openssl.org/source/openssl-3.1.1.tar.gz
  tar -xf openssl-3.1.1.tar.gz && cd openssl-3.1.1
  CFLAGS="-fPIC" CXXFLAGS="-fPIC" \
  ~/.perl5/bin/perl Configure no-shared no-zlib --prefix="$OPENSSL_INSTALL_PREFIX"
  make && make install
  cd .. && rm -rf openssl-3.1.1.tar.gz openssl-3.1.1 ~/.perl5
  remove_temp_installs
fi

# [CURL] Needed for communication with external services
if [ ! -f "$CURL_INSTALL_PREFIX/lib/libcurl.a" ]; then
  echo "Installing Curl..."
  temp_install_if_command_unknown wget wget
  temp_install_if_command_unknown make make

  wget https://github.com/curl/curl/releases/download/curl-8_5_0/curl-8.5.0.tar.gz
  tar -xzvf curl-8.5.0.tar.gz && cd curl-8.5.0
  wget https://curl.haxx.se/ca/cacert.pem
  CFLAGS="-fPIC" CXXFLAGS="-fPIC" LDFLAGS="-L$OPENSSL_INSTALL_PREFIX/lib64 $LDFLAGS" \
  ./configure --prefix="$CURL_INSTALL_PREFIX" \
    --enable-shared=no --enable-static=yes \
    --with-openssl="$OPENSSL_INSTALL_PREFIX" --with-zlib="$ZLIB_INSTALL_PREFIX" \
    --with-ca-bundle=cacert.pem \
    --without-zstd --without-brotli \
    --disable-ftp --disable-tftp --disable-smtp --disable-ldap --disable-ldaps \
    --disable-smb --disable-gopher --disable-telnet --disable-rtsp \
    --disable-pop3 --disable-imap --disable-file  --disable-dict \
    --disable-versioned-symbols --disable-manual
  make && make install
  cd .. && rm -rf curl-8.5.0.tar.gz curl-8.5.0
  remove_temp_installs
fi

# [LLVM/MLIR] Needed to build the CUDA Quantum toolchain
llvm_dir="$LLVM_INSTALL_PREFIX/lib/cmake/llvm"
if [ ! -d "$llvm_dir" ]; then
  echo "Installing LLVM libraries..."
  bash "$this_file_dir/build_llvm.sh" -v
else 
  echo "Configured C compiler: $CC"
  echo "Configured C++ compiler: $CXX"
fi

echo "All prerequisites have been installed."
# Make sure to call prepare_exit so that we properly uninstalled all helper tools,
# and so that we are in the correct directory also when this script is sourced.
prepare_exit && ((return 0 2>/dev/null) && return 0 || exit 0)

