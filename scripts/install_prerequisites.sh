#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Usage: 
# This script builds and installs a minimal set of dependencies needed to build 
# CUDA-Q from source. 
#
# Usage: 
# bash install_prerequisites.sh
#
# For the libraries LLVM, BLAS, ZLIB, OPENSSL, CURL, CUQUANTUM, CUTENSOR, if the
# library is not found in the location defined by the corresponding environment variable
# *_INSTALL_PREFIX, it will be built from source and installed in that location.
# If the LLVM libraries are built from source, the environment variable LLVM_PROJECTS
# can be used to customize which projects are built, and pybind11 will be built and 
# installed in the location defined by PYBIND11_INSTALL_PREFIX if necessary.
# The cuQuantum and cuTensor libraries are only installed if a suitable CUDA compiler 
# is installed. 
# 
# By default, all prerequisites outlined above are installed even if the
# corresponding *_INSTALL_PREFIX is not defined. The command line flag -m changes
# that behavior to only install the libraries for which this variable is defined.
# A compiler toolchain, cmake, and ninja will be installed unless the the -m flag 
# is passed or the corresponding commands already exist. If the commands already 
# exist, compatibility or versions won't be validated.

# Process command line arguments
toolchain=''
exclude_prereq=''
install_all=true
__optind__=$OPTIND
OPTIND=1
while getopts ":e:t:ml:-:" opt; do
  case $opt in
    e) exclude_prereq="$(echo "$OPTARG" | tr '[:upper:]' '[:lower:]')"
    ;;
    t) toolchain="$OPTARG"
    ;;
    m) install_all=false
    ;;
    :) echo "Option -$OPTARG requires an argument."
    (return 0 2>/dev/null) && return 1 || exit 1
    ;;
    \?) echo "Invalid command line option -$OPTARG" >&2
    (return 0 2>/dev/null) && return 1 || exit 1
    ;;
  esac
done
OPTIND=$__optind__

# Set default install prefix environment variables (only when install_all is true)
if $install_all; then
  source "$(dirname "${BASH_SOURCE[0]}")/set_env_defaults.sh"
fi

# Create a temporary directory for building source packages
PREREQS_BUILD_DIR=$(mktemp -d)
echo "Building prerequisites in $PREREQS_BUILD_DIR"
# Remove below if you wish to debug pre-req build failures
trap "rm -rf $PREREQS_BUILD_DIR" EXIT

function temp_install_if_command_unknown {
  if [ ! -x "$(command -v $1)" ]; then
    if [ -x "$(command -v apt-get)" ]; then
      if [ -z "$PKG_UNINSTALL" ]; then apt-get update; fi
      apt-get install -y --no-install-recommends $2
    elif [ -x "$(command -v dnf)" ]; then
      dnf install -y --nobest --setopt=install_weak_deps=False $2
    elif [ -x "$(command -v brew)" ]; then
      HOMEBREW_NO_AUTO_UPDATE=1 brew install $2
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
    elif [ -x "$(command -v brew)" ]; then
      brew uninstall --force $PKG_UNINSTALL 
    else
      echo "No package manager configured for clean up." >&2
    fi
    unset PKG_UNINSTALL
  fi
}

working_dir=`pwd`
read __errexit__ < <(echo $SHELLOPTS | grep -Eo '(^|:)errexit(:|$)' || echo)
function prepare_exit {
  cd "$working_dir" && remove_temp_installs
  if [ -z "$__errexit__" ]; then set +e; fi
}

set -e
trap 'prepare_exit && ((return 0 2>/dev/null) && return 1 || exit 1)' EXIT
this_file_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# [Toolchain] CMake, ninja and C/C++ compiler
if $install_all && [ -z "$(echo $exclude_prereq | grep toolchain)" ]; then
  if [ -n "$toolchain" ] || [ ! -x "$(command -v "$CC")" ] || [ ! -x "$(command -v "$CXX")" ]; then
    echo "Installing toolchain ${toolchain}..."
    if [ "$toolchain" = "llvm" ] && [ ! -d "$LLVM_STAGE1_BUILD" ]; then
      llvm_stage1_tmpdir="$(mktemp -d)"
      LLVM_STAGE1_BUILD="$llvm_stage1_tmpdir/llvm"
      echo "Installing LLVM stage-1 build in $LLVM_STAGE1_BUILD."
    fi

    # Note that when we first build the compiler/runtime built here we need to make sure it is
    # the same version as CUDA Quantum depends on, even if we rebuild the runtime libraries later,
    # since otherwise we need to rebuild zlib.
    # On macOS, use Apple's system clang to avoid header conflicts with macOS SDK
    if [ "$(uname)" = "Darwin" ]; then
      export CC=clang
      export CXX=clang++
      echo "Using Apple Clang: $(clang --version | head -1)"
    else
      LLVM_INSTALL_PREFIX="$LLVM_STAGE1_BUILD" LLVM_BUILD_FOLDER="stage1_build" \
      source "$this_file_dir/install_toolchain.sh" -t ${toolchain:-gcc12}
    fi
  fi
  if [ ! -x "$(command -v cmake)" ]; then
    echo "Installing CMake..."
    temp_install_if_command_unknown wget wget
    pushd "$PREREQS_BUILD_DIR"
    if [ "$(uname)" = "Darwin" ]; then
      cmake_arch="$(uname -m)"
      wget "https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4-macos-universal.tar.gz" -O cmake.tar.gz
      tar -xzf cmake.tar.gz
      mv cmake-3.26.4-macos-universal/CMake.app/Contents/bin/* /usr/local/bin/
      mv cmake-3.26.4-macos-universal/CMake.app/Contents/share/* /usr/local/share/
    else
      wget https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4-linux-$(uname -m).sh -O cmake-install.sh
      bash cmake-install.sh --skip-licence --exclude-subdir --prefix=/usr/local
    fi
    popd
  fi
  if [ ! -x "$(command -v ninja)" ]; then
    echo "Installing Ninja..."
    temp_install_if_command_unknown wget wget
    temp_install_if_command_unknown make make

    pushd "$PREREQS_BUILD_DIR"

    # The pre-built binary for Linux on GitHub is built for x86_64 only,
    # see also https://github.com/ninja-build/ninja/issues/2284.
    wget https://github.com/ninja-build/ninja/archive/refs/tags/v1.11.1.tar.gz
    tar -xzvf v1.11.1.tar.gz && cd ninja-1.11.1
    if [ "$(uname)" = "Darwin" ]; then
      cmake -B build
    else
      LDFLAGS="-static-libstdc++" cmake -B build
    fi
    cmake --build build
    mv build/ninja /usr/local/bin/

    popd
  fi
fi

# [Zlib] Needed to build LLVM with zlib support (used by linker)
# [Minizip] Needed by rest_server for archive handling
# Build both from source for consistency across platforms.
if [ -n "$ZLIB_INSTALL_PREFIX" ] && [ -z "$(echo $exclude_prereq | grep zlib)" ]; then
  if [ ! -f "$ZLIB_INSTALL_PREFIX/lib/libz.a" ] || [ ! -f "$ZLIB_INSTALL_PREFIX/lib/libminizip.a" ]; then
    echo "Installing libz and minizip..."
    temp_install_if_command_unknown wget wget
    temp_install_if_command_unknown make make
    temp_install_if_command_unknown automake automake
    temp_install_if_command_unknown libtool libtool

    pushd "$PREREQS_BUILD_DIR"

    wget -O zlib-1.3.1.tar.gz https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz
    tar -xzf zlib-1.3.1.tar.gz && cd zlib-1.3.1
    CC="$CC" CFLAGS="-fPIC" \
    ./configure --prefix="$ZLIB_INSTALL_PREFIX" --static
    make CC="$CC" && make install
    cd contrib/minizip
    autoreconf --install
    CC="$CC" CFLAGS="-fPIC" \
    ./configure --prefix="$ZLIB_INSTALL_PREFIX" --disable-shared
    make CC="$CC" && make install

    popd
    remove_temp_installs
  else
    echo "libz and minizip already installed in $ZLIB_INSTALL_PREFIX."
  fi
fi

# [LLVM/MLIR] Needed to build the CUDA Quantum toolchain
if [ -n "$LLVM_INSTALL_PREFIX" ] && [ -z "$(echo $exclude_prereq | grep llvm)" ]; then
  if [ ! -d "$LLVM_INSTALL_PREFIX/lib/cmake/llvm" ]; then
    echo "Installing LLVM libraries..."
    LLVM_INSTALL_PREFIX="$LLVM_INSTALL_PREFIX" \
    LLVM_PROJECTS="$LLVM_PROJECTS" \
    PYBIND11_INSTALL_PREFIX="$PYBIND11_INSTALL_PREFIX" \
    Python3_EXECUTABLE="$Python3_EXECUTABLE" \
    bash "$this_file_dir/build_llvm.sh" -v
  else 
    echo "LLVM already installed in $LLVM_INSTALL_PREFIX."
  fi

  if [ "$toolchain" = "llvm" ]; then
    #rm -rf "$llvm_stage1_tmpdir"
    export CC="$LLVM_INSTALL_PREFIX/bin/clang" 
    export CXX="$LLVM_INSTALL_PREFIX/bin/clang++"
    export FC="$LLVM_INSTALL_PREFIX/bin/flang-new"
    echo "Configured C compiler: $CC"
    echo "Configured C++ compiler: $CXX"
    echo "Configured Fortran compiler: $FC"
  fi
fi

# [Blas] Needed for certain optimizers
if [ -n "$BLAS_INSTALL_PREFIX" ] && [ -z "$(echo $exclude_prereq | grep blas)" ]; then
  if [ ! -f "$BLAS_INSTALL_PREFIX/libblas.a" ] && [ ! -f "$BLAS_INSTALL_PREFIX/lib/libblas.a" ]; then
    echo "Installing BLAS..."
    temp_install_if_command_unknown wget wget
    temp_install_if_command_unknown make make
    if [ ! -x "$(command -v "$FC")" ]; then
      unset FC
      temp_install_if_command_unknown gfortran gfortran
    fi

    pushd "$PREREQS_BUILD_DIR"

    # See also: https://github.com/NVIDIA/cuda-quantum/issues/452
    wget http://www.netlib.org/blas/blas-3.11.0.tgz
    tar -xzvf blas-3.11.0.tgz && cd BLAS-3.11.0
    make FC="${FC:-gfortran}"
    mkdir -p "$BLAS_INSTALL_PREFIX"
    mv blas_*.a "$BLAS_INSTALL_PREFIX/libblas.a"

    popd
    remove_temp_installs
  else
    echo "BLAS already installed in $BLAS_INSTALL_PREFIX."
  fi
fi

# [OpenSSL] Needed for communication with external services
if [ -n "$OPENSSL_INSTALL_PREFIX" ] && [ -z "$(echo $exclude_prereq | grep ssl)" ]; then
  if [ ! -d "$OPENSSL_INSTALL_PREFIX" ] || \
     [ -z "$(find "$OPENSSL_INSTALL_PREFIX" -name 'libssl.a' 2>/dev/null)" ]; then
    echo "Installing OpenSSL..."
    temp_install_if_command_unknown wget wget
    temp_install_if_command_unknown make make

    pushd "$PREREQS_BUILD_DIR"

    # Not all perl installations include all necessary modules.
    # To facilitate a consistent build across platforms and to minimize dependencies,
    # we just use our own perl version for the OpenSSL build.
    wget https://www.cpan.org/src/5.0/perl-5.38.2.tar.gz
    tar -xzf perl-5.38.2.tar.gz && cd perl-5.38.2
    ./Configure -des -Dcc="$CC" -Dprefix="$PREREQS_BUILD_DIR/perl5"
    make CC="$CC" && make install
    cd ..
    # Additional perl modules can be installed with cpan, e.g.
    # PERL_MM_USE_DEFAULT=1 $PREREQS_BUILD_DIR/perl5/bin/cpan App::cpanminus

    if [ ! -x "$(command -v ar)" ]; then
      cc_exe_dir=`dirname "$CC"`
      if [ -x "$(command -v "$cc_exe_dir/ar")" ]; then AR="$cc_exe_dir/ar"
      else AR=`find "$cc_exe_dir" -maxdepth 1 -name 'llvm-ar'`
      fi
    fi

    wget https://www.openssl.org/source/openssl-3.5.1.tar.gz
    tar -xf openssl-3.5.1.tar.gz && cd openssl-3.5.1
    CC="$CC" CFLAGS="-fPIC" CXX="$CXX" CXXFLAGS="-fPIC" AR="${AR:-ar}" \
    "$PREREQS_BUILD_DIR/perl5/bin/perl" Configure no-shared \
      --prefix="$OPENSSL_INSTALL_PREFIX" zlib \
      --with-zlib-include="$ZLIB_INSTALL_PREFIX/include" \
      --with-zlib-lib="$ZLIB_INSTALL_PREFIX/lib"
    make CC="$CC" CXX="$CXX" && make install

    popd
    remove_temp_installs
  else
    echo "OpenSSL already installed in $OPENSSL_INSTALL_PREFIX."
  fi
fi

# [CURL] Needed for communication with external services
if [ -n "$CURL_INSTALL_PREFIX" ] && [ -z "$(echo $exclude_prereq | grep curl)" ]; then
  if [ ! -f "$CURL_INSTALL_PREFIX/lib/libcurl.a" ]; then
    echo "Installing Curl..."
    temp_install_if_command_unknown wget wget
    temp_install_if_command_unknown make make

    pushd "$PREREQS_BUILD_DIR"

    # The arguments --with-ca-path and --with-ca-bundle can be used to configure the default
    # locations where Curl will look for certificates. Note that the paths where certificates
    # are stored by default varies across operating systems, and to build a Curl library that
    # can run out of the box on various operating systems pretty much necessitates including
    # and distributing a certificate bundle, or downloading such a bundle dynamically at
    # at runtime if needed. The Mozilla certificate bundle can be
    # downloaded from https://curl.se/ca/cacert.pem. For more information, see
    # - https://curl.se/docs/sslcerts.html
    # - https://curl.se/docs/caextract.html
    wget https://curl.se/ca/cacert.pem
    wget https://curl.se/ca/cacert.pem.sha256
    if [ -x "$(command -v sha256sum)" ]; then
      computed_sha256="$(sha256sum cacert.pem)"
    else
      computed_sha256="$(shasum -a 256 cacert.pem)"
    fi
    if [ "$computed_sha256" != "$(cat cacert.pem.sha256)" ]; then
      echo -e "\e[01;31mWarning: Incorrect sha256sum of cacert.pem. The file cacert.pem has been removed. The file can be downloaded manually from https://curl.se/docs/sslcerts.html.\e[0m" >&2
    else
      mkdir -p "$CURL_INSTALL_PREFIX" && mv cacert.pem "$CURL_INSTALL_PREFIX"
    fi

    # Unfortunately, it looks like the default paths need to be absolute and known at compile time.
    # Note that while the environment variable CURL_CA_BUNDLE allows to easily override the default
    # path when invoking the Curl executable, this variable is *not* respected by default by the
    # built library itself; instead, the user of libcurl is responsible for picking up the
    # environment variables and passing them to curl via CURLOPT_CAINFO and CURLOPT_PROXY_CAINFO.
    # We opt to build Curl without any default paths, and instead have the CUDA-Q runtime
    # determine and pass a suitable path.
    #
    # Build curl with CMake to generate proper CMake config files (CURLConfig.cmake).
    # This allows CMake's find_package(CURL) to use config mode, which correctly encodes
    # full paths to dependencies (OpenSSL, zlib) and avoids pkg-config issues where
    # -lssl/-lcrypto resolve to the wrong system libraries on macOS.
    wget https://github.com/curl/curl/releases/download/curl-8_5_0/curl-8.5.0.tar.gz
    tar -xzvf curl-8.5.0.tar.gz && cd curl-8.5.0
    cmake -G Ninja -B build \
      -DCMAKE_C_COMPILER="$CC" \
      -DCMAKE_C_FLAGS="-fPIC" \
      -DCMAKE_INSTALL_PREFIX="$CURL_INSTALL_PREFIX" \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=OFF \
      -DBUILD_CURL_EXE=ON \
      -DCURL_USE_OPENSSL=ON \
      -DOPENSSL_ROOT_DIR="$OPENSSL_INSTALL_PREFIX" \
      -DCURL_ZLIB=ON \
      -DZLIB_ROOT="$ZLIB_INSTALL_PREFIX" \
      -DCURL_CA_BUNDLE=none \
      -DCURL_CA_PATH=none \
      -DCURL_USE_LIBSSH2=OFF \
      -DCURL_USE_LIBPSL=OFF \
      -DUSE_LIBIDN2=OFF \
      -DCURL_BROTLI=OFF \
      -DCURL_ZSTD=OFF \
      -DUSE_NGHTTP2=OFF \
      -DENABLE_ARES=OFF \
      -DCURL_DISABLE_FTP=ON \
      -DCURL_DISABLE_TFTP=ON \
      -DCURL_DISABLE_SMTP=ON \
      -DCURL_DISABLE_LDAP=ON \
      -DCURL_DISABLE_LDAPS=ON \
      -DCURL_DISABLE_SMB=ON \
      -DCURL_DISABLE_GOPHER=ON \
      -DCURL_DISABLE_TELNET=ON \
      -DCURL_DISABLE_RTSP=ON \
      -DCURL_DISABLE_POP3=ON \
      -DCURL_DISABLE_IMAP=ON \
      -DCURL_DISABLE_FILE=ON \
      -DCURL_DISABLE_DICT=ON
    cmake --build build --config Release
    cmake --install build --config Release

    popd
    remove_temp_installs
  else
    echo "Curl already installed in $CURL_INSTALL_PREFIX."
  fi
fi

# [AWS SDK] Needed for communication with Braket
if [ -n "$AWS_INSTALL_PREFIX" ] && [ -z "$(echo $exclude_prereq | grep aws)" ]; then
  if [ ! -d "$AWS_INSTALL_PREFIX" ] || [ -z "$(ls -A "$AWS_INSTALL_PREFIX"/* 2> /dev/null)" ]; then
    pushd "$PREREQS_BUILD_DIR"

    aws_service_components='braket s3-crt sts'
    git clone --filter=tree:0 https://github.com/aws/aws-sdk-cpp aws-sdk-cpp
    cd aws-sdk-cpp && git checkout 1.11.454 && git submodule update --init --recursive

    # FIXME: CUDAQ VERSION?
    mkdir build && cd build
    cmake -G Ninja .. \
      -DCMAKE_INSTALL_PREFIX="${AWS_INSTALL_PREFIX}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF \
      -DAWS_SDK_WARNINGS_ARE_ERRORS=OFF \
      -DAWS_USER_AGENT_CUSTOMIZATION=CUDA-Q/${CUDA_QUANTUM_VERSION} \
      -DBUILD_ONLY="$(echo $aws_service_components | tr ' ' ';')" \
      -DBUILD_SHARED_LIBS=OFF \
      -DZLIB_ROOT="${ZLIB_INSTALL_PREFIX}" \
      -DZLIB_USE_STATIC_LIBS=ON \
      -DOPENSSL_ROOT_DIR="${OPENSSL_INSTALL_PREFIX}" \
      -DCURL_LIBRARY="${CURL_INSTALL_PREFIX}/lib/libcurl.a" \
      -DCURL_INCLUDE_DIR="${CURL_INSTALL_PREFIX}/include" \
      -Dcrypto_LIBRARY="$(find "$OPENSSL_INSTALL_PREFIX" -name libcrypto.a)" \
      -Dcrypto_INCLUDE_DIR="${OPENSSL_INSTALL_PREFIX}/include" \
      -DENABLE_TESTING=OFF \
      -DAUTORUN_UNIT_TESTS=OFF
    cmake --build . --config=Release
    cmake --install . --config=Release

    popd
    remove_temp_installs
  else
    echo "AWS SDK already installed in $AWS_INSTALL_PREFIX."
  fi
fi

# [cuQuantum and cuTensor] Needed for GPU-accelerated components
cuda_driver=${CUDACXX:-${CUDA_HOME:-/usr/local/cuda}/bin/nvcc}
cuda_version=`"$cuda_driver" --version 2>/dev/null | grep -o 'release [0-9]*\.[0-9]*' | cut -d ' ' -f 2`
if [ -n "$cuda_version" ]; then
  if [ -n "$CUQUANTUM_INSTALL_PREFIX" ] && [ -z "$(echo $exclude_prereq | grep cuquantum)" ]; then
    if [ ! -d "$CUQUANTUM_INSTALL_PREFIX" ] || [ -z "$(ls -A "$CUQUANTUM_INSTALL_PREFIX"/* 2> /dev/null)" ]; then
      echo "Installing cuQuantum..."
      bash "$this_file_dir/configure_build.sh" install-cuquantum
    else 
      echo "cuQuantum already installed in $CUQUANTUM_INSTALL_PREFIX."
    fi
  fi
  if [ -n "$CUTENSOR_INSTALL_PREFIX" ] && [ -z "$(echo $exclude_prereq | grep cutensor)" ]; then
    if [ ! -d "$CUTENSOR_INSTALL_PREFIX" ] || [ -z "$(ls -A "$CUTENSOR_INSTALL_PREFIX"/* 2> /dev/null)" ]; then
      echo "Installing cuTensor..."
      bash "$this_file_dir/configure_build.sh" install-cutensor
    else 
      echo "cuTensor already installed in $CUTENSOR_INSTALL_PREFIX."
    fi
  fi
fi

exclude_prereq="$(echo $exclude_prereq | tr ';' ' ' | sed 's/  */, /g')"
if $install_all; then echo "All prerequisites have been installed (excluded: ${exclude_prereq:-none})."
else echo "Prerequisites for which an *_INSTALL_PREFIX variable was defined have been installed (excluded: ${exclude_prereq:-none})."
fi
# Make sure to call prepare_exit so that we properly uninstalled all helper tools,
# and so that we are in the correct directory also when this script is sourced.
prepare_exit && ((return 0 2>/dev/null) && return 0 || exit 0)

