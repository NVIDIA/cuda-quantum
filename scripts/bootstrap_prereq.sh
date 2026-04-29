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
#   -e <name>     Exclude a prerequisite (e.g. zlib, llvm, blas, ssl, curl, aws, cuquantum, cutensor, toolchain)
#   -t <name>     Select toolchain (e.g. gcc12, llvm)
#   -m            Only install libraries for which an *_INSTALL_PREFIX is defined
#   -l            Generate a prerequisites lock file and exit (no installation)
#
# When the -l flag is used, a lock file named cudaq_prereqs.lock (or the path
# given via the PREREQS_LOCK_FILE environment variable) is generated that
# enumerates the source locations for all prerequisites that would be installed
# for the current configuration. This can be used to pre-download sources in
# controlled build environments.
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

# Centralized version / source definitions used by both installation and lockfile
# generation. Keeping these here avoids duplication between code paths.
CMAKE_VERSION=3.26.4
CMAKE_MACOS_TARBALL_URL="https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-macos-universal.tar.gz"
CMAKE_LINUX_INSTALLER_URL_BASE="https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-"

NINJA_VERSION=1.11.1
NINJA_TARBALL_URL="https://github.com/ninja-build/ninja/archive/refs/tags/v${NINJA_VERSION}.tar.gz"

ZLIB_VERSION=1.3.1
ZLIB_TARBALL_URL="https://github.com/madler/zlib/releases/download/v${ZLIB_VERSION}/zlib-${ZLIB_VERSION}.tar.gz"

BLAS_VERSION=3.11.0
BLAS_TARBALL_URL="http://www.netlib.org/blas/blas-${BLAS_VERSION}.tgz"

PERL_VERSION=5.38.2
PERL_TARBALL_URL="https://www.cpan.org/src/5.0/perl-${PERL_VERSION}.tar.gz"

OPENSSL_VERSION=3.5.1
OPENSSL_TARBALL_URL="https://www.openssl.org/source/openssl-${OPENSSL_VERSION}.tar.gz"

CURL_VERSION=8.5.0
CURL_VERSION_UNDERSCORE=curl-8_5_0
CURL_TARBALL_URL="https://github.com/curl/curl/releases/download/${CURL_VERSION_UNDERSCORE}/curl-${CURL_VERSION}.tar.gz"
CACERT_URL="https://curl.se/ca/cacert.pem"
CACERT_SHA256_URL="${CACERT_URL}.sha256"

AWS_SDK_CPP_URL="https://github.com/aws/aws-sdk-cpp"
AWS_SDK_CPP_REF="1.11.454"

# QRMI pre-built C artifacts for Pasqal QRMI connector
QRMI_RELEASE_REPO=${QRMI_RELEASE_REPO:-qiskit-community/qrmi}
QRMI_RELEASE_TAG=${QRMI_RELEASE_TAG:-v0.12.0}
QRMI_RELEASE_VERSION=${QRMI_RELEASE_TAG#v}
QRMI_RELEASE_BASE="https://github.com/${QRMI_RELEASE_REPO}/releases/download/${QRMI_RELEASE_TAG}"
QRMI_ARCHIVE="libqrmi-${QRMI_RELEASE_VERSION}-el8-x86_64.tar.gz"
QRMI_UNPACK_DIR="libqrmi-${QRMI_RELEASE_VERSION}"
# NOTE: This needs to be updated whenever the pre-built artifacts are updated. The SHA-256 can be computed with:
#   wget -O qrmi.tar.gz "${QRMI_RELEASE_BASE}/${QRMI_ARCHIVE}"
#   sha256sum qrmi.tar.gz | awk '{print $1}'
QRMI_ARCHIVE_SHA256=${QRMI_ARCHIVE_SHA256:-2986150d4f55e1f6566bef16d9fb3897ca04dd7eaa681865f7ef244f298a6746}

# Process command line arguments
toolchain=''
exclude_prereq=''
install_all=true
lock_mode=false
__optind__=$OPTIND
OPTIND=1
while getopts ":e:t:ml-:" opt; do
  case $opt in
    e) exclude_prereq="$(echo "$OPTARG" | tr '[:upper:]' '[:lower:]')"
    ;;
    t) toolchain="$OPTARG"
    ;;
    m) install_all=false
    ;;
    l) lock_mode=true
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

# If requested, generate a lock file describing all source archives / repositories
# that would be used to build the prerequisites, then exit without installing.
if $lock_mode; then
  LOCK_FILE="${PREREQS_LOCK_FILE:-cudaq_prereqs.lock}"

  # Helper to append one entry to the lock file in a simple key=value format.
  function add_lock_line {
    local name="$1"; shift
    echo "name=${name} $*" >> "$LOCK_FILE"
  }

  # Initialize / truncate the lock file and add a short header.
  {
    echo "# CUDA-Q prerequisites lockfile"
    echo "# Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "# Format: name=<id> key=value ..."
  } > "$LOCK_FILE"

  # [Toolchain] CMake and Ninja sources (compiler toolchain itself is handled
  # via install_toolchain.sh or the system toolchain and is not pinned here).
  # In lockfile mode, always list the toolchain sources regardless of what is
  # currently installed or excluded.
  add_lock_line "cmake-macos" \
    "type=tar" \
    "url=${CMAKE_MACOS_TARBALL_URL}" \
    "version=${CMAKE_VERSION}"
  add_lock_line "cmake" \
    "type=sh" \
    "url=${CMAKE_LINUX_INSTALLER_URL_BASE}$(uname -m).sh" \
    "version=${CMAKE_VERSION}"
  add_lock_line "ninja" \
    "type=tar" \
    "url=${NINJA_TARBALL_URL}" \
    "version=${NINJA_VERSION}"

  # [Zlib / Minizip]
  add_lock_line "zlib" \
    "type=tar" \
    "url=${ZLIB_TARBALL_URL}" \
    "version=${ZLIB_VERSION}"

  # [BLAS]
  add_lock_line "blas" \
    "type=tar" \
    "url=${BLAS_TARBALL_URL}" \
    "version=${BLAS_VERSION}"

  # [OpenSSL] (and its private Perl used only for the build)
  add_lock_line "perl" \
    "type=tar" \
    "url=${PERL_TARBALL_URL}" \
    "version=${PERL_VERSION}"
  add_lock_line "openssl" \
    "type=tar" \
    "url=${OPENSSL_TARBALL_URL}" \
    "version=${OPENSSL_VERSION}"

  # [CURL] (including CA bundle)
  add_lock_line "cacert" \
    "type=pem" \
    "url=${CACERT_URL}"
  add_lock_line "curl" \
    "type=tar" \
    "url=${CURL_TARBALL_URL}" \
    "version=${CURL_VERSION}"

  # [AWS SDK]
  add_lock_line "aws-sdk-cpp" \
    "type=git" \
    "url=${AWS_SDK_CPP_URL}" \
    "ref=${AWS_SDK_CPP_REF}"

  # [QRMI] Pre-built C artifacts for Pasqal QRMI connector
  # Keep this in sync with the QRMI section in the installation path below.
  add_lock_line "qrmi" \
    "type=tar" \
    "url=${QRMI_RELEASE_BASE}/${QRMI_ARCHIVE}" \
    "version=${QRMI_RELEASE_VERSION}" \
    "sha256=${QRMI_ARCHIVE_SHA256}"

  echo "Prerequisites lockfile written to ${LOCK_FILE}."
  (return 0 2>/dev/null) && return 0 || exit 0
fi

# Create a temporary directory for building source packages
PREREQS_BUILD_DIR=$(mktemp -d)
: "${PREREQS_BUILD_DIR:?ERROR mktemp failed}"
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

if [ "$(uname)" = "Darwin" ] && [ -x "$(command -v xcrun)" ]; then
  export SDKROOT="${SDKROOT:-$(xcrun --show-sdk-path)}"
fi

# [Toolchain] CMake, ninja and C/C++ compiler
if $install_all && [ -z "$(echo $exclude_prereq | grep toolchain)" ]; then
    if [ "$(uname)" = "Darwin" ]; then
      export CC=clang
      export CXX=clang++
      echo "Using Apple Clang: $(clang --version | head -1)"
    else
      export LLVM_STAGE1_BUILD="$LLVM_INSTALL_PREFIX/bootstrap"
      if [ ! -x "$LLVM_STAGE1_BUILD/bin/clang" ]; then
        temp_install_if_command_unknown cmake cmake
        temp_install_if_command_unknown ninja ninja-build
        echo "Building stage1 LLVM (clang;lld;runtimes)..."
        LLVM_INSTALL_PREFIX="$LLVM_STAGE1_BUILD" \
        LLVM_PROJECTS='clang;lld;runtimes' \
        LLVM_ENABLE_ZLIB=OFF \
        LLVM_BUILD_FOLDER=bootstrap_build \
        LLVM_EXTRA_CMAKE_ARGS='-DLIBCXX_USE_COMPILER_RT=YES -DLIBCXXABI_USE_COMPILER_RT=YES -DLIBUNWIND_USE_COMPILER_RT=YES' \
        bash "$this_file_dir/build_llvm.sh" -v
        remove_temp_installs
      fi
      export CC="$LLVM_STAGE1_BUILD/bin/clang"
      export CXX="$LLVM_STAGE1_BUILD/bin/clang++"
      stage1_libdirs="$LLVM_STAGE1_BUILD/lib$(ls -d "$LLVM_STAGE1_BUILD/lib/"*linux* 2>/dev/null | sed 's/^/:/')"
      export LD_LIBRARY_PATH="${stage1_libdirs}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
      echo "Using stage1 clang: $CC"
    fi
  if [ ! -x "$(command -v cmake)" ]; then
    echo "Installing CMake..."
    temp_install_if_command_unknown wget wget
    pushd "$PREREQS_BUILD_DIR"
    if [ "$(uname)" = "Darwin" ]; then
      cmake_arch="$(uname -m)"
      wget "${CMAKE_MACOS_TARBALL_URL}" -O cmake.tar.gz
      tar -xzf cmake.tar.gz
      mv "cmake-${CMAKE_VERSION}-macos-universal/CMake.app/Contents/bin/"* "$HOME/.local/bin/"
      mv "cmake-${CMAKE_VERSION}-macos-universal/CMake.app/Contents/share/"* "$HOME/.local/share/"
    else
      wget "${CMAKE_LINUX_INSTALLER_URL_BASE}$(uname -m).sh" -O cmake-install.sh
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
    wget "${NINJA_TARBALL_URL}"
    tar -xzvf "v${NINJA_VERSION}.tar.gz" && cd "ninja-${NINJA_VERSION}"
    if [ "$(uname)" = "Darwin" ]; then
      cmake -B build
    else
      lld="${LLVM_STAGE1_BUILD:+$LLVM_STAGE1_BUILD/bin/ld.lld}"
      if [ -x "$lld" ]; then
        cmake -B build -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=$lld"
      else
        LDFLAGS="-static-libstdc++" cmake -B build
      fi
    fi
    cmake --build build

    if [ "$(uname)" = "Darwin" ]; then
      mv build/ninja $HOME/.local/bin/
    else
      mv build/ninja /usr/local/bin/
    fi

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
    # On macOS, Apple ships /usr/bin/libtool which is NOT GNU libtool.
    # Homebrew installs GNU libtool as 'glibtool' to avoid conflicts.
    # We need glibtoolize for autoreconf, so check for that instead.
    if [ "$(uname)" = "Darwin" ]; then
      temp_install_if_command_unknown glibtoolize libtool
    else
      temp_install_if_command_unknown libtool libtool
    fi

    pushd "$PREREQS_BUILD_DIR"

    wget -O "zlib-${ZLIB_VERSION}.tar.gz" "${ZLIB_TARBALL_URL}"
    tar -xzf "zlib-${ZLIB_VERSION}.tar.gz" && cd "zlib-${ZLIB_VERSION}"
    CC="$CC" CFLAGS="-fPIC" \
    ./configure --prefix="$ZLIB_INSTALL_PREFIX" --static
    make CC="$CC" && make install
    cd contrib/minizip
    # On macOS with Homebrew, set up environment for autoreconf:
    # - Add Homebrew's m4 macros to aclocal search path
    # - Point LIBTOOLIZE to glibtoolize (Homebrew's GNU libtoolize)
    if [ "$(uname)" = "Darwin" ] && [ -x "$(command -v brew)" ]; then
      export ACLOCAL_PATH="$(brew --prefix)/share/aclocal${ACLOCAL_PATH:+:$ACLOCAL_PATH}"
      export LIBTOOLIZE=glibtoolize
    fi
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

# [nanobind] Needed for MLIR Python bindings (MLIR 22+)
# Install nanobind independently of the LLVM build so that it is available
# even when LLVM is restored from cache.
if [ -n "$NANOBIND_INSTALL_PREFIX" ]; then
  if [ ! -d "$NANOBIND_INSTALL_PREFIX" ] || [ -z "$(ls -A "$NANOBIND_INSTALL_PREFIX"/* 2> /dev/null)" ]; then
    echo "Building nanobind..."
    cd "$this_file_dir" && cd $(git rev-parse --show-toplevel)
    git submodule update --init --recursive --recommend-shallow --single-branch tpls/nanobind
    mkdir -p "tpls/nanobind/build" && cd "tpls/nanobind/build"
    cmake -G Ninja ../ -DCMAKE_INSTALL_PREFIX="$NANOBIND_INSTALL_PREFIX" -DNB_TEST=False
    cmake --build . --target install --config Release
    cd "$working_dir"
  else
    echo "nanobind already installed in $NANOBIND_INSTALL_PREFIX."
  fi
fi

# [LLVM/MLIR] Needed to build the CUDA Quantum toolchain
if [ -n "$LLVM_INSTALL_PREFIX" ] && [ -z "$(echo $exclude_prereq | grep llvm)" ]; then
  if [ ! -d "$LLVM_INSTALL_PREFIX/lib/cmake/llvm" ]; then
    echo "Installing LLVM libraries..."
    LLVM_INSTALL_PREFIX="$LLVM_INSTALL_PREFIX" \
    LLVM_PROJECTS="$LLVM_PROJECTS" \
    PYBIND11_INSTALL_PREFIX="$PYBIND11_INSTALL_PREFIX" \
    NANOBIND_INSTALL_PREFIX="$NANOBIND_INSTALL_PREFIX" \
    Python3_EXECUTABLE="$Python3_EXECUTABLE" \
    LLVM_EXTRA_CMAKE_ARGS='-DLIBCXX_USE_COMPILER_RT=YES -DLIBUNWIND_USE_COMPILER_RT=YES' \
    bash "$this_file_dir/build_llvm.sh" -v
  else
    echo "LLVM already installed in $LLVM_INSTALL_PREFIX."
  fi

  export CC="$LLVM_INSTALL_PREFIX/bin/clang"
  export CXX="$LLVM_INSTALL_PREFIX/bin/clang++"
  export FC="$LLVM_INSTALL_PREFIX/bin/flang"
  echo "Configured C compiler: $CC"
  echo "Configured C++ compiler: $CXX"
  echo "Configured Fortran compiler: $FC"
fi

# [Blas] Needed for certain optimizers
if [ -n "$BLAS_INSTALL_PREFIX" ] && [ -z "$(echo $exclude_prereq | grep blas)" ]; then
  if [ ! -f "$BLAS_INSTALL_PREFIX/libblas.a" ] && [ ! -f "$BLAS_INSTALL_PREFIX/lib/libblas.a" ]; then
    echo "Installing BLAS..."
    temp_install_if_command_unknown wget wget
    temp_install_if_command_unknown make make
    if [ ! -x "$(command -v "$FC")" ]; then
      unset FC
      # On macOS, 'brew install gfortran' installs gcc which provides gfortran
      temp_install_if_command_unknown gfortran gfortran
    fi

    pushd "$PREREQS_BUILD_DIR"

    # See also: https://github.com/NVIDIA/cuda-quantum/issues/452
    wget "${BLAS_TARBALL_URL}"
    tar -xzvf "blas-${BLAS_VERSION}.tgz" && cd BLAS-3.11.0
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
    wget "${PERL_TARBALL_URL}"
    tar -xzf "perl-${PERL_VERSION}.tar.gz" && cd "perl-${PERL_VERSION}"
    ./Configure -des -Dcc="$CC" -Dprefix="$PREREQS_BUILD_DIR/perl5"
    find . -name "*.PL" -exec touch {} + # normalize WSL clock skew
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

    wget "${OPENSSL_TARBALL_URL}"
    tar -xf "openssl-${OPENSSL_VERSION}.tar.gz" && cd "openssl-${OPENSSL_VERSION}"
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
    wget "${CACERT_URL}"
    wget "${CACERT_SHA256_URL}"
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
    wget "${CURL_TARBALL_URL}"
    tar -xzvf "curl-${CURL_VERSION}.tar.gz" && cd "curl-${CURL_VERSION}"
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
    git clone --filter=tree:0 "${AWS_SDK_CPP_URL}" aws-sdk-cpp
    cd aws-sdk-cpp && git checkout "${AWS_SDK_CPP_REF}" && git submodule update --init --recursive

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

# [QRMI] Needed for the Pasqal QRMI connector
if [ -n "$QRMI_INSTALL_PREFIX" ] && [ -z "$(echo $exclude_prereq | grep qrmi)" ] && [ "$(uname)" = "Linux" ] && [ "$(uname -m)" = "x86_64" ]; then
  qrmi_header="$QRMI_INSTALL_PREFIX/include/qrmi.h"
  qrmi_library="$QRMI_INSTALL_PREFIX/lib64/libqrmi.so"
  if [ ! -f "$qrmi_header" ] || [ ! -f "$qrmi_library" ]; then
    echo "Installing QRMI C artifacts..."
    temp_install_if_command_unknown wget wget
    pushd "$PREREQS_BUILD_DIR"

    mkdir -p "$QRMI_INSTALL_PREFIX/include" "$QRMI_INSTALL_PREFIX/lib64"
    wget "${QRMI_RELEASE_BASE}/${QRMI_ARCHIVE}" -O "${QRMI_ARCHIVE}"

    if [ -x "$(command -v sha256sum)" ]; then
      computed_sha256="$(sha256sum "${QRMI_ARCHIVE}" | awk '{print $1}')"
    else
      computed_sha256="$(shasum -a 256 "${QRMI_ARCHIVE}" | awk '{print $1}')"
    fi
    if [ "$computed_sha256" != "$QRMI_ARCHIVE_SHA256" ]; then
      echo -e "\e[01;31mError: SHA-256 checksum mismatch for ${QRMI_ARCHIVE}.\e[0m" >&2
      echo "Expected: $QRMI_ARCHIVE_SHA256" >&2
      echo "Got:      $computed_sha256" >&2
      rm -f "${qrmi_archive}"
      (return 1 2>/dev/null) && return 1 || exit 1
    fi

    tar -xzf "${QRMI_ARCHIVE}"
    cp "${QRMI_UNPACK_DIR}/qrmi.h" "$qrmi_header"
    cp "${QRMI_UNPACK_DIR}/libqrmi.so" "$qrmi_library"
    rm -rf "${QRMI_ARCHIVE}" "${QRMI_UNPACK_DIR}"

    popd
    remove_temp_installs
  else
    echo "QRMI already installed in $QRMI_INSTALL_PREFIX."
  fi
elif [ -n "$QRMI_INSTALL_PREFIX" ] && [ -z "$(echo $exclude_prereq | grep qrmi)" ]; then
  echo "Skipping QRMI C artifacts install (supported only on Linux x86_64)."
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
