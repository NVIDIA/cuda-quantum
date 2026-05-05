#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Sets default install prefix environment variables for CUDA-Q builds.
# This script is designed to be sourced by other build scripts.
#
# Usage: source scripts/set_env_defaults.sh
#
# The script respects existing environment variable values. If a variable
# is already set, it won't be overridden.
#
# Platform-specific defaults:
#   macOS:  Uses $HOME/.local/* for user-local installs
#   Linux:  Uses /usr/local/* and /opt/* for system-wide installs

if [ "$(uname)" = "Darwin" ]; then
  # macOS: user-local installations (no sudo required)
  mkdir -p "$HOME/.local/bin"
  mkdir -p "$HOME/.local/share/lib"
  export PATH="$PATH:$HOME/.local/bin"
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$HOME/.local/share/lib"
  export LLVM_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX:-$HOME/.local/llvm}
  export BLAS_INSTALL_PREFIX=${BLAS_INSTALL_PREFIX:-$HOME/.local/blas}
  export ZLIB_INSTALL_PREFIX=${ZLIB_INSTALL_PREFIX:-$HOME/.local/zlib}
  export OPENSSL_INSTALL_PREFIX=${OPENSSL_INSTALL_PREFIX:-$HOME/.local/ssl}
  export CURL_INSTALL_PREFIX=${CURL_INSTALL_PREFIX:-$HOME/.local/curl}
  export PYBIND11_INSTALL_PREFIX=${PYBIND11_INSTALL_PREFIX:-$HOME/.local/pybind11}
  export NANOBIND_INSTALL_PREFIX=${NANOBIND_INSTALL_PREFIX:-$HOME/.local/nanobind}
  export AWS_INSTALL_PREFIX=${AWS_INSTALL_PREFIX:-$HOME/.local/aws}
  # Include OpenMP by default on macOS since CUDA/GPU acceleration is unavailable.
  # To skip building OpenMP with LLVM (e.g., if using
  # Homebrew's libomp via 'brew install libomp'), set LLVM_PROJECTS without openmp.
  # `export LLVM_PROJECTS='clang;lld;mlir;python-bindings'`
  #
  # Detect whether the active macOS SDK's libc++ headers are compatible with
  # LLVM 16. SDK 26+ (Tahoe) removed __has_builtin guards for builtins like
  # __builtin_ctzg, making the headers incompatible with LLVM 16. In that case,
  # include runtimes (libc++, libcxxabi, etc.) so that nvq++ uses LLVM's own
  # libc++ headers. To force this behavior, set LLVM_PROJECTS to include 'runtimes'.
  if [ -z "${LLVM_PROJECTS:-}" ]; then
    _cudaq_llvm_projects='clang;lld;mlir;python-bindings;openmp'
    _sdk_major="$(xcrun --show-sdk-version 2>/dev/null | cut -d. -f1)"
    if [ -n "$_sdk_major" ] && [ "$_sdk_major" -ge 26 ] 2>/dev/null; then
      _cudaq_llvm_projects="${_cudaq_llvm_projects};runtimes"
    fi
    export LLVM_PROJECTS="$_cudaq_llvm_projects"
    unset _cudaq_llvm_projects _sdk_major
  fi
  # Set minimum macOS deployment target for consistent builds.
  # This ensures LLVM/clang and CUDA-Q libraries use the same target.
  export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-13.0}"
  # Default CC/CXX to the built LLVM toolchain once it exists. This keeps the
  # CUDA-Q build, nvq++, and the just-built MLIR/Clang all on the same
  # compiler (same warning set, same libc++ target), avoiding the drift
  # between Apple Clang / Homebrew Clang / upstream Clang that makes the
  # macOS path fragile. Guarded on the install existing so the first run of
  # build_llvm.sh (which needs a working system compiler) isn't broken.
  if [ -x "$LLVM_INSTALL_PREFIX/bin/clang++" ]; then
    export CC="${CC:-$LLVM_INSTALL_PREFIX/bin/clang}"
    export CXX="${CXX:-$LLVM_INSTALL_PREFIX/bin/clang++}"
  fi
else
  # Linux: system-wide installations (may require sudo)
  export LLVM_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX:-/opt/llvm}
  export PYBIND11_INSTALL_PREFIX=${PYBIND11_INSTALL_PREFIX:-/usr/local/pybind11}
  export NANOBIND_INSTALL_PREFIX=${NANOBIND_INSTALL_PREFIX:-/usr/local/nanobind}
  export BLAS_INSTALL_PREFIX=${BLAS_INSTALL_PREFIX:-/usr/local/blas}
  export ZLIB_INSTALL_PREFIX=${ZLIB_INSTALL_PREFIX:-/usr/local/zlib}
  export OPENSSL_INSTALL_PREFIX=${OPENSSL_INSTALL_PREFIX:-/usr/lib/ssl}
  export CURL_INSTALL_PREFIX=${CURL_INSTALL_PREFIX:-/usr/local/curl}
  export AWS_INSTALL_PREFIX=${AWS_INSTALL_PREFIX:-/usr/local/aws}
  export QRMI_INSTALL_PREFIX=${QRMI_INSTALL_PREFIX:-/usr/local/qrmi}
  export CUQUANTUM_INSTALL_PREFIX=${CUQUANTUM_INSTALL_PREFIX:-/opt/nvidia/cuquantum}
  export CUTENSOR_INSTALL_PREFIX=${CUTENSOR_INSTALL_PREFIX:-/opt/nvidia/cutensor}
fi
