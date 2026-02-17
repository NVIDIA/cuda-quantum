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
  export AWS_INSTALL_PREFIX=${AWS_INSTALL_PREFIX:-$HOME/.local/aws}
  # Include OpenMP by default on macOS since CUDA/GPU acceleration is unavailable.
  # To skip building OpenMP with LLVM (e.g., if using
  # Homebrew's libomp via 'brew install libomp'), set LLVM_PROJECTS without openmp.
  # `export LLVM_PROJECTS='clang;lld;mlir;python-bindings'`
  export LLVM_PROJECTS=${LLVM_PROJECTS:-'clang;lld;mlir;python-bindings;openmp'}
  # Set minimum macOS deployment target for consistent builds.
  # This ensures LLVM/clang and CUDA-Q libraries use the same target.
  export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-13.0}"
else
  # Linux: system-wide installations (may require sudo)
  export LLVM_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX:-/opt/llvm}
  export BLAS_INSTALL_PREFIX=${BLAS_INSTALL_PREFIX:-/usr/local/blas}
  export ZLIB_INSTALL_PREFIX=${ZLIB_INSTALL_PREFIX:-/usr/local/zlib}
  export OPENSSL_INSTALL_PREFIX=${OPENSSL_INSTALL_PREFIX:-/usr/lib/ssl}
  export CURL_INSTALL_PREFIX=${CURL_INSTALL_PREFIX:-/usr/local/curl}
  export AWS_INSTALL_PREFIX=${AWS_INSTALL_PREFIX:-/usr/local/aws}
  export CUQUANTUM_INSTALL_PREFIX=${CUQUANTUM_INSTALL_PREFIX:-/opt/nvidia/cuquantum}
  export CUTENSOR_INSTALL_PREFIX=${CUTENSOR_INSTALL_PREFIX:-/opt/nvidia/cutensor}
fi

