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
#   macOS:  Uses $HOME/.local/* and $HOME/.llvm for user-local installs
#   Linux:  Uses /usr/local/* and /opt/* for system-wide installs

if [ "$(uname)" = "Darwin" ]; then
  # macOS: user-local installations (no sudo required)
  export LLVM_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX:-$HOME/.llvm}
  export BLAS_INSTALL_PREFIX=${BLAS_INSTALL_PREFIX:-$HOME/.local/blas}
  export ZLIB_INSTALL_PREFIX=${ZLIB_INSTALL_PREFIX:-$HOME/.local/zlib}
  export OPENSSL_INSTALL_PREFIX=${OPENSSL_INSTALL_PREFIX:-$HOME/.local/ssl}
  export CURL_INSTALL_PREFIX=${CURL_INSTALL_PREFIX:-$HOME/.local/curl}
  export PYBIND11_INSTALL_PREFIX=${PYBIND11_INSTALL_PREFIX:-$HOME/.local/pybind11}
  export AWS_INSTALL_PREFIX=${AWS_INSTALL_PREFIX:-$HOME/.local/aws}
else
  # Linux: system-wide installations (may require sudo)
  export LLVM_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX:-/opt/llvm}
  export PYBIND11_INSTALL_PREFIX=${PYBIND11_INSTALL_PREFIX:-/usr/local/pybind11}
  export BLAS_INSTALL_PREFIX=${BLAS_INSTALL_PREFIX:-/usr/local/blas}
  export ZLIB_INSTALL_PREFIX=${ZLIB_INSTALL_PREFIX:-/usr/local/zlib}
  export OPENSSL_INSTALL_PREFIX=${OPENSSL_INSTALL_PREFIX:-/usr/lib/ssl}
  export CURL_INSTALL_PREFIX=${CURL_INSTALL_PREFIX:-/usr/local/curl}
  export AWS_INSTALL_PREFIX=${AWS_INSTALL_PREFIX:-/usr/local/aws}
  export CUQUANTUM_INSTALL_PREFIX=${CUQUANTUM_INSTALL_PREFIX:-/opt/nvidia/cuquantum}
  export CUTENSOR_INSTALL_PREFIX=${CUTENSOR_INSTALL_PREFIX:-/opt/nvidia/cutensor}
fi

