#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Builds the MLIR Python bindings against an existing LLVM build tree and
# installs them into LLVM_INSTALL_PREFIX. Requires the manylinux devdeps
# base image, where build_llvm.sh has already configured and built the
# LLVM/MLIR libraries (without python-bindings) at LLVM_SOURCE/build.
#
# This avoids reconfiguring/rebuilding the full LLVM tree once per Python
# version: only the binding objects (linked against the active interpreter
# ABI) and a small amount of tablegen output get produced.
#
# Required environment:
#   LLVM_SOURCE              path to the llvm-project source (default: /llvm-project)
#   LLVM_INSTALL_PREFIX      install destination (default: /usr/local/llvm)
#   Python3_EXECUTABLE       interpreter to bind against
#   NANOBIND_INSTALL_PREFIX  nanobind install dir (default: /usr/local/nanobind)
#
# Usage:
#   Python3_EXECUTABLE=$(which python3.11) bash scripts/build_mlir_python_bindings.sh

set -e

LLVM_SOURCE=${LLVM_SOURCE:-/llvm-project}
LLVM_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX:-/usr/local/llvm}
NANOBIND_INSTALL_PREFIX=${NANOBIND_INSTALL_PREFIX:-/usr/local/nanobind}
LLVM_BUILD_FOLDER=${LLVM_BUILD_FOLDER:-build}

if [ -z "$Python3_EXECUTABLE" ]; then
  echo "Python3_EXECUTABLE must be set."
  exit 1
fi

llvm_build_dir="$LLVM_SOURCE/$LLVM_BUILD_FOLDER"
if [ ! -f "$llvm_build_dir/CMakeCache.txt" ]; then
  echo "Expected pre-configured LLVM build at $llvm_build_dir."
  echo "This script must run on top of the manylinux devdeps base image."
  exit 1
fi

this_file_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# nanobind links against the active interpreter ABI; build it if missing.
if [ ! -d "$NANOBIND_INSTALL_PREFIX" ] || [ -z "$(ls -A "$NANOBIND_INSTALL_PREFIX"/* 2>/dev/null)" ]; then
  echo "Building nanobind..."
  cd "$this_file_dir/.." && repo_root=$(git rev-parse --show-toplevel) && cd "$repo_root"
  git submodule update --init --recursive --recommend-shallow --single-branch tpls/nanobind
  mkdir -p tpls/nanobind/build && cd tpls/nanobind/build
  cmake -G Ninja ../ \
    -DCMAKE_INSTALL_PREFIX="$NANOBIND_INSTALL_PREFIX" \
    -DPython3_EXECUTABLE="$Python3_EXECUTABLE" \
    -DNB_TEST=False
  cmake --build . --target install --config Release
fi

cd "$llvm_build_dir"

# Reconfigure the existing build tree: enable MLIR python bindings and
# point at the active interpreter. Other cache entries (compiler flags,
# enabled projects, ccache launcher, build type) are preserved.
echo "Reconfiguring LLVM build for python bindings (Python: $Python3_EXECUTABLE)..."
cmake . \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DPython3_EXECUTABLE="$Python3_EXECUTABLE" \
  -Dnanobind_DIR="$NANOBIND_INSTALL_PREFIX/nanobind/cmake"

# Build and install only the python-binding components. Other targets in
# the existing tree are left untouched and remain installed at
# LLVM_INSTALL_PREFIX from the base image.
echo "Building and installing MLIR python bindings..."
ninja install-MLIRPythonModules install-mlir-python-sources

echo "Installed MLIR python bindings into $LLVM_INSTALL_PREFIX."
