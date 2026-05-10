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
#   LLVM_SOURCE          path to the llvm-project source (default: /llvm-project)
#   LLVM_INSTALL_PREFIX  install destination (default: /usr/local/llvm)
#   Python3_EXECUTABLE   interpreter to bind against; must have nanobind and
#                        numpy already installed (e.g. via pip).
#
# Usage:
#   Python3_EXECUTABLE=$(which python3.11) bash scripts/build_mlir_python_bindings.sh

set -e

LLVM_SOURCE=${LLVM_SOURCE:-/llvm-project}
LLVM_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX:-/usr/local/llvm}
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

# Locate the pip-installed nanobind's CMake config. nanobind ships its own
# cmake module via pip, so we don't need to build it from the submodule.
nanobind_cmake_dir=$("$Python3_EXECUTABLE" -m nanobind --cmake_dir)
if [ ! -d "$nanobind_cmake_dir" ]; then
  echo "Could not locate nanobind cmake dir from $Python3_EXECUTABLE."
  echo "Install with: $Python3_EXECUTABLE -m pip install 'nanobind>=2.9.0'"
  exit 1
fi

cd "$llvm_build_dir"

# Reconfigure the existing build tree: enable MLIR python bindings and
# point at the active interpreter. Other cache entries (compiler flags,
# enabled projects, ccache launcher, build type) are preserved.
echo "Reconfiguring LLVM build for python bindings (Python: $Python3_EXECUTABLE)..."
cmake . \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DPython3_EXECUTABLE="$Python3_EXECUTABLE" \
  -Dnanobind_DIR="$nanobind_cmake_dir"

# Build and install only the python-binding components. Other targets in
# the existing tree are left untouched and remain installed at
# LLVM_INSTALL_PREFIX from the base image.
echo "Building and installing MLIR python bindings..."
ninja install-MLIRPythonModules install-mlir-python-sources

echo "Installed MLIR python bindings into $LLVM_INSTALL_PREFIX."
