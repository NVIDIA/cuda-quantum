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

# Augment LLVM_DISTRIBUTION_COMPONENTS to include the python-binding install
# components. The base image's build_llvm.sh sets this variable based on
# which projects were enabled at original configure time. If python-bindings
# wasn't in LLVM_PROJECTS then, MLIRPythonModules and mlir-python-sources are
# absent from the distribution list, and LLVM's distribution-aware export
# logic (LLVMDistributionSupport.cmake) filters MLIRPython* targets out of
# the MLIRTargets export set -- even after we flip MLIR_ENABLE_BINDINGS_PYTHON
# to ON. The downstream symptom is cuda-quantum's wheel cmake configure
# failing with `get_target_property() called with non-existent target
# "MLIRPythonExtension.RegisterEverything"` (etc) from
# python/extension/CMakeLists.txt. Adding these components here makes the
# reconfigure register them and regenerates MLIRTargets.cmake with the
# MLIRPython* entries the wheel build needs.
distribution_components_arg=""
existing_components=$(grep '^LLVM_DISTRIBUTION_COMPONENTS' CMakeCache.txt | cut -d= -f2- || true)
if [ -n "$existing_components" ]; then
  augmented="$existing_components"
  case ";$augmented;" in *";MLIRPythonModules;"*) ;; *) augmented="${augmented};MLIRPythonModules" ;; esac
  case ";$augmented;" in *";mlir-python-sources;"*) ;; *) augmented="${augmented};mlir-python-sources" ;; esac
  distribution_components_arg="-DLLVM_DISTRIBUTION_COMPONENTS=$augmented"
fi

# Reconfigure the existing build tree: enable MLIR python bindings and
# point at the active interpreter. Other cache entries (compiler flags,
# enabled projects, ccache launcher, build type) are preserved.
echo "Reconfiguring LLVM build for python bindings (Python: $Python3_EXECUTABLE)..."
cmake . \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DPython3_EXECUTABLE="$Python3_EXECUTABLE" \
  -Dnanobind_DIR="$nanobind_cmake_dir" \
  ${distribution_components_arg}

# Build and install only the python-binding components plus the regenerated
# MLIR cmake exports. The base image's mlir-cmake-exports was installed
# before python-bindings was enabled, so its MLIRTargets.cmake omits the
# MLIRPython* targets (MLIRPythonExtension.*, MLIRPythonSources.*,
# MLIRPythonCAPI.*). Reinstalling install-mlir-cmake-exports here regenerates
# the file with those entries, which downstream find_package(MLIR) +
# add_mlir_python_common_capi_library calls (in cuda-quantum's
# python/extension/CMakeLists.txt) require.
echo "Building and installing MLIR python bindings..."
ninja install-MLIRPythonModules install-mlir-python-sources install-mlir-cmake-exports

echo "Installed MLIR python bindings into $LLVM_INSTALL_PREFIX."
