#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This scripts builds the clang and mlir project from the source in the LLVM submodule.
# The binaries will be installed in the folder defined by the LLVM_INSTALL_PREFIX environment
# variable, or in $HOME/.llvm if LLVM_INSTALL_PREFIX is not defined. 
#
# Usage:
# bash scripts/build_llvm.sh
# -or-
# bash scripts/build_llvm.sh -c DEBUG
# -or-
# LLVM_INSTALL_PREFIX=/installation/path/ bash scripts/build_llvm.sh

LLVM_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX:-$HOME/.llvm}

# Process command line arguments
(return 0 2>/dev/null) && is_sourced=true || is_sourced=false
build_configuration=Release
llvm_projects="clang;lld;mlir"
verbose=false

__optind__=$OPTIND
OPTIND=1
while getopts ":c:s:p:v" opt; do
  case $opt in
    c) build_configuration="$OPTARG"
    ;;
    s) llvm_source="$OPTARG"
    ;;
    p) llvm_projects="$OPTARG"
    ;;
    v) verbose=true
    ;;
    \?) echo "Invalid command line option -$OPTARG" >&2
    if $is_sourced; then return 1; else exit 1; fi
    ;;
  esac
done
OPTIND=$__optind__

working_dir=`pwd`

if [ "$llvm_source" = "" ]; then
  cd $(git rev-parse --show-toplevel)
  echo "Cloning LLVM submodule..."
  git submodule update --init --recursive --recommend-shallow tpls/llvm
  llvm_source=tpls/llvm
fi

echo "Configured C compiler: $CC"
echo "Configured C++ compiler: $CXX"

# Prepare the build directory
mkdir -p "$LLVM_INSTALL_PREFIX"
mkdir -p "$llvm_source/build" && cd "$llvm_source/build" && rm -rf *
mkdir -p logs && rm -rf logs/* 

# Specify which components we need to keep the size of the LLVM build down
llvm_components="cmake-exports;llvm-headers;llvm-libraries;"
llvm_components+="clang-cmake-exports;clang-headers;clang-libraries;clang-resource-headers;"
llvm_components+="mlir-cmake-exports;mlir-headers;mlir-libraries;mlir-tblgen;"
llvm_components+="llvm-config;clang-format;lld;llc;clang;FileCheck;count;not"

# Generate CMake files
echo "Preparing LLVM build..."
cmake_args="-G Ninja ../llvm \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DCMAKE_BUILD_TYPE=$build_configuration \
  -DCMAKE_INSTALL_PREFIX="$LLVM_INSTALL_PREFIX" \
  -DLLVM_ENABLE_PROJECTS="$llvm_projects" \
  -DLLVM_DISTRIBUTION_COMPONENTS=$llvm_components \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_OPTIMIZED_TABLEGEN=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_ENABLE_OCAMLDOC=OFF \
  -DLLVM_ENABLE_BINDINGS=OFF \
  -DLLVM_INSTALL_UTILS=ON"
if $verbose; then
  cmake $cmake_args
else
  cmake $cmake_args 2> logs/cmake_error.txt 1> logs/cmake_output.txt
fi

# Build and install clang in a folder
echo "Building LLVM with configuration $build_configuration..."
if $verbose; then
  ninja install-distribution-stripped
  status=$?
else
  echo "The progress of the build is being logged to `pwd`/logs/ninja_output.txt."
  ninja install-distribution-stripped 2> logs/ninja_error.txt 1> logs/ninja_output.txt
  status=$?
fi

if [ "$status" = "" ] || [ ! "$status" -eq "0" ]; then
  echo "Build failed. Please check the files in the `pwd`/logs directory."
  cd "$working_dir" && if $is_sourced; then return 1; else exit 1; fi
else
  cp bin/llvm-lit "$LLVM_INSTALL_PREFIX/bin/"
  cd "$working_dir" && echo "Installed llvm build in directory: $LLVM_INSTALL_PREFIX"
fi
