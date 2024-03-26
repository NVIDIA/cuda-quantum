#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Usage:
# bash scripts/build_cudaq.sh
# -or-
# bash scripts/build_cudaq.sh -c DEBUG
# -or-
# LLVM_INSTALL_PREFIX=/path/to/dir bash scripts/build_cudaq.sh
# -or-
# CUDAQ_INSTALL_PREFIX=/path/for/installing/cudaq LLVM_INSTALL_PREFIX=/path/to/dir bash scripts/build_cudaq.sh
# -or-
# CUQUANTUM_INSTALL_PREFIX=/path/to/dir bash scripts/build_cudaq.sh
#
# Prerequisites:
# - git, ninja-build, python3, libpython3-dev, libstdc++-12-dev (all available via apt install)
# - LLVM binaries, libraries, and headers as built by scripts/build_llvm.sh.
# - To include simulator backends that use cuQuantum the packages cuquantum and cuquantum-dev are needed. 
# - Additional python dependencies for running and testing: lit pytest numpy (available via pip install)
# - Additional dependencies for GPU-accelerated components: cuquantum, cutensor, cuda-11-8
#
# Note:
# The CUDA Quantum build automatically detects whether the necessary libraries to build
# GPU-based components are available and will omit them from the build if they are not. 
#
# Note:
# By default, the CUDA Quantum is done with warnings-as-errors turned on.
# You can turn this setting off by defining the environment variable CUDAQ_WERROR=OFF.

LLVM_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX:-/opt/llvm}
CUQUANTUM_INSTALL_PREFIX=${CUQUANTUM_INSTALL_PREFIX:-/opt/nvidia/cuquantum}
CUTENSOR_INSTALL_PREFIX=${CUTENSOR_INSTALL_PREFIX:-/opt/nvidia/cutensor}
CUDAQ_INSTALL_PREFIX=${CUDAQ_INSTALL_PREFIX:-"$HOME/.cudaq"}

# Process command line arguments
(return 0 2>/dev/null) && is_sourced=true || is_sourced=false
build_configuration=${CMAKE_BUILD_TYPE:-Release}
verbose=false
install_prereqs=false

__optind__=$OPTIND
OPTIND=1
while getopts ":c:uv" opt; do
  case $opt in
    c) build_configuration="$OPTARG"
    ;;
    u) install_prereqs=true
    ;;
    v) verbose=true
    ;;
    \?) echo "Invalid command line option -$OPTARG" >&2
    if $is_sourced; then return 1; else exit 1; fi
    ;;
  esac
done
OPTIND=$__optind__

# Run the script from the top-level of the repo
working_dir=`pwd`
this_file_dir=`dirname "$(readlink -f "${BASH_SOURCE[0]}")"`
repo_root=$(cd "$this_file_dir" && git rev-parse --show-toplevel)

# Prepare the build directory
mkdir -p "$CUDAQ_INSTALL_PREFIX/bin"
mkdir -p "$working_dir/build" && cd "$working_dir/build" && rm -rf * 
mkdir -p logs && rm -rf logs/*

if $install_prereqs; then
  echo "Installing pre-requisites..."
  if $verbose; then
    source "$this_file_dir/install_prerequisites.sh"
    status=$?
  else
    echo "The install log can be found in `pwd`/logs/prereqs_output.txt."
    source "$this_file_dir/install_prerequisites.sh" 2> logs/prereqs_error.txt 1> logs/prereqs_output.txt
    status=$?
  fi

  (return 0 2>/dev/null) && is_sourced=true || is_sourced=false
  if [ "$status" = "" ] || [ ! "$status" -eq "0" ]; then
    echo "Failed to install prerequisites."
    cd "$working_dir" && if $is_sourced; then return 1; else exit 1; fi
  fi
fi

# Check if a suitable CUDA version is installed
cuda_version=`"${CUDACXX:-nvcc}" --version 2>/dev/null | grep -o 'release [0-9]*\.[0-9]*' | cut -d ' ' -f 2`
cuda_major=`echo $cuda_version | cut -d '.' -f 1`
cuda_minor=`echo $cuda_version | cut -d '.' -f 2`
if [ "$cuda_version" = "" ] || [ "$cuda_major" -lt "11" ] || ([ "$cuda_minor" -lt "8" ] && [ "$cuda_major" -eq "11" ]); then
  echo "CUDA version requirement not satisfied (required: >= 11.8, got: $cuda_version)."
  echo "GPU-accelerated components will be omitted from the build."
else 
  echo "CUDA version $cuda_version detected."
  if [ ! -d "$CUQUANTUM_INSTALL_PREFIX" ] || [ -z "$(ls -A "$CUQUANTUM_INSTALL_PREFIX"/* 2> /dev/null)" ]; then
    echo "No cuQuantum installation detected. Please set the environment variable CUQUANTUM_INSTALL_PREFIX to enable cuQuantum integration."
    echo "Some backends will be omitted from the build."
  else
    echo "Using cuQuantum installation in $CUQUANTUM_INSTALL_PREFIX."
  fi
  if [ ! -d "$CUTENSOR_INSTALL_PREFIX" ] || [ -z "$(ls -A "$CUTENSOR_INSTALL_PREFIX"/* 2> /dev/null)" ]; then
    echo "No cuTensor installation detected. Please set the environment variable CUTENSOR_INSTALL_PREFIX to enable cuTensor integration."
    echo "Some backends will be omitted from the build."
  else
    echo "Using cuTensor installation in $CUTENSOR_INSTALL_PREFIX."
  fi
fi

# Determine linker and linker flags
if [ -x "$(command -v "$LLVM_INSTALL_PREFIX/bin/ld.lld")" ]; then
  echo "Configuring nvq++ to use the lld linker by default."
  NVQPP_LD_PATH="$LLVM_INSTALL_PREFIX/bin/ld.lld"
fi

# Generate CMake files 
# (utils are needed for custom testing tools, e.g. CircuitCheck)
echo "Preparing CUDA Quantum build with LLVM installation in $LLVM_INSTALL_PREFIX..."
cmake_args="-G Ninja "$repo_root" \
  -DCMAKE_INSTALL_PREFIX="$CUDAQ_INSTALL_PREFIX" \
  -DNVQPP_LD_PATH="$NVQPP_LD_PATH" \
  -DCMAKE_CUDA_HOST_COMPILER="$CXX" \
  -DCMAKE_BUILD_TYPE=$build_configuration \
  -DCUDAQ_ENABLE_PYTHON=${CUDAQ_PYTHON_SUPPORT:-TRUE} \
  -DCUDAQ_BUILD_TESTS=${CUDAQ_BUILD_TESTS:-TRUE} \
  -DCUDAQ_TEST_MOCK_SERVERS=${CUDAQ_BUILD_TESTS:-TRUE} \
  -DCMAKE_COMPILE_WARNING_AS_ERROR=${CUDAQ_WERROR:-ON}"
# Note that even though we specify CMAKE_CUDA_HOST_COMPILER above, it looks like the 
# CMAKE_CUDA_COMPILER_WORKS checks do *not* use that host compiler unless the CUDAHOSTCXX 
# environment variable is specified. Setting this variable may hence be necessary in 
# some environments. On the other hand, this will also make CMake not detect CUDA, if 
# the set host compiler is not officially supported. We hence don't set that variable 
# here, but keep the definition for CMAKE_CUDA_HOST_COMPILER.
if $verbose; then 
  cmake $cmake_args
else
  cmake $cmake_args \
    2> logs/cmake_error.txt 1> logs/cmake_output.txt
fi

# Build and install CUDAQ
echo "Building CUDA Quantum with configuration $build_configuration..."
logs_dir=`pwd`/logs
function fail_gracefully {
  echo "Build failed. Please check the console output or the files in the $logs_dir directory."
  cd "$working_dir" && if $is_sourced; then return 1; else exit 1; fi
}

if $verbose; then 
  ninja install || fail_gracefully
else
  echo "The progress of the build is being logged to $logs_dir/ninja_output.txt."
  ninja install 2> "$logs_dir/ninja_error.txt" 1> "$logs_dir/ninja_output.txt" || fail_gracefully
fi

cp "$repo_root/LICENSE" "$CUDAQ_INSTALL_PREFIX/LICENSE"
cp "$repo_root/NOTICE" "$CUDAQ_INSTALL_PREFIX/NOTICE"
cp "$repo_root/scripts/cudaq_set_env.sh" "$CUDAQ_INSTALL_PREFIX/set_env.sh"

# The CUDA Quantum installation as built above is not fully self-contained;
# It will, in particular, break if the LLVM tools are not in the expected location.
# We save any system configurations that are assumed by the installation with the installation.
echo "<build_config>" > "$CUDAQ_INSTALL_PREFIX/build_config.xml"
echo "<LLVM_INSTALL_PREFIX>$LLVM_INSTALL_PREFIX</LLVM_INSTALL_PREFIX>" >> "$CUDAQ_INSTALL_PREFIX/build_config.xml"
echo "<CUQUANTUM_INSTALL_PREFIX>$CUQUANTUM_INSTALL_PREFIX</CUQUANTUM_INSTALL_PREFIX>" >> "$CUDAQ_INSTALL_PREFIX/build_config.xml"
echo "<CUTENSOR_INSTALL_PREFIX>$CUTENSOR_INSTALL_PREFIX</CUTENSOR_INSTALL_PREFIX>" >> "$CUDAQ_INSTALL_PREFIX/build_config.xml"
echo "</build_config>" >> "$CUDAQ_INSTALL_PREFIX/build_config.xml"

cd "$working_dir" && echo "Installed CUDA Quantum in directory: $CUDAQ_INSTALL_PREFIX"
