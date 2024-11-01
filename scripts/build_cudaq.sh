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
# bash scripts/build_cudaq.sh -c Debug
# -or-
# LLVM_INSTALL_PREFIX=/path/to/dir bash scripts/build_cudaq.sh
# -or-
# CUDAQ_INSTALL_PREFIX=/path/for/installing/cudaq LLVM_INSTALL_PREFIX=/path/to/dir bash scripts/build_cudaq.sh
# -or-
# CUQUANTUM_INSTALL_PREFIX=/path/to/dir bash scripts/build_cudaq.sh
#
# Prerequisites:
# - glibc including development headers (available via package manager)
# - git, ninja-build, python3, libpython3-dev (all available via apt install)
# - LLVM binaries, libraries, and headers as built by scripts/build_llvm.sh.
# - To include simulator backends that use cuQuantum the packages cuquantum and cuquantum-dev are needed. 
# - Additional python dependencies for running and testing: lit pytest numpy (available via pip install)
# - Additional dependencies for GPU-accelerated components: cuquantum, cutensor, cuda-11-8
#
# Note:
# The CUDA-Q build automatically detects whether the necessary libraries to build
# GPU-based components are available and will omit them from the build if they are not. 
#
# Note:
# By default, the CUDA-Q is done with warnings-as-errors turned on.
# You can turn this setting off by defining the environment variable CUDAQ_WERROR=OFF.
#
# For more information about building cross-platform CUDA libraries,
# see https://developer.nvidia.com/blog/building-cuda-applications-cmake/
# (specifically also CUDA_SEPARABLE_COMPILATION)

LLVM_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX:-/opt/llvm}
CUQUANTUM_INSTALL_PREFIX=${CUQUANTUM_INSTALL_PREFIX:-/opt/nvidia/cuquantum}
CUTENSOR_INSTALL_PREFIX=${CUTENSOR_INSTALL_PREFIX:-/opt/nvidia/cutensor}
CUDAQ_INSTALL_PREFIX=${CUDAQ_INSTALL_PREFIX:-"$HOME/.cudaq"}

# Process command line arguments
build_configuration=${CMAKE_BUILD_TYPE:-Release}
verbose=false
install_toolchain=""

__optind__=$OPTIND
OPTIND=1
while getopts ":c:t:v" opt; do
  case $opt in
    c) build_configuration="$OPTARG"
    ;;
    t) install_toolchain="$OPTARG"
    ;;
    v) verbose=true
    ;;
    \?) echo "Invalid command line option -$OPTARG" >&2
    (return 0 2>/dev/null) && return 1 || exit 1
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

if [ -n "$install_toolchain" ]; then
  echo "Installing pre-requisites..."
  if $verbose; then
    source "$this_file_dir/install_prerequisites.sh" -t "$install_toolchain"
    status=$?
  else
    echo "The install log can be found in `pwd`/logs/prereqs_output.txt."
    source "$this_file_dir/install_prerequisites.sh" -t "$install_toolchain" \
      2> logs/prereqs_error.txt 1> logs/prereqs_output.txt
    status=$?
  fi

  if [ "$status" = "" ] || [ ! "$status" -eq "0" ]; then
    echo -e "\e[01;31mError: Failed to install prerequisites.\e[0m" >&2
    cd "$working_dir" && (return 0 2>/dev/null) && return 1 || exit 1
  fi
fi

# Check if a suitable CUDA version is installed
cuda_driver=${CUDACXX:-${CUDA_HOME:-/usr/local/cuda}/bin/nvcc}
cuda_version=`"$cuda_driver" --version 2>/dev/null | grep -o 'release [0-9]*\.[0-9]*' | cut -d ' ' -f 2`
cuda_major=`echo $cuda_version | cut -d '.' -f 1`
cuda_minor=`echo $cuda_version | cut -d '.' -f 2`
if [ "$cuda_version" = "" ] || [ "$cuda_major" -lt "11" ] || ([ "$cuda_minor" -lt "8" ] && [ "$cuda_major" -eq "11" ]); then
  echo "CUDA version requirement not satisfied (required: >= 11.8, got: $cuda_version)."
  echo "GPU-accelerated components will be omitted from the build."
  unset cuda_driver
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

# Determine CUDA flags
if [ -z "$CUDAHOSTCXX" ] && [ -z "$CUDAFLAGS" ]; then
  CUDAFLAGS='-allow-unsupported-compiler'
  if [ -x "$CXX" ] && [ -n "$("$CXX" --version | grep -i clang)" ]; then
    CUDAFLAGS+=" --compiler-options --stdlib=libstdc++"
  fi
  if [ -d "$GCC_TOOLCHAIN" ]; then 
    # e.g. GCC_TOOLCHAIN=/opt/rh/gcc-toolset-11/root/usr/
    CUDAFLAGS+=" --compiler-options --gcc-toolchain=\"$GCC_TOOLCHAIN\""
  fi
fi

# Determine OpenMP flags
if [ -n "$(find "$LLVM_INSTALL_PREFIX" -name 'libomp.so')" ]; then
  OMP_LIBRARY=${OMP_LIBRARY:-libomp}
  OpenMP_libomp_LIBRARY=${OMP_LIBRARY#lib}
  OpenMP_FLAGS="${OpenMP_FLAGS:-'-fopenmp'}"
fi

# Generate CMake files 
# (utils are needed for custom testing tools, e.g. CircuitCheck)
echo "Preparing CUDA-Q build with LLVM installation in $LLVM_INSTALL_PREFIX..."
cmake_args="-G Ninja '"$repo_root"' \
  -DCMAKE_INSTALL_PREFIX='"$CUDAQ_INSTALL_PREFIX"' \
  -DCMAKE_BUILD_TYPE=$build_configuration \
  -DNVQPP_LD_PATH='"$NVQPP_LD_PATH"' \
  -DCMAKE_CUDA_COMPILER='"$cuda_driver"' \
  -DCMAKE_CUDA_FLAGS='"$CUDAFLAGS"' \
  -DCMAKE_CUDA_HOST_COMPILER='"${CUDAHOSTCXX:-$CXX}"' \
  ${OpenMP_libomp_LIBRARY:+-DOpenMP_C_LIB_NAMES=lib$OpenMP_libomp_LIBRARY} \
  ${OpenMP_libomp_LIBRARY:+-DOpenMP_CXX_LIB_NAMES=lib$OpenMP_libomp_LIBRARY} \
  ${OpenMP_libomp_LIBRARY:+-DOpenMP_libomp_LIBRARY=$OpenMP_libomp_LIBRARY} \
  ${OpenMP_FLAGS:+"-DOpenMP_C_FLAGS='"$OpenMP_FLAGS"'"} \
  ${OpenMP_FLAGS:+"-DOpenMP_CXX_FLAGS='"$OpenMP_FLAGS"'"} \
  -DCUDAQ_REQUIRE_OPENMP=${CUDAQ_REQUIRE_OPENMP:-FALSE} \
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
  echo $cmake_args | xargs cmake
else
  echo $cmake_args | xargs cmake \
    2> logs/cmake_error.txt 1> logs/cmake_output.txt
fi

# Build and install CUDA-Q
echo "Building CUDA-Q with configuration $build_configuration..."
logs_dir=`pwd`/logs
if $verbose; then 
  ninja install
  status=$?
else
  echo "The progress of the build is being logged to $logs_dir/ninja_output.txt."
  ninja install 2> "$logs_dir/ninja_error.txt" 1> "$logs_dir/ninja_output.txt"
  status=$?
fi

if [ "$status" = "" ] || [ ! "$status" -eq "0" ]; then
  echo -e "\e[01;31mError: Build failed. Please check the console output or the files in the $logs_dir directory.\e[0m" >&2
  cd "$working_dir" && (return 0 2>/dev/null) && return 1 || exit 1
fi

cp "$repo_root/LICENSE" "$CUDAQ_INSTALL_PREFIX/LICENSE"
cp "$repo_root/NOTICE" "$CUDAQ_INSTALL_PREFIX/NOTICE"
cp "$repo_root/scripts/cudaq_set_env.sh" "$CUDAQ_INSTALL_PREFIX/set_env.sh"

# The CUDA-Q installation as built above is not fully self-contained;
# It will, in particular, break if the LLVM tools are not in the expected location.
# We save any system configurations that are assumed by the installation with the installation.
echo "<build_config>" > "$CUDAQ_INSTALL_PREFIX/build_config.xml"
echo "<LLVM_INSTALL_PREFIX>$LLVM_INSTALL_PREFIX</LLVM_INSTALL_PREFIX>" >> "$CUDAQ_INSTALL_PREFIX/build_config.xml"
echo "<CUQUANTUM_INSTALL_PREFIX>$CUQUANTUM_INSTALL_PREFIX</CUQUANTUM_INSTALL_PREFIX>" >> "$CUDAQ_INSTALL_PREFIX/build_config.xml"
echo "<CUTENSOR_INSTALL_PREFIX>$CUTENSOR_INSTALL_PREFIX</CUTENSOR_INSTALL_PREFIX>" >> "$CUDAQ_INSTALL_PREFIX/build_config.xml"
echo "</build_config>" >> "$CUDAQ_INSTALL_PREFIX/build_config.xml"

cd "$working_dir" && echo "Installed CUDA-Q in directory: $CUDAQ_INSTALL_PREFIX"
