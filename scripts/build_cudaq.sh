#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
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
# Options:
# -c <build_configuration>: The build configuration to use. Defaults to Release.
# -t <install_toolchain>: The toolchain to use. Defaults to None.
# -j <num_jobs>: The number of jobs to use. Defaults to None.
# -v: Whether to print verbose output. Defaults to False.
# -B <build_dir>: The build directory to use. Defaults to build.
# -i: Whether to build incrementally. Defaults to False.
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

CUDAQ_INSTALL_PREFIX=${CUDAQ_INSTALL_PREFIX:-"$HOME/.cudaq"}

# Process command line arguments
build_configuration=${CMAKE_BUILD_TYPE:-Release}
verbose=false
clean_build=true
install_toolchain=""
num_jobs=""

# Run the script from the top-level of the repo
working_dir=`pwd`
this_file_dir=`dirname "$(readlink -f "${BASH_SOURCE[0]}")"`
repo_root=$(cd "$this_file_dir" && git rev-parse --show-toplevel)
build_dir="$working_dir/build"

__optind__=$OPTIND
OPTIND=1
while getopts ":c:t:j:vB:i" opt; do
  case $opt in
    c) build_configuration="$OPTARG"
    ;;
    t) install_toolchain="$OPTARG"
    ;;
    j) num_jobs="-j $OPTARG"
    ;;
    v) verbose=true
    ;;
    B) build_dir="$OPTARG"
    ;;
    i) clean_build=false
    ;;
    \?) echo "Invalid command line option -$OPTARG" >&2
    (return 0 2>/dev/null) && return 1 || exit 1
    ;;
  esac
done
OPTIND=$__optind__

# Prepare the build directory
echo "Build directory: $build_dir"
mkdir -p "$CUDAQ_INSTALL_PREFIX/bin"
mkdir -p "$build_dir" && cd "$build_dir"
if $clean_build; then
  rm -rf *
fi
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
if [ "$cuda_version" = "" ] || [ "$cuda_major" -lt "12" ]; then
  echo "CUDA version requirement not satisfied (required: >= 12.0, got: $cuda_version)."
  echo "GPU-accelerated components will be omitted from the build."
  unset cuda_driver
else
  echo "CUDA version $cuda_version detected."
  if [ -z "$CUQUANTUM_INSTALL_PREFIX" ] && [ -x "$(command -v pip)" ] && [ -n "$(pip list | grep -o cuquantum-python-cu$cuda_major)" ]; then
    CUQUANTUM_INSTALL_PREFIX="$(pip show cuquantum-python-cu$cuda_major | sed -nE 's/Location: (.*)$/\1/p')/cuquantum"
  fi
  if [ ! -d "$CUQUANTUM_INSTALL_PREFIX" ] || [ -z "$(ls -A "$CUQUANTUM_INSTALL_PREFIX"/* 2> /dev/null)" ]; then
    echo "No cuQuantum installation detected. Please set the environment variable CUQUANTUM_INSTALL_PREFIX to enable cuQuantum integration."
    echo "Some backends will be omitted from the build."
  else
    echo "Using cuQuantum installation in $CUQUANTUM_INSTALL_PREFIX."
  fi

  if [ -z "$CUTENSOR_INSTALL_PREFIX" ] && [ -x "$(command -v pip)" ] && [ -n "$(pip list | grep -o cutensor-cu$cuda_major)" ]; then
    CUTENSOR_INSTALL_PREFIX="$(pip show cutensor-cu$cuda_major | sed -nE 's/Location: (.*)$/\1/p')/cutensor"
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
  echo "Configuring nvq++ and local build to use the lld linker by default."
  NVQPP_LD_PATH="$LLVM_INSTALL_PREFIX/bin/ld.lld"
  LINKER_TO_USE="lld"
  LINKER_FLAGS="-fuse-ld=lld -B$LLVM_INSTALL_PREFIX/bin"
  LINKER_FLAG_LIST="\
    -DCMAKE_LINKER='"$LINKER_TO_USE"' \
    -DCMAKE_EXE_LINKER_FLAGS='"$LINKER_FLAGS"' \
    -DLLVM_USE_LINKER='"$LINKER_TO_USE"'"
else
  echo "No lld linker detected. Using the system linker."
  LINKER_FLAG_LIST=""
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

# Check for ccache and configure compiler launcher
CCACHE_FLAGS=""
if [ -x "$(command -v ccache)" ]; then
  echo "ccache detected. Configuring build to use ccache for faster recompilation."
  CCACHE_FLAGS="\
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
  # Also enable ccache for CUDA if CUDA compiler is available
  if [ -n "$cuda_driver" ]; then
    CCACHE_FLAGS+=" -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache"
  fi
else
  echo "ccache not found. To speed up recompilation, consider installing ccache."
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
  ${LINKER_FLAG_LIST} \
  ${CCACHE_FLAGS} \
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
  status=$?
else
  echo $cmake_args | xargs cmake \
    2> logs/cmake_error.txt 1> logs/cmake_output.txt
  status=$?
fi

# Check if cmake succeeded
if [ "$status" -ne 0 ]; then
  echo -e "\e[01;31mError: CMake configuration failed. Please check logs/cmake_error.txt for details.\e[0m" >&2
  cat logs/cmake_error.txt >&2
  cd "$working_dir" && (return 0 2>/dev/null) && return 1 || exit 1
fi

# Build and install CUDA-Q
echo "Building CUDA-Q with configuration $build_configuration..."
logs_dir=`pwd`/logs
if $verbose; then 
  ninja ${num_jobs} install
  status=$?
else
  echo "The progress of the build is being logged to $logs_dir/ninja_output.txt."
  ninja ${num_jobs} install 2> "$logs_dir/ninja_error.txt" 1> "$logs_dir/ninja_output.txt"
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
