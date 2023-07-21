#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
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
# The CUDA Quantum build automatically detects whether GPUs are available and will 
# only include any GPU based components if they are. It is possible to override this 
# behavior and force building GPU components even if no GPU is detected by setting the
# FORCE_COMPILE_GPU_COMPONENTS environment variable to true. This is useful primarily
# when building docker images since GPUs may not be accessible during build.

LLVM_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX:-/opt/llvm}
CUQUANTUM_INSTALL_PREFIX=${CUQUANTUM_INSTALL_PREFIX:-/opt/nvidia/cuquantum}
CUDAQ_INSTALL_PREFIX=${CUDAQ_INSTALL_PREFIX:-"$HOME/.cudaq"}
BLAS_INSTALL_PREFIX=${BLAS_INSTALL_PREFIX:-/usr/local/blas}
BLAS_LIBRARIES=${BLAS_LIBRARIES:-"$BLAS_INSTALL_PREFIX/libblas.a"}

# Process command line arguments
(return 0 2>/dev/null) && is_sourced=true || is_sourced=false
build_configuration=${CMAKE_BUILD_TYPE:-Release}
verbose=false

__optind__=$OPTIND
OPTIND=1
while getopts ":c:v" opt; do
  case $opt in
    c) build_configuration="$OPTARG"
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

source "$this_file_dir/install_prerequisites.sh"
(return 0 2>/dev/null) && is_sourced=true || is_sourced=false

# Check if a suitable CUDA version is installed
cuda_version=`nvcc --version 2>/dev/null | grep -o 'release [0-9]*\.[0-9]*' | cut -d ' ' -f 2`
cuda_major=`echo $cuda_version | cut -d '.' -f 1`
cuda_minor=`echo $cuda_version | cut -d '.' -f 2`
if [ ! -x "$(command -v nvidia-smi)" ] && [ "$FORCE_COMPILE_GPU_COMPONENTS" != "true" ] ; then # the second check here is to avoid having to use https://discuss.huggingface.co/t/how-to-deal-with-no-gpu-during-docker-build-time/28544 
  echo "No GPU detected - GPU backends will be omitted from the build."
  custatevec_flag=""
elif [ "$cuda_version" = "" ] || [ "$cuda_major" -lt "11" ] || ([ "$cuda_minor" -lt "8" ] && [ "$cuda_major" -eq "11" ]); then
  echo "CUDA version requirement not satisfied (required: >= 11.8, got: $cuda_version)."
  echo "GPU backends will be omitted from the build."
  custatevec_flag=""
else 
  echo "CUDA version $cuda_version detected."
  if [ ! -d "$CUQUANTUM_INSTALL_PREFIX" ]; then
    echo "No cuQuantum installation detected. Please set the environment variable CUQUANTUM_INSTALL_PREFIX to enable cuQuantum integration."
    echo "GPU backends will be omitted from the build."
    custatevec_flag=""
  else
    echo "Using cuQuantum installation in $CUQUANTUM_INSTALL_PREFIX."
    custatevec_flag="-DCUSTATEVEC_ROOT=$CUQUANTUM_INSTALL_PREFIX"
  fi
fi

# Prepare the build directory
mkdir -p "$CUDAQ_INSTALL_PREFIX/bin"
mkdir -p "$working_dir/build" && cd "$working_dir/build" && rm -rf * 
mkdir -p logs && rm -rf logs/* 

# Determine linker and linker flags
cmake_common_linker_flags_init=""
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
  -DCMAKE_BUILD_TYPE=$build_configuration \
  -DCUDAQ_ENABLE_PYTHON=TRUE \
  -DCUDAQ_TEST_MOCK_SERVERS=FALSE \
  -DBLAS_LIBRARIES="${BLAS_LIBRARIES}" \
  -DCMAKE_EXE_LINKER_FLAGS_INIT="$cmake_common_linker_flags_init" \
  -DCMAKE_MODULE_LINKER_FLAGS_INIT="$cmake_common_linker_flags_init" \
  -DCMAKE_SHARED_LINKER_FLAGS_INIT="$cmake_common_linker_flags_init" \
  $custatevec_flag"
if $verbose; then 
  LLVM_INSTALL_PREFIX="$LLVM_INSTALL_PREFIX" cmake $cmake_args
else
  LLVM_INSTALL_PREFIX="$LLVM_INSTALL_PREFIX" cmake $cmake_args \
    2> logs/cmake_error.txt 1> logs/cmake_output.txt
fi

# Build and install CUDAQ
echo "Building CUDA Quantum with configuration $build_configuration..."
logs_dir=`pwd`/logs
if $verbose; then 
  ninja install
else
  echo "The progress of the build is being logged to $logs_dir/ninja_output.txt."
  ninja install 2> "$logs_dir/ninja_error.txt" 1> "$logs_dir/ninja_output.txt"
fi

if [ ! "$?" -eq "0" ]; then
  echo "Build failed. Please check the console output or the files in the $logs_dir directory."
  cd "$working_dir" && if $is_sourced; then return 1; else exit 1; fi
else
  cp "$repo_root/LICENSE" "$CUDAQ_INSTALL_PREFIX/LICENSE"
  cp "$repo_root/NOTICE" "$CUDAQ_INSTALL_PREFIX/NOTICE"

  # The CUDA Quantum installation as built above is not fully self-container;
  # It will, in particular, break if the LLVM tools are not in the expected location.
  # We save any system configurations that are assumed by the installation with the installation.
  echo "<build_config>" > "$CUDAQ_INSTALL_PREFIX/build_config.xml"
  echo "<LLVM_INSTALL_PREFIX>$LLVM_INSTALL_PREFIX</LLVM_INSTALL_PREFIX>" >> "$CUDAQ_INSTALL_PREFIX/build_config.xml"
  echo "<CUQUANTUM_INSTALL_PREFIX>$CUQUANTUM_INSTALL_PREFIX</CUQUANTUM_INSTALL_PREFIX>" >> "$CUDAQ_INSTALL_PREFIX/build_config.xml"
  echo "</build_config>" >> "$CUDAQ_INSTALL_PREFIX/build_config.xml"

  cd "$working_dir" && echo "Installed CUDA Quantum in directory: $CUDAQ_INSTALL_PREFIX"
fi
