#!/bin/bash

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Configure, build and install the standalone cudaq-realtime libraries.
#
# Usage:
# bash realtime/scripts/build_realtime.sh
# -or-
# bash realtime/scripts/build_realtime.sh -c Debug -s
# -or-
# CUDAQ_REALTIME_INSTALL_PREFIX=/path/for/installing bash realtime/scripts/build_realtime.sh
# -or-
# bash realtime/scripts/build_realtime.sh -- -DCMAKE_CUDA_FLAGS=-D_SOME_GUARD
#
# Options:
# -c <build_configuration>: The build configuration to use. Defaults to Release.
# -B <build_dir>: The build directory to use. Defaults to realtime/build.
# -i: Whether to build incrementally. Defaults to False.
# -s: Enable sanitizers (ASan, UBSan) for memory error detection. Defaults to
#     False. Intended to be combined with -c Debug.
# -j <num_jobs>: The number of jobs to use. Defaults to all available cores.
# -h: Print this help text.
# --: Arguments after -- are passed directly to cmake (e.g., -DVAR=value).
#     They are appended last, so they take precedence over the defaults set
#     by this script.
#
# Environment:
#   CUDAQ_REALTIME_INSTALL_PREFIX   Install prefix. Falls back to
#                                   CUDAQ_INSTALL_PREFIX, then
#                                   $HOME/.cudaq_realtime. Point a subsequent
#                                   CUDA-Q build at it with
#                                   -DCUDAQ_REALTIME_DIR=<prefix>.
#   CC / CXX                        Host compilers.
#   CMAKE_CUDA_FLAGS                Extra nvcc flags, appended to the flags this
#                                   script sets itself. A -DCMAKE_CUDA_FLAGS
#                                   given after -- overrides both.
#   CUDAQ_REALTIME_BUILD_TESTS      Build the unit tests. Defaults to ON.
#   CUDAQ_REALTIME_BUILD_EXAMPLES   Build the examples. Defaults to OFF.
#   HSB_ROOT                     Holoscan Sensor Bridge source checkout.
#                                   HSB tools are enabled when it exists.
#                                   Defaults to $HOME/cudaq/holoscan-sensor-bridge;
#                                   set it to the empty string to never enable them.
#
# Prerequisites:
# - CMake 4.0+, ninja-build
# - CUDA toolkit (12+); a CUDA-less configure builds only the UDP transport.
# - For HSB tools: DOCA with gpunetio and a built holoscan-sensor-bridge tree,
#   see isntall_dev_prerequisites.sh

set -euo pipefail

this_file_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
realtime_src="$(cd "$this_file_dir/.." && pwd -P)"

usage() {
  cat <<'EOF'
Usage:
  bash realtime/scripts/build_realtime.sh [options] [-- <extra cmake args>]

Options:
  -c <build_configuration>  Build configuration (default: Release)
  -B <build_dir>            Build directory (default: realtime/build)
  -i                        Build incrementally (default: clean build)
  -s                        Enable sanitizers (ASan, UBSan)
  -j <num_jobs>             Number of build jobs (default: all cores)
  -h                        Print this help text
EOF
}

build_configuration=${CMAKE_BUILD_TYPE:-Release}
build_dir="$realtime_src/build"
clean_build=true
enable_sanitizers=false
parallel_args=(--parallel)

# Extract extra cmake args after the -- separator.
extra_cmake_args=()
args_before_sep=()
found_sep=false
for arg in "$@"; do
  if [ "$arg" = "--" ] && ! $found_sep; then
    found_sep=true
  elif $found_sep; then
    extra_cmake_args+=("$arg")
  else
    args_before_sep+=("$arg")
  fi
done
set -- ${args_before_sep[@]+"${args_before_sep[@]}"}

__optind__=$OPTIND
OPTIND=1
while getopts ":c:B:isj:h" opt; do
  case $opt in
    c) build_configuration="$OPTARG"
    ;;
    B) build_dir="$OPTARG"
    ;;
    i) clean_build=false
    ;;
    s) enable_sanitizers=true
    ;;
    j) parallel_args=(--parallel "$OPTARG")
    ;;
    h) usage
    exit 0
    ;;
    \?) echo "Invalid command line option -$OPTARG" >&2
    usage >&2
    exit 1
    ;;
  esac
done
OPTIND=$__optind__

install_prefix=${CUDAQ_REALTIME_INSTALL_PREFIX:-${CUDAQ_INSTALL_PREFIX:-$HOME/.cudaq_realtime}}

# HSB tools need a built holoscan-sensor-bridge tree; enable them if we find one.
# An explicitly empty HSB_SRC_DIR opts out of the search.
hsb_src_dir=${HSB_SRC_DIR-/tmp/holoscan-sensor-bridge}
hsb_cmake_args=(-DCUDAQ_REALTIME_ENABLE_HSB_TOOLS=OFF)
if [ -n "$hsb_src_dir" ] && [ -d "$hsb_src_dir" ]; then
  echo "Holoscan Sensor Bridge detected in $hsb_src_dir. Enabling HSB tools."
  hsb_cmake_args=(
    -DCUDAQ_REALTIME_ENABLE_HSB_TOOLS=ON
    "-DHOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR=$hsb_src_dir"
    "-DHOLOSCAN_SENSOR_BRIDGE_BUILD_DIR=$hsb_src_dir/build"
  )
fi

ccache_args=()
if [ -x "$(command -v ccache)" ]; then
  echo "ccache detected. Configuring build to use ccache for faster recompilation."
  ccache_args=(
    -DCMAKE_C_COMPILER_LAUNCHER=ccache
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
    -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache
  )
else
  echo "ccache not found. To speed up recompilation, consider installing ccache."
fi

sanitizer_args=()
if $enable_sanitizers; then
  echo "Enabling Address Sanitizer (ASan) and Undefined Behavior Sanitizer (UBSan)..."
  sanitizer_args=(-DCUDAQ_REALTIME_ENABLE_SANITIZERS=ON)
fi

# The caller's CMAKE_CUDA_FLAGS is appended to the flags this script needs
# itself (none so far), so callers never have to repeat the script's own.
cuda_flags="${CMAKE_CUDA_FLAGS:-}"

echo "Build directory: $build_dir"
if $clean_build && [ -d "$build_dir" ]; then
  echo "Removing existing build directory for a clean build..."
  rm -rf "$build_dir"
fi
mkdir -p "$build_dir"

echo "Preparing cudaq-realtime build from $realtime_src..."
cmake_args=(
  -G Ninja
  -S "$realtime_src"
  -B "$build_dir"
  "-DCMAKE_BUILD_TYPE=$build_configuration"
  "-DCMAKE_INSTALL_PREFIX=$install_prefix"
  "-DCUDAQ_REALTIME_BUILD_TESTS=${CUDAQ_REALTIME_BUILD_TESTS:-ON}"
  "-DCUDAQ_REALTIME_BUILD_EXAMPLES=${CUDAQ_REALTIME_BUILD_EXAMPLES:-OFF}"
  "${hsb_cmake_args[@]}"
  ${ccache_args[@]+"${ccache_args[@]}"}
  ${sanitizer_args[@]+"${sanitizer_args[@]}"}
  ${cuda_flags:+"-DCMAKE_CUDA_FLAGS=$cuda_flags"}
  ${extra_cmake_args[@]+"${extra_cmake_args[@]}"}
)
cmake "${cmake_args[@]}"

echo "Building cudaq-realtime with configuration $build_configuration..."
cmake --build "$build_dir" --target install "${parallel_args[@]}"

echo "Installed cudaq-realtime in directory: $install_prefix"
