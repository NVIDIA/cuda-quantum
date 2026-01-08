#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Unified wheel build script for Linux and macOS.
#
# Usage:
#   bash scripts/build_wheel.sh              # macOS only (uses cu13)
#   bash scripts/build_wheel.sh -c 12        # Linux: build cu12 wheel
#   bash scripts/build_wheel.sh -c 13        # Linux: build cu13 wheel
#
# Options:
#   -c <cuda_version>: CUDA variant, 12 or 13 (Linux only; macOS always uses cu13)
#   -o <output_dir>: Output directory for wheels (default: dist)
#   -a <assets_dir>: Directory containing external simulator assets (default: assets)
#   -t: Run validation tests after build
#   -q: Quick test mode (only run core tests, implies -t)
#   -p: Install prerequisites before building
#   -T <toolchain>: Toolchain to use with prerequisites (e.g., gcc12, llvm)
#   -i: Incremental build (reuse existing build artifacts)
#   -v: Verbose output
#
# Environment variables:
#   PYTHON: Python interpreter to use (default: python3)
#   CUDA_QUANTUM_VERSION: Version string for the wheel (default: 0.0.0)
#   CUDACXX: Path to nvcc compiler
#   CUDAHOSTCXX: Host compiler for CUDA

set -e

# Run from repo root
this_file_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root=$(cd "$this_file_dir" && git rev-parse --show-toplevel)
cd "$repo_root"

# Detect platform
platform=$(uname)
arch=$(uname -m)

# Default values
cuda_variant=""
output_dir="dist"
assets_dir="assets"
run_tests=false
quick_test=false
install_prereqs=false
install_toolchain=""
incremental=false
verbose=false

# Parse command line arguments
__optind__=$OPTIND
OPTIND=1
while getopts ":c:o:a:tqpT:iv" opt; do
  case $opt in
    c) cuda_variant="$OPTARG"
    ;;
    o) output_dir="$OPTARG"
    ;;
    a) assets_dir="$OPTARG"
    ;;
    t) run_tests=true
    ;;
    q) quick_test=true; run_tests=true
    ;;
    p) install_prereqs=true
    ;;
    T) install_prereqs=true; install_toolchain="$OPTARG"
    ;;
    i) incremental=true
    ;;
    v) verbose=true
    ;;
    \?) echo "Invalid command line option -$OPTARG" >&2
    exit 1
    ;;
  esac
done
OPTIND=$__optind__

if $verbose; then
  echo "Verbose mode enabled"
  echo "Platform: $platform ($arch)"
fi

# Install prerequisites (opt-in with -p or -T)
# When installing prerequisites, we also set default install prefix env vars
# so CMake knows where to find them. Without -p/-T, CMake uses standard discovery.
if $install_prereqs; then
  # Set defaults for where prerequisites will be installed
  source "$this_file_dir/set_env_defaults.sh"
  
  echo "Installing prerequisites..."
  # Save and clear positional parameters to avoid passing them to sourced script
  saved_args=("$@")
  if [ -n "$install_toolchain" ]; then
    set -- -t "$install_toolchain"
  else
    set --
  fi
  if $verbose; then
    source "$this_file_dir/install_prerequisites.sh" "$@"
  else
    source "$this_file_dir/install_prerequisites.sh" "$@" 2>&1 | tail -5
  fi
  # Restore positional parameters
  set -- "${saved_args[@]}"
  if [ $? -ne 0 ]; then
    echo "Error: Failed to install prerequisites" >&2
    exit 1
  fi
fi

# Determine CUDA variant
if [ "$platform" = "Darwin" ]; then
  # macOS: always use cu13 (CPU-only, CUDA deps excluded via env markers)
  cuda_variant="13"
  echo "macOS: building cu$cuda_variant wheel (CPU-only)"
else
  # Linux: require explicit -c option
  if [ -z "$cuda_variant" ]; then
    echo "Error: CUDA variant required. Use -c 12 or -c 13" >&2
    exit 1
  fi
  if [ "$cuda_variant" != "12" ] && [ "$cuda_variant" != "13" ]; then
    echo "Error: CUDA variant must be 12 or 13, got: $cuda_variant" >&2
    exit 1
  fi
  echo "Linux: building cu$cuda_variant wheel"
fi

# Set up Python
python="${PYTHON:-python3}"
if ! command -v "$python" &> /dev/null; then
  echo "Error: $python not found" >&2
  exit 1
fi
echo "Using Python: $($python --version)"

# Copy appropriate pyproject.toml
pyproject_src="pyproject.toml.cu${cuda_variant}"
if [ ! -f "$pyproject_src" ]; then
  echo "Error: $pyproject_src not found" >&2
  exit 1
fi
echo "Using pyproject: $pyproject_src"
cp -f "$pyproject_src" pyproject.toml 2>/dev/null || true

# Set up library path environment variable
if [ "$platform" = "Darwin" ]; then
  lib_path_var="DYLD_LIBRARY_PATH"
  lib_ext="dylib"
else
  lib_path_var="LD_LIBRARY_PATH"
  lib_ext="so"
fi

# Find external NVQIR simulator assets
export CUDAQ_EXTERNAL_NVQIR_SIMS=$(bash scripts/find_wheel_assets.sh "$assets_dir")
if [ -n "$CUDAQ_EXTERNAL_NVQIR_SIMS" ]; then
  echo "Found external simulator assets: $CUDAQ_EXTERNAL_NVQIR_SIMS"
  eval "export $lib_path_var=\"\${$lib_path_var:+\$$lib_path_var:}$(pwd)/$assets_dir\""
fi

# Set version
export SETUPTOOLS_SCM_PRETEND_VERSION=${CUDA_QUANTUM_VERSION:-0.0.0}
echo "Building wheel version: $SETUPTOOLS_SCM_PRETEND_VERSION"

# Set CUDA compiler if available (Linux only)
if [ "$platform" != "Darwin" ]; then
  if [ -n "$CUDACXX" ]; then
    export CUDACXX
  elif [ -f "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" ]; then
    export CUDACXX="${CUDA_HOME:-/usr/local/cuda}/bin/nvcc"
  fi
  if [ -n "$CUDAHOSTCXX" ]; then
    export CUDAHOSTCXX
  elif [ -n "$CXX" ]; then
    export CUDAHOSTCXX="$CXX"
  fi
fi

# Clean previous build artifacts (unless incremental)
if $incremental; then
  echo "Incremental build: reusing existing build artifacts"
  rm -rf dist/*.whl "$output_dir"/*.whl 2>/dev/null || true
else
  rm -rf _skbuild dist/*.whl "$output_dir"/*.whl 2>/dev/null || true
fi
mkdir -p "$output_dir"

# Build the wheel
echo "Building wheel..."
if $verbose; then
  echo "  Command: $python -m build --wheel"
  echo "  SETUPTOOLS_SCM_PRETEND_VERSION=$SETUPTOOLS_SCM_PRETEND_VERSION"
  if [ -n "$CUDAQ_EXTERNAL_NVQIR_SIMS" ]; then
    echo "  CUDAQ_EXTERNAL_NVQIR_SIMS=$CUDAQ_EXTERNAL_NVQIR_SIMS"
  fi
  echo ""
  $python -m build --wheel -v
else
  $python -m build --wheel 2>&1 | tail -20
fi

# Find the built wheel
wheel_file=$(ls dist/cuda_quantum*.whl 2>/dev/null | head -1)
if [ -z "$wheel_file" ]; then
  echo "Error: No wheel file found in dist/" >&2
  exit 1
fi
echo "Built wheel: $wheel_file"

# Repair the wheel (bundle dependencies)
echo "Repairing wheel..."
if $verbose; then
  echo "  Input wheel: $wheel_file"
fi

if [ "$platform" = "Darwin" ]; then
  # macOS: use delocate
  if ! command -v delocate-wheel &> /dev/null; then
    echo "Error: delocate not found. Install with: pip install -r requirements-dev.txt" >&2
    exit 1
  fi
  
  # delocate repairs the wheel in place or to wheelhouse/
  # Use --ignore-missing because internal libs reference each other via @rpath
  # and delocate can't resolve them (they're all packaged together)
  mkdir -p wheelhouse
  delocate_args="--ignore-missing -w wheelhouse"
  if $verbose; then
    echo "  Command: delocate-wheel -v $delocate_args $wheel_file"
    delocate-wheel -v $delocate_args "$wheel_file"
  else
    delocate-wheel $delocate_args "$wheel_file"
  fi
  
  # Move repaired wheel to output
  repaired_wheel=$(ls wheelhouse/cuda_quantum*.whl 2>/dev/null | head -1)
  if [ -n "$repaired_wheel" ]; then
    mv "$repaired_wheel" "$output_dir/"
    echo "Repaired wheel: $output_dir/$(basename "$repaired_wheel")"
  else
    # If delocate didn't produce output, use original
    mv "$wheel_file" "$output_dir/"
    echo "Wheel (no repair needed): $output_dir/$(basename "$wheel_file")"
  fi
  rm -rf wheelhouse
else
  # Linux: use auditwheel
  if ! command -v auditwheel &> /dev/null; then
    echo "Error: auditwheel not found. Install with: pip install -r requirements-dev.txt" >&2
    exit 1
  fi
  
  # Determine CUDA library exclusions
  cuda_major="$cuda_variant"
  cudart_libsuffix=$([ "$cuda_major" = "11" ] && echo "11.0" || echo "12")
  
  # Add build lib to library path for auditwheel
  eval "export $lib_path_var=\"\${$lib_path_var:+\$$lib_path_var:}$(pwd)/_skbuild/lib\""
  
  mkdir -p wheelhouse
  auditwheel_args="repair $wheel_file -w wheelhouse"
  auditwheel_args="$auditwheel_args --exclude libcustatevec.so.1"
  auditwheel_args="$auditwheel_args --exclude libcutensornet.so.2"
  auditwheel_args="$auditwheel_args --exclude libcudensitymat.so.0"
  auditwheel_args="$auditwheel_args --exclude libcublas.so.$cuda_major"
  auditwheel_args="$auditwheel_args --exclude libcublasLt.so.$cuda_major"
  auditwheel_args="$auditwheel_args --exclude libcurand.so.10"
  auditwheel_args="$auditwheel_args --exclude libcusolver.so.11"
  auditwheel_args="$auditwheel_args --exclude libcusparse.so.$cuda_major"
  auditwheel_args="$auditwheel_args --exclude libcutensor.so.2"
  auditwheel_args="$auditwheel_args --exclude libnvToolsExt.so.1"
  auditwheel_args="$auditwheel_args --exclude libcudart.so.$cudart_libsuffix"
  auditwheel_args="$auditwheel_args --exclude libnvidia-ml.so.1"
  auditwheel_args="$auditwheel_args --exclude libcuda.so.1"
  
  if $verbose; then
    echo "  Command: auditwheel $auditwheel_args"
    auditwheel -v $auditwheel_args
  else
    auditwheel $auditwheel_args
  fi
  
  # Move repaired wheel to output
  repaired_wheel=$(ls wheelhouse/*manylinux*.whl 2>/dev/null | head -1)
  if [ -n "$repaired_wheel" ]; then
    mv "$repaired_wheel" "$output_dir/"
    echo "Repaired wheel: $output_dir/$(basename "$repaired_wheel")"
  else
    mv "$wheel_file" "$output_dir/"
    echo "Wheel: $output_dir/$(basename "$wheel_file")"
  fi
  rm -rf wheelhouse
fi

echo "Done! Wheel available in $output_dir/"

# Run validation tests if requested
if $run_tests; then
  echo ""
  echo "Running validation tests..."
  
  # Build validation script arguments (auto-detects test files from repo)
  validate_args="-v $SETUPTOOLS_SCM_PRETEND_VERSION -i $output_dir"
  
  if $quick_test; then
    validate_args="$validate_args -q"
  fi
  
  # Add CUDA version for Linux
  if [ "$platform" != "Darwin" ]; then
    # Determine full CUDA version for conda (e.g., 12.6.0)
    if [ "$cuda_variant" = "12" ]; then
      cuda_version_conda="${CUDA_VERSION_CONDA:-12.6.0}"
    else
      cuda_version_conda="${CUDA_VERSION_CONDA:-13.0.0}"
    fi
    validate_args="$validate_args -c $cuda_version_conda"
  fi
  
  # Run validation (will auto-detect test files from repo)
  if $verbose; then
    echo "  Command: bash $this_file_dir/validate_pycudaq.sh $validate_args"
  fi
  bash "$this_file_dir/validate_pycudaq.sh" $validate_args
  if [ $? -ne 0 ]; then
    echo "Validation failed!" >&2
    exit 1
  fi
  echo "Validation passed!"
fi
