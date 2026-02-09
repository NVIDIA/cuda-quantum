#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Unified installer build script for Linux and macOS.
#
# This script packages a CUDA-Q installation into a self-extracting archive
# using makeself (https://makeself.io/).
#
# Usage:
#   bash scripts/build_installer.sh              # macOS only (CPU-only)
#   bash scripts/build_installer.sh -c 12        # Linux: build cu12 installer
#   bash scripts/build_installer.sh -c 13        # Linux: build cu13 installer
#
# Options:
#   -c <cuda_version>: CUDA variant, 12 or 13 (Linux only; macOS is CPU-only)
#   -o <output_dir>: Output directory for installer (default: out)
#   -i <install_dir>: Directory with built CUDA-Q installation (default: $HOME/.cudaq)
#   -V <version>: CUDA-Q version string for installer name (e.g., 0.9.0)
#   -d: Docker mode (assets already in place, skip verification checks)
#   -v: Verbose output
#
# Prerequisites:
#   - CUDA-Q must be built (run build_cudaq.sh first)
#   - makeself must be installed
#
# Environment variables (set by default by `source scripts/set_env_defaults.sh`):
#   CUDAQ_INSTALL_PREFIX: Path to CUDA-Q installation (default: $HOME/.cudaq)
#   LLVM_INSTALL_PREFIX: Path to LLVM installation
#   CUQUANTUM_INSTALL_PREFIX: Path to cuQuantum (Linux only)
#   CUTENSOR_INSTALL_PREFIX: Path to cuTensor (Linux only)

set -euo pipefail

# ============================================================================ #
# Setup
# ============================================================================ #

# Run from repo root
this_file_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root=$(cd "$this_file_dir" && git rev-parse --show-toplevel)
cd "$repo_root"

# Detect platform
platform=$(uname)
arch=$(uname -m)

# Default values
cuda_variant=""
output_dir="out"
install_dir=""
cudaq_version=""
docker_mode=false
verbose=false

# Parse command line arguments
__optind__=$OPTIND
OPTIND=1
while getopts ":c:o:i:V:dv" opt; do
  case $opt in
    c) cuda_variant="$OPTARG" ;;
    o) output_dir="$OPTARG" ;;
    i) install_dir="$OPTARG" ;;
    V) cudaq_version="$OPTARG" ;;
    d) docker_mode=true ;;
    v) verbose=true ;;
    \?)
      echo "Invalid command line option -$OPTARG" >&2
      exit 1
      ;;
  esac
done
OPTIND=$__optind__

if $verbose; then
  echo "Platform: $platform ($arch)"
fi

# ============================================================================ #
# Platform-specific configuration
# ============================================================================ #

# Build installer name based on platform and version
# Format: install_cuda_quantum[_darwin|_cu<N>][-<version>].<arch>
# - macOS: install_cuda_quantum_darwin[-<version>].arm64
# - Linux: install_cuda_quantum_cu<N>[-<version>].<arch>
build_installer_name() {
  local name="install_cuda_quantum"
  if [ "$platform" = "Darwin" ]; then
    name="${name}_darwin"
  elif [ -n "$cuda_variant" ]; then
    name="${name}_cu${cuda_variant}"
  fi
  if [ -n "$cudaq_version" ]; then
    name="${name}-${cudaq_version}"
  fi
  echo "${name}.${arch}"
}

if [ "$platform" = "Darwin" ]; then
  # macOS: CPU-only, no CUDA dependencies
  if [ -n "$cuda_variant" ]; then
    echo "Warning: -c option ignored on macOS (CPU-only)" >&2
  fi
  cuda_variant=""
  include_cuda_deps=false
  echo "macOS: building CPU-only installer"
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
  include_cuda_deps=true
  echo "Linux: building cu${cuda_variant} installer"
fi

installer_name=$(build_installer_name)

# Source environment variables for LLVM_INSTALL_PREFIX, etc.
if $docker_mode; then
  # Docker mode: use Docker container paths (from configure_build.sh)
  source "$this_file_dir/configure_build.sh"
else
  # Local mode: use platform-specific defaults
  source "$this_file_dir/set_env_defaults.sh"
fi

# Set install directory (use -i flag or CUDAQ_INSTALL_PREFIX env var)
# Defaults match build_cudaq.sh: $HOME/.cudaq for local builds
if [ -z "$install_dir" ]; then
  install_dir="${CUDAQ_INSTALL_PREFIX:-$HOME/.cudaq}"
fi

# Verification checks (skipped in Docker mode where assets are pre-configured)
if ! $docker_mode; then
  # Verify CUDA-Q is built
  if [ ! -d "$install_dir" ] || [ ! -f "$install_dir/bin/nvq++" ]; then
    echo "Error: CUDA-Q installation not found at $install_dir" >&2
    echo "Please build CUDA-Q first: bash scripts/build_cudaq.sh" >&2
    exit 1
  fi

  # Verify LLVM is available
  llvm_prefix="${LLVM_INSTALL_PREFIX:-/opt/llvm}"
  if [ ! -d "$llvm_prefix" ] || [ ! -f "$llvm_prefix/bin/clang" ]; then
    echo "Error: LLVM installation not found at $llvm_prefix" >&2
    exit 1
  fi

  # Verify makeself is installed
  if ! command -v makeself &>/dev/null; then
    echo "Error: makeself not found" >&2
    if [ "$platform" = "Darwin" ]; then
      echo "Install with: brew install makeself" >&2
    else
      echo "Install with: apt install makeself  # or: yum install makeself" >&2
    fi
    exit 1
  fi
fi

# Set llvm_prefix for verbose output (with default fallback)
llvm_prefix="${LLVM_INSTALL_PREFIX:-/opt/llvm}"

if $verbose; then
  echo "CUDAQ_INSTALL_PREFIX: $install_dir"
  echo "LLVM_INSTALL_PREFIX: $llvm_prefix"
  if $include_cuda_deps; then
    echo "CUQUANTUM_INSTALL_PREFIX: ${CUQUANTUM_INSTALL_PREFIX:-not set}"
    echo "CUTENSOR_INSTALL_PREFIX: ${CUTENSOR_INSTALL_PREFIX:-not set}"
  fi
fi

# ============================================================================ #
# Create staging directory with assets
# ============================================================================ #

# Work in build directory so relative paths in documented snippet work
assets_parent="$repo_root/build"
mkdir -p "$assets_parent"
rm -rf "$assets_parent/cuda_quantum_assets"

echo "Creating installer assets..."

# Verify CUDA dependencies exist (Linux only)
if $include_cuda_deps; then
  if [ ! -d "${CUQUANTUM_INSTALL_PREFIX}" ]; then
    echo "Error: cuQuantum not found at ${CUQUANTUM_INSTALL_PREFIX}" >&2
    exit 1
  fi
  if [ ! -d "${CUTENSOR_INSTALL_PREFIX}" ]; then
    echo "Error: cuTensor not found at ${CUTENSOR_INSTALL_PREFIX}" >&2
    exit 1
  fi
fi

echo "  Copying LLVM assets..."
echo "  Copying CUDA-Q assets..."
if $include_cuda_deps; then
  echo "  Copying cuQuantum assets..."
  echo "  Copying cuTensor assets..."
fi

# Export CUDAQ_INSTALL_PREFIX so documented snippet works
export CUDAQ_INSTALL_PREFIX="$install_dir"

# Change to build directory so documented relative paths work
pushd "$assets_parent" >/dev/null

# [>CUDAQuantumAssets]
mkdir -p cuda_quantum_assets/llvm/bin cuda_quantum_assets/llvm/lib cuda_quantum_assets/llvm/include
cp -a "${LLVM_INSTALL_PREFIX}/bin/"clang* cuda_quantum_assets/llvm/bin/
rm -f cuda_quantum_assets/llvm/bin/clang-format*
cp -a "${LLVM_INSTALL_PREFIX}/bin/llc" "${LLVM_INSTALL_PREFIX}/bin/lld" "${LLVM_INSTALL_PREFIX}/bin/ld.lld" cuda_quantum_assets/llvm/bin/
cp -a "${LLVM_INSTALL_PREFIX}/lib/"* cuda_quantum_assets/llvm/lib/
cp -a "${LLVM_INSTALL_PREFIX}/include/"* cuda_quantum_assets/llvm/include/
# Copy cuTensor and cuQuantum (Linux only; variables unset on macOS)
if $include_cuda_deps; then
  cp -a "${CUTENSOR_INSTALL_PREFIX}" cuda_quantum_assets
  cp -a "${CUQUANTUM_INSTALL_PREFIX}" cuda_quantum_assets
fi
# Copy CUDA-Q installation and build config
cp -a "${CUDAQ_INSTALL_PREFIX}/build_config.xml" cuda_quantum_assets/build_config.xml
cp -a "${CUDAQ_INSTALL_PREFIX}" cuda_quantum_assets/cudaq
# [<CUDAQuantumAssets]

popd >/dev/null

# Full path for rest of script
cuda_quantum_assets="$assets_parent/cuda_quantum_assets"

# Copy install script
cp "$this_file_dir/migrate_assets.sh" "$cuda_quantum_assets/install.sh"
chmod a+x "$cuda_quantum_assets/install.sh"

# ============================================================================ #
# Create self-extracting archive
# ============================================================================ #

mkdir -p "$output_dir"

echo "Creating self-extracting archive..."
if $verbose; then
  makeself_args="--gzip --sha256"
else
  makeself_args="--gzip --sha256 --quiet"
fi

# Add license if available
if [ -f "$repo_root/LICENSE" ]; then
  makeself_args="$makeself_args --license $repo_root/LICENSE"
fi

# Default installation target (same for both platforms)
default_target='/opt/nvidia/cudaq'

makeself $makeself_args \
  "$cuda_quantum_assets" \
  "$output_dir/$installer_name" \
  "CUDA-Q toolkit for heterogeneous quantum-classical workflows" \
  bash install.sh -t "$default_target"

echo ""
echo "Done! Installer created: $output_dir/$installer_name"
echo "To install: bash $output_dir/$installer_name --accept"

# Cleanup staging
if ! $verbose; then
  rm -rf "$cuda_quantum_assets"
fi
