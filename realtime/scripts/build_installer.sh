#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Installer build script for CUDA-Q realtime
#
# This script packages a CUDA-Q realtime installation into a self-extracting archive
# using makeself (https://makeself.io/).
#
#
# Prerequisites:
#   - CUDA-Q realtime must be built (run build_cudaq.sh first)
#   - makeself must be installed
#
# Environment variables:
#   CUDAQ_REALTIME_INSTALL_PREFIX: Path to CUDA-Q realtime installation (default: $HOME/.cudaq_realtime)

set -euo pipefail

# ============================================================================ #
# Setup
# ============================================================================ #

# Run from repo root
this_file_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cuda_variant=""
arch=$(uname -m)

# Parse command line arguments
__optind__=$OPTIND
OPTIND=1
while getopts ":c:" opt; do
  case $opt in
    c) cuda_variant="$OPTARG" ;;
    \?)
      echo "Invalid command line option -$OPTARG" >&2
      exit 1
      ;;
  esac
done
OPTIND=$__optind__

# require explicit -c option
if [ -z "$cuda_variant" ]; then
  echo "Error: CUDA variant required. Use -c 12 or -c 13" >&2
  exit 1
fi
if [ "$cuda_variant" != "12" ] && [ "$cuda_variant" != "13" ]; then
  echo "Error: CUDA variant must be 12 or 13, got: $cuda_variant" >&2
  exit 1
fi


installer_name=install_cuda_quantum_realtime_cu${cuda_variant}.${arch}

echo "Building installer $installer_name for CUDA $cuda_variant on $arch..."

install_dir="${CUDAQ_REALTIME_INSTALL_PREFIX:-$HOME/.cudaq_realtime}"   

# Verify CUDA-Q Realtime is built
if [ ! -d "$install_dir" ] || [ ! -f "$install_dir/lib/libcudaq-realtime.so" ]; then
    echo "Error: CUDA-Q Realtime installation not found at $install_dir" >&2
    echo "Please build CUDA-Q Realtime first" >&2
    exit 1
fi

# Verify makeself is installed
if ! command -v makeself &>/dev/null; then
    echo "Error: makeself not found" >&2
    echo "Install with: apt install makeself  # or: yum install makeself" >&2
    exit 1
fi

echo "CUDAQ_REALTIME_INSTALL_PREFIX: $install_dir"

# ============================================================================ #
# Create self-extracting archive
# ============================================================================ #
output_dir="${output_dir:-out}"
mkdir -p "$output_dir"

echo "Creating self-extracting archive..."

makeself_args="--gzip --sha256"
# Add license if available
if [ -f "$this_file_dir/../LICENSE" ]; then
  makeself_args="$makeself_args --license $this_file_dir/../LICENSE"
fi

# Copy install script
cp "$this_file_dir/migrate_assets.sh" "$install_dir/install.sh"
chmod a+x "$install_dir/install.sh"

# Default installation target 
default_target='/opt/nvidia/cudaq/realtime'

makeself $makeself_args \
  "$install_dir" \
  "$output_dir/$installer_name" \
  "CUDA-Q Realtime" \
  bash install.sh -t "$default_target"

echo ""
echo "Done! Installer created: $output_dir/$installer_name"
echo "To install: bash $output_dir/$installer_name --accept"

# TODO: add uninstall script
