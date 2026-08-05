#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Fail if the cusvsim libraries that back the `nvidia` target are missing from
# an installed artifact. Without this, validation passes silently: absent
# backends are skipped rather than failing.
#
# Usage: validate_cusvsim_libraries.sh <search-root> [library ...]
#   <search-root>  tree to search (e.g. $CUDA_QUANTUM_PATH or a wheel venv)
#   [library ...]  libraries to require (default: single-device + mgpu set)

set -uo pipefail

search_root="${1:?usage: $0 <search-root> [library ...]}"
shift

libraries=("$@")
if [ ${#libraries[@]} -eq 0 ]; then
  libraries=(
    libnvqir-cusvsim-fp64.so       # nvidia target, fp64 (default)
    libnvqir-cusvsim-fp32.so       # nvidia target, fp32
    libnvqir-nvidia-mgpu.so        # nvidia target, mgpu fp64
    libnvqir-nvidia-mgpu-fp32.so   # nvidia target, mgpu fp32
  )
fi

if [ ! -d "$search_root" ]; then
  echo "::error::cusvsim library check: search root '$search_root' does not exist."
  exit 1
fi

echo "Checking for cusvsim GPU backend libraries under: $search_root"
missing=0
for lib in "${libraries[@]}"; do
  # -L so a symlinked library still resolves to its regular-file target.
  found=$(find -L "$search_root" -type f -name "$lib" 2>/dev/null | head -n1)
  if [ -z "$found" ]; then
    echo "::error::Required cusvsim GPU backend library '$lib' is not included in the artifact."
    missing=1
  else
    echo "  found: $found"
  fi
done

if [ "$missing" -ne 0 ]; then
  echo "::error::One or more cusvsim GPU backend libraries are missing from the artifact."
  exit 1
fi

echo "All required cusvsim GPU backend libraries are present."
