#!/bin/bash

# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set -o errexit
umask 0

SCRIPT=`realpath "$0"`
HERE=`dirname "$SCRIPT"`
ROOT=`realpath "$HERE/.."`
VERSION="latest"

# Detect CUDA version on host
CUDA_VERSION=""
if command -v nvidia-smi &> /dev/null; then
    # Get CUDA version from nvidia-smi
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | sed -n 's/.*CUDA Version: \([0-9]\+\)\.\([0-9]\+\).*/\1.\2/p')
    echo "Detected CUDA version from nvidia-smi: $CUDA_VERSION"
fi

# Determine base image based on CUDA major version
if [[ "$CUDA_VERSION" == 12.* ]]; then
    BASE_IMAGE="ghcr.io/nvidia/cuda-quantum-dev:ext-cu12.6-gcc11-main"
    echo "Using CUDA 12 base image: $BASE_IMAGE"
elif [[ "$CUDA_VERSION" == 13.* ]]; then
    BASE_IMAGE="ghcr.io/nvidia/cuda-quantum-dev:ext-cu13.0-gcc11-main"
    echo "Using CUDA 13 base image: $BASE_IMAGE"
else
    echo "Warning: Could not detect CUDA version or unsupported version ($CUDA_VERSION)"
    echo "Defaulting to CUDA 13 base image"
    BASE_IMAGE="ghcr.io/nvidia/cuda-quantum-dev:ext-cu13.0-gcc11-main"
fi

DOCKER_BUILDKIT=1 docker build \
    --network=host \
    --build-arg base_image="$BASE_IMAGE" \
    -t nvqlink-prototype:$VERSION \
    -f $HERE/Dockerfile \
    $ROOT
