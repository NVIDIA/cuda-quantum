#!/bin/bash

# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Helper script to run the container with Soft-RoCE (rxe) support

set -o errexit

SCRIPT=`realpath "$0"`
HERE=`dirname "$SCRIPT"`
ROOT=`realpath "$HERE/.."`
VERSION=`cat $ROOT/VERSION 2>/dev/null || echo "latest"`

# Check if rdma_rxe kernel module is loaded on host
if ! lsmod | grep -q rdma_rxe; then
    echo "Warning: rdma_rxe kernel module is not loaded on the host."
    echo "To load it, run:"
    echo "  sudo modprobe rdma_rxe"
    echo ""
fi

# Check if any rxe devices exist
if ! rdma link show 2>/dev/null | grep -q rxe; then
    echo "Warning: No rxe devices found on the host."
    echo "To create one, run (replace eth0 with your network interface):"
    echo "  sudo rdma link add rxe0 type rxe netdev eth0"
    echo ""
fi

echo "Starting container with Soft-RoCE and GPU support..."

docker run -it --rm \
    --gpus all \
    --cap-add=NET_ADMIN \
    --cap-add=SYS_MODULE \
    --cap-add=IPC_LOCK \
    --device=/dev/infiniband \
    --network=host \
    --ulimit memlock=-1:-1 \
    -v "$ROOT:/workspace" \
    -w /workspace \
    nvqlink-prototype:$VERSION \
    "$@"

