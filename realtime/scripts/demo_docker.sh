#!/bin/bash

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This script runs CUDA-Q realtime demo environment in a Docker container. 
# It mounts the current directory, expected to be the installed CUDA-Q realtime directory, 
# into the container to allow running demos and utilities from the container.


set -o errexit
set -o xtrace

SCRIPT=`realpath "$0"`
HERE=`dirname "$SCRIPT"`

# Default container name, can be overridden with --name= option.
NAME=cudaq_realtime_demo

# See if we need to run our container differently.
while [ $# -ge 1 ]
do
case "$1" in
    --name=*)
        NAME="${1#--name=}"
        ;;
  *)
    break
    ;;
esac
shift
done

# Determine the CUDA version from the host system's NVIDIA driver, to select the appropriate container image.
DRIVER_CUDA_VERSION_MAJOR=$(nvidia-smi | grep -oE "CUDA Version: [0-9]+" | awk '{print $3}' )

# DOCA version
DOCA_VERSION=3.3.0

# Only support CUDA 13 for now, as the DOCA version we use only supports CUDA 13.
if [ "$DRIVER_CUDA_VERSION_MAJOR" != "13" ]; then
  echo "Warning: Detected NVIDIA driver CUDA version $DRIVER_CUDA_VERSION_MAJOR, but this demo script is designed for CUDA 13. The container image used may not be compatible with your system. Please ensure you have the appropriate NVIDIA driver installed for CUDA 13 to run this demo." >&2
  exit 1  
fi

IMAGE_NAME=cudaq-realtime-demo:doca${DOCA_VERSION}-cuda${DRIVER_CUDA_VERSION_MAJOR}
# Build the Docker image for the demo environment, to set the LD_LIBRARY_PATH and ensure the correct DOCA version is used.
docker build \
    --build-arg DOCA_VERSION=${DOCA_VERSION} \
    --build-arg CUDA_VERSION=${DRIVER_CUDA_VERSION_MAJOR} \
    --build-arg CUDAQ_REALTIME_DIR=$HERE \
    -t $IMAGE_NAME \
    -f "$HERE/demo.Dockerfile" \
    "$HERE"    

# Check if $HERE is indeed the installed CUDA-Q realtime directory by looking for the `bin/`, `include/`, and `lib/` directories, and `validate.sh`.
if [ ! -d "$HERE/bin" ] || [ ! -d "$HERE/include" ] || [ ! -d "$HERE/lib" ] || [ ! -f "$HERE/validate.sh" ]; then
  echo "Error: The current directory does not appear to be a valid CUDA-Q realtime installation. Please ensure you have built and installed CUDA-Q realtime, and that you are running this script from the installed directory." >&2
  exit 1
fi


# Run the container with the appropriate mounts and environment variables.
# Add $ROOT/lib to LD_LIBRARY_PATH in the container to ensure it can find the CUDA-Q realtime libraries.
# The run command is adapted from NVIDIA's holoscan-sensor-bridge 
# (https://github.com/nvidia-holoscan/holoscan-sensor-bridge) demo script. 
docker run \
    -it \
    --rm \
    --net host \
    --gpus all \
    --runtime=nvidia \
    --shm-size=1gb \
    --privileged \
    --name "$NAME" \
    -v $PWD:$PWD \
    -v $ROOT:$ROOT \
    -v $HOME:$HOME \
    -v /sys/bus/pci/devices:/sys/bus/pci/devices \
    -v /sys/kernel/mm/hugepages:/sys/kernel/mm/hugepages \
    -v /dev:/dev \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /tmp/argus_socket:/tmp/argus_socket \
    -v /sys/devices:/sys/devices \
    -v /var/nvidia/nvcam/settings:/var/nvidia/nvcam/settings \
    -v /opt/mellanox/doca \
    -w $PWD \
    -e NVIDIA_DRIVER_CAPABILITIES=graphics,video,compute,utility,display \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e DISPLAY=$DISPLAY \
    -e enableRawReprocess=2 \
    $IMAGE_NAME \
    $*
