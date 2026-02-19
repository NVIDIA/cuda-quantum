#!/bin/bash

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Usage: 
# This script builds and installs a minimal set of dependencies needed to build 
# CUDA-Q realtime with Hololink stack. 
#
# Usage: 
# bash install_dev_prerequisites.sh


if [ -x "$(command -v apt-get)" ]; then
  # [libibverbs]
  echo "Installing libibverbs..."
  apt-get update && apt-get install -y --no-install-recommends libibverbs-dev

  # [DOCA Host]

  if [ ! -x "$(command -v curl)" ]; then
    apt-get update && apt-get install -y --no-install-recommends curl
  fi

  DOCA_VERSION=3.2.1
  echo "Installing DOCA version $DOCA_VERSION..."
  arch=$(uname -m)
  distro=$(. /etc/os-release && echo ${ID}${VERSION_ID}) # e.g., ubuntu24.04
  export DOCA_URL="https://linux.mellanox.com/public/repo/doca/$DOCA_VERSION/$distro/$arch/"
  echo "Using DOCA_REPO_LINK=${DOCA_URL}" 
  curl https://linux.mellanox.com/public/repo/doca/GPG-KEY-Mellanox.pub | gpg --dearmor > /etc/apt/trusted.gpg.d/GPG-KEY-Mellanox.pub
  echo "deb [signed-by=/etc/apt/trusted.gpg.d/GPG-KEY-Mellanox.pub] $DOCA_URL ./" > /etc/apt/sources.list.d/doca.list
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get -y install doca-all doca-sdk-gpunetio libdoca-sdk-gpunetio-dev

  # [Holoscan SDK]
  CUDA_MAJOR_VERSION=$(nvcc --version | sed -n 's/^.*release \([0-9]\+\).*$/\1/p')
  if [ -z "$CUDA_MAJOR_VERSION" ]; then
    echo "Could not determine CUDA version from nvcc. Is the CUDA toolkit installed?" >&2
    exit 1
  fi
  apt-get update && apt-get install -y --no-install-recommends holoscan-cuda-$CUDA_MAJOR_VERSION

elif [ -x "$(command -v dnf)" ]; then
  echo "TODO: Support RHEL." >&2
else
  echo "No supported package manager detected." >&2
fi
