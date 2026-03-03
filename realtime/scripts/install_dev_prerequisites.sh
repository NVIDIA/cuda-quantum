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
# CUDA-Q realtime from source. 
#
# Usage: 
# bash install_dev_prerequisites.sh

# [DOCA Host]
# 'doca_gpunetio_dev_verbs_common.cuh' is required to build the realtime dispatch library.

DOCA_VERSION=3.3.0
echo "Installing DOCA version $DOCA_VERSION..."
arch=$(uname -m)
if [ "$arch" == "aarch64" ] || [ "$arch" == "arm64" ]; then
  arch="arm64-sbsa"
fi

if [ -x "$(command -v apt-get)" ]; then
  if [ ! -x "$(command -v curl)" ]; then
    apt-get update && apt-get install -y --no-install-recommends curl
  fi

  distro=$(. /etc/os-release && echo ${ID}${VERSION_ID}) # e.g., ubuntu24.04
  export DOCA_URL="https://linux.mellanox.com/public/repo/doca/$DOCA_VERSION/$distro/$arch/"
  echo "Using DOCA_REPO_LINK=${DOCA_URL}" 
  curl https://linux.mellanox.com/public/repo/doca/GPG-KEY-Mellanox.pub | gpg --dearmor > /etc/apt/trusted.gpg.d/GPG-KEY-Mellanox.pub
  echo "deb [signed-by=/etc/apt/trusted.gpg.d/GPG-KEY-Mellanox.pub] $DOCA_URL ./" > /etc/apt/sources.list.d/doca.list
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get -y install libdoca-sdk-gpunetio-dev

elif [ -x "$(command -v dnf)" ]; then
  DOCA_FULL_VERSION=3.3.0-088000_26.01
  # Find the rhel version, e.g., rhel9.2 -> rhel9
  distro=$(cat /etc/os-release | grep -E '^ID=' | cut -d= -f2 | tr -d '"')$(cat /etc/os-release | grep -E '^VERSION_ID=' | cut -d= -f2 | cut -d. -f1 | tr -d '"') # e.g., rhel9
  DOCA_URL=https://www.mellanox.com/downloads/DOCA/DOCA_v$DOCA_VERSION/host/doca-host-$DOCA_FULL_VERSION_$distro.$arch.rpm
  echo "DOCA_URL=${DOCA_URL}"
  wget $DOCA_URL -O doca-sdk.rpm
  dnf install -y doca-sdk.rpm
  rm doca-sdk.rpm
else
  echo "No supported package manager detected." >&2
fi
