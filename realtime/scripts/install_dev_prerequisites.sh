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
# CUDA-Q realtime from source with DOCA/HSB support. 
#
# Usage: 
# bash install_dev_prerequisites.sh
#
# Environment variables:
#   HSB_ROOT                     Where to clone and build HSB.
#                                Default: /tmp/holoscan-sensor-bridge
#   CUDA_NATIVE_ARCH             CUDA architectures to compile HSB for.
#                                Default: derived from the CUDA toolkit version.
#   CUDAQ_REALTIME_SKIP_HSB=1    Install the SDKs but skip the HSB build.
#
# Containers that already ship Mellanox OFED cannot use this script, because
# doca-all conflicts with the OFED packages; use install_devdeps.sh instead.

set -e

. "$(dirname "$0")/deps_common.sh"

if [ -x "$(command -v apt-get)" ]; then
  # Fail early if the CUDA toolkit is missing.
  cudaq_realtime_cuda_major > /dev/null

  # [Build tools]
  # Needed to build HSB from source below.
  apt-get update && apt-get install -y --no-install-recommends git ninja-build pkg-config

  # [libibverbs]
  echo "Installing libibverbs..."
  apt-get update && apt-get install -y --no-install-recommends libibverbs-dev

  # [DOCA Host]
  cudaq_realtime_add_doca_repo
  DEBIAN_FRONTEND=noninteractive apt-get -y install doca-all libdoca-sdk-gpunetio-dev

  # [Holoscan SDK]
  cudaq_realtime_install_holoscan

  cudaq_realtime_verify_sdks

  # [Holoscan Sensor Bridge]
  if [ "${CUDAQ_REALTIME_SKIP_HSB:-0}" == "1" ]; then
    echo "CUDAQ_REALTIME_SKIP_HSB is set, skipping the HSB build."
  else
    cudaq_realtime_build_hsb
  fi

elif [ -x "$(command -v dnf)" ]; then
  echo "RHEL is not supported. Please install DOCA and Holoscan SDK manually." >&2
else
  echo "No supported package manager detected." >&2
fi
