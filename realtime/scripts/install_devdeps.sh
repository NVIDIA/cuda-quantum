#!/bin/bash

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This script installs the DOCA/HSB dependencies needed to build CUDA-Q realtime
# inside a CI container that already ships Mellanox OFED. Such containers cannot
# use install_dev_prerequisites.sh, because doca-all and the Ubuntu libibverbs
# packages conflict with the container's OFED packages; only the GPUNetIO dev
# package is installed here.
#
# Usage:
# bash install_devdeps.sh
#
# Environment variables:
#   HSB_ROOT           Where to clone and build HSB.
#                      Default: /tmp/holoscan-sensor-bridge
#   CUDA_NATIVE_ARCH   CUDA architectures to compile HSB for.
#                      Default: derived from the CUDA toolkit version.

set -e

. "$(dirname "$0")/deps_common.sh"

if [ ! -x "$(command -v apt-get)" ]; then
  echo "install_devdeps.sh requires apt-get." >&2
  exit 1
fi

# Fail early if the CUDA toolkit is missing; the exact version is needed below.
CUDA_FULL_VERSION=$(cudaq_realtime_cuda_version)

retry apt-get update
retry apt-get install -y --no-install-recommends \
  git ninja-build curl pkg-config

# [DOCA Host]
# Only the GPUNetIO dev package, not doca-all.
cudaq_realtime_add_doca_repo
retry apt-get -y install --no-install-recommends libdoca-sdk-gpunetio-dev

# hololink_core links CUDA::nvrtc -- must match the exact toolkit version
CUDA_VER_DASH=$(echo $CUDA_FULL_VERSION | sed 's/\./-/')
retry apt-get install -y cuda-nvrtc-dev-$CUDA_VER_DASH 2>/dev/null || true

# [Holoscan SDK]
export CUDAQ_REALTIME_HOLOSCAN_FORCE_DEPS=1
cudaq_realtime_install_holoscan

cudaq_realtime_verify_sdks

# [Holoscan Sensor Bridge]
export CUDAQ_REALTIME_HSB_STRIP_OPERATORS=1
cudaq_realtime_build_hsb
