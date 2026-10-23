#!/bin/bash

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under    #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This script installs what a machine needs to run CUDA-Q Realtime from its
# installer, on top of the CUDA toolkit: the packages the installed libraries
# load at run time. The development scripts, install_dev_prerequisites.sh and
# install_devdeps.sh, install this set plus the build-only packages.
#
# Usage:
# bash install_prerequisites.sh

set -e

. "$(dirname "$0")/deps_common.sh"

if [ -x "$(command -v apt-get)" ]; then
  # Fail early if the CUDA toolkit is missing.
  cudaq_realtime_cuda_major > /dev/null

  retry apt-get update

  # [libibverbs]
  # The cpu_roce and gpu_roce transports load it.
  retry apt-get install -y --no-install-recommends libibverbs1

  # [DOCA Host]
  cudaq_realtime_add_doca_repo

  # [Runtime packages]
  cudaq_realtime_install_runtime_packages

elif [ -x "$(command -v dnf)" ]; then
  echo "RHEL is not supported. Please install DOCA manually." >&2
else
  echo "No supported package manager detected." >&2
fi
