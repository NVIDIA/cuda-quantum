#!/bin/bash
# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
#
# Regression test for the nvidia backend existence check.
# Verifies that a CUDAQ_INSTALL_PREFIX containing spaces is handled correctly.

set -e

CUDAQ_INSTALL_PREFIX="$(mktemp -d)/cuda quantum install"
mkdir -p "$CUDAQ_INSTALL_PREFIX/targets"
touch "$CUDAQ_INSTALL_PREFIX/targets/nvidia.yml"

if [ ! -e "$CUDAQ_INSTALL_PREFIX/targets/nvidia.yml" ]; then
    echo -e "\e[01;31mError: Missing nvidia backend.\e[0m" >&2
    rm -rf "$(dirname "$CUDAQ_INSTALL_PREFIX")"
    exit 1
fi

rm -rf "$(dirname "$CUDAQ_INSTALL_PREFIX")"
echo "PASS"
