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
VERSION=`cat $ROOT/VERSION`

DOCKER_BUILDKIT=1 docker build \
    --network=host \
    -t nvqlink-prototype:$VERSION \
    -f $HERE/Dockerfile \
    $ROOT
