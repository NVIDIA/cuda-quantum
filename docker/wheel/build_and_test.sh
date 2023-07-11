#!/bin/bash 

# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Build the wheels in a manylinux image, then pass the wheel off to a fresh 
# Ubuntu image where we will install it, then confirm it works by running the
# CUDA-Quantum pytest suite. 
# Usage:
#   bash build_and_test.sh 

# Remove old outputs.
rm -rf out/*
# Build the manylinux dependency image.
DOCKER_BUILDKIT=1 docker build -t nvidia/cudaq_manylinux_build -f docker/wheel/Dockerfile . --output out
# Test the wheels in a fresh Ubuntu image. This will install the wheel that was built
# in the manylinux container, then run the pytest suite using the cuda-quantum pip package.
DOCKER_BUILDKIT=1 docker build -t nvidia/cudaq_manylinux_test -f docker/wheel/tests/Dockerfile.ubuntu2204 . 
# Cleanup.
docker rmi -f nvidia/cudaq_manylinux_test nvidia/cudaq_manylinux_build 