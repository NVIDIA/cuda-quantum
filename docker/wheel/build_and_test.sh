#!/bin/bash 

# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Build and test Manylinux Python Wheels run with 
#
# ./build_and_test.sh 

export DOCKER_BUILDKIT=1
# Remove old outputs.
rm -rf out/*
# Build the wheels in a manylinux container.
docker build -t nvidia/cudaq_manylinux_build --network host . --output out
# Test the wheels in a fresh Ubuntu image. This will install the wheel that was built
# in the manylinux container, then run the pytest suite using the cuda-quantum pip package.
docker build -t nvidia/cudaq_manylinux_test --network host -f tests/Dockerfile.ubuntu2204 . 
# Cleanup.
docker rmi -f nvidia/cudaq_manylinux_test nvidia/cudaq_manylinux_build 