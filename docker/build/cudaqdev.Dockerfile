# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Usage:
# Build from the repo root with
#   docker build -t nvidia/cuda-quantum-dev:$tag -f docker/build/cudaqdev.Dockerfile . \
#   --build-arg tag=$tag --build-arg workspace=. --build-arg destination=workspaces/cuda-quantum 
#
# If a custom build_environment is used, then the build environment must 
# 1) have all the necessary build dependendencies installed
# 2) define the LLVM_INSTALL_PREFIX environment variable indicating where the 
#    the LLVM binaries that CUDA Quantum depends on are installed
# 3) set the CC and CXX environment variable to use the same compiler toolchain
#    as the LLVM dependencies have been built with.

# To keep the default build environment image to a reasonable size, it does not 
# contain the necessary dependencies to develop GPU-based components. You may hence
# see a message along the lines of "no GPU detected" during the CUDA Quantum build.
# Please install the necessary prerequisites listed in the CUDA Quantum build script,
# or use a suitable build_environment, to enable developing these components.
ARG build_environment=ghcr.io/nvidia/cuda-quantum-devdeps
ARG tag=llvm-main
FROM $build_environment:$tag

ARG workspace=.
ARG destination=workspaces/host
ADD "$workspace" "/$destination"

ENV PATH="${HOME}/.cudaq/bin:${PATH}"
ENV PYTHONPATH="${HOME}/.cudaq:${PYTHONPATH}"
ENV CUDAQ_REPO_ROOT=/workspaces/cuda-quantum

# Passing a build argument 
#   installation="CUDAQ_INSTALL_PREFIX=/opt/nvidia/cudaq CMAKE_BUILD_TYPE=Release FORCE_COMPILE_GPU_COMPONENTS=true"
# creates a suitable build environment based on which the released cuda-quantum image can be created.
ARG installation=
RUN if [ -n "$installation" ]; \
    then \
        export $installation; \
        cd "$CUDAQ_REPO_ROOT"; \
        bash scripts/build_cudaq.sh -v; \
    fi