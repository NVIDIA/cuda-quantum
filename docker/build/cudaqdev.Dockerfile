# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Usage:
# Build from the repo root with
#   docker build -t nvidia/cuda-quantum-dev:latest -f docker/build/cudaqdev.Dockerfile .
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
ARG env_tag=llvm-main
FROM $build_environment:$env_tag

ENV CUDAQ_REPO_ROOT=/workspaces/cuda-quantum
ADD . "$CUDAQ_REPO_ROOT"
WORKDIR "$CUDAQ_REPO_ROOT"

ENV CUDAQ_INSTALL_PREFIX=/usr/local/cudaq
ENV PATH="$CUDAQ_INSTALL_PREFIX/bin:${PATH}"
ENV PYTHONPATH="$CUDAQ_INSTALL_PREFIX:${PYTHONPATH}"

# Configuring build_environment that contains the necessary dependencies for GPU
# accelerated components and passing a build argument 
#   install="CMAKE_BUILD_TYPE=Release FORCE_COMPILE_GPU_COMPONENTS=true"
# creates a dev image that can be used as argument to docker/release/cudaq.Dockerfile
# to create the released cuda-quantum image.
ARG install=
RUN if [ -n "$install" ]; \
    then \
        expected_prefix=$CUDAQ_INSTALL_PREFIX; \
        export $install; \
        bash scripts/build_cudaq.sh -v; \
        if [ "$CUDAQ_INSTALL_PREFIX" != "$expected_prefix" ]; then \
            mkdir -p "$expected_prefix"; \
            mv "$CUDAQ_INSTALL_PREFIX"/* "$expected_prefix"; \
            rmdir "$CUDAQ_INSTALL_PREFIX"; \
        fi \
    fi
