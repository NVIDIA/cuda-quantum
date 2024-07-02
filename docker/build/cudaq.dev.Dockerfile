# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Usage:
# Build from the repo root with
#   docker build -t nvidia/cuda-quantum-dev:latest -f docker/build/cudaq.dev.Dockerfile .
#
# If a custom base image is used, then that image (i.e. the build environment) must 
# 1) have all the necessary build dependendencies installed
# 2) define the LLVM_INSTALL_PREFIX environment variable indicating where the 
#    the LLVM binaries that CUDA-Q depends on are installed
# 3) set the CC and CXX environment variable to use the same compiler toolchain
#    as the LLVM dependencies have been built with.

ARG base_image=ghcr.io/nvidia/cuda-quantum-devdeps:ext-gcc11-main
FROM $base_image

ENV CUDAQ_REPO_ROOT=/workspaces/cuda-quantum
ENV CUDAQ_INSTALL_PREFIX=/usr/local/cudaq
ENV PATH="$CUDAQ_INSTALL_PREFIX/bin:${PATH}"
ENV PYTHONPATH="$CUDAQ_INSTALL_PREFIX:${PYTHONPATH}"

ARG workspace=.
ARG destination="$CUDAQ_REPO_ROOT"
ADD "$workspace" "$destination"
WORKDIR "$destination"

# mpich or openmpi
ARG mpi=
RUN if [ -n "$mpi" ]; \
    then \
        if [ ! -z "$MPI_PATH" ]; then \
            echo "Using a base image with MPI is not supported when passing a 'mpi' build argument." && exit 1; \
        else \
			apt update && apt install -y lib$mpi-dev ; \
		fi \
    fi

# Configuring a base image that contains the necessary dependencies for GPU
# accelerated components and passing a build argument 
#   install="CMAKE_BUILD_TYPE=Release CUDA_QUANTUM_VERSION=latest"
# creates a dev image that can be used as argument to docker/release/cudaq.Dockerfile
# to create the released cuda-quantum image.
ARG install=
ARG git_source_sha=xxxxxxxx
RUN if [ -n "$install" ]; \
    then \
        expected_prefix=$CUDAQ_INSTALL_PREFIX; \
        install=`echo $install | xargs` && export $install; \
        bash scripts/build_cudaq.sh -v; \
        if [ ! "$?" -eq "0" ]; then \
            exit 1; \
        elif [ "$CUDAQ_INSTALL_PREFIX" != "$expected_prefix" ]; then \
            mkdir -p "$expected_prefix"; \
            mv "$CUDAQ_INSTALL_PREFIX"/* "$expected_prefix"; \
            rmdir "$CUDAQ_INSTALL_PREFIX"; \
        fi; \
        echo "source-sha: $git_source_sha" > "$CUDAQ_INSTALL_PREFIX/build_info.txt"; \
    fi
