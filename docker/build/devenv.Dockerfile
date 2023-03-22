# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This file builds the development environment that contains the necessary development 
# dependencies for building and testing QODA. This does not include the CUDA, OpenMPI 
# and other dependencies that some of the simulator backends require. These backends
# will be omitted from the build if this environment is used.
#
# Usage:
# Must be built from the repo root with:
#   docker build -t ghcr.io/nvidia/cuda-quantum-devdeps:$toolchain -f docker/build/devenv.Dockerfile --build-arg toolchain=$toolchain .
#
# The variable $toolchain indicates which compiler toolchain to build the llvm libraries with. 
# The toolchain used to build the llvm binaries that CUDA Quantum depends on must be used to build
# CUDA Quantum. This image sets the CC and CXX environment variables to use that toolchain. 
# Currently, llvm (default), and gcc11 are supported. To use a different toolchain, add support 
# for it to the install_toolchain.sh script. If the toolchain is set to llvm, then the toolchain 
# will be built from source.

FROM ubuntu:22.04 as llvmbuild
SHELL ["/bin/bash", "-c"]

ARG toolchain=llvm
ADD ../../scripts /scripts

# When a dialogue box would be needed during install, assume default configurations.
# Set here to avoid setting it for all install commands. 
# Given as arg to make sure that this value is only set during build but not in the launched container.
ARG DEBIAN_FRONTEND=noninteractive
RUN apt update && apt-get install -y --no-install-recommends \
        ca-certificates openssl apt-utils \
    && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/* 

# Clone the LLVM source code
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && mkdir /llvm-project && cd /llvm-project && git init \
    && git remote add origin https://github.com/llvm/llvm-project \
    && git fetch origin --depth=1 c0b45fef155fbe3f17f9a6f99074682c69545488 && git reset --hard FETCH_HEAD \
    && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/* 

# Install prerequisites for building LLVM
RUN apt-get update && apt-get install -y --no-install-recommends \
        ninja-build cmake python3 \
    && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/* 

FROM ubuntu:22.04
SHELL ["/bin/bash", "-c"]

# When a dialogue box would be needed during install, assume default configurations.
# Set here to avoid setting it for all install commands. 
# Given as arg to make sure that this value is only set during build but not in the launched container.
ARG DEBIAN_FRONTEND=noninteractive
ENV HOME=/home SHELL=/bin/bash LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install the C++ standard library. We could alternatively build libc++ 
# as part of the LLVM build and compile against that instead of libstdc++.
RUN apt-get update && apt-get install -y --no-install-recommends libstdc++-11-dev \
    && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/*
ENV CPLUS_INCLUDE_PATH="$CPLUS_INCLUDE_PATH:/usr/include/c++/11/:/usr/include/x86_64-linux-gnu/c++/11"

# Install additional dependencies required to build and test CUDA Quantum.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ninja-build cmake \
        python3 python3-pip libpython3-dev \
        libblas-dev \
    && python3 -m pip install --no-cache-dir lit pytest numpy \
    && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/*
