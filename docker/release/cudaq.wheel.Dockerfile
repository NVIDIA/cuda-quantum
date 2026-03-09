# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This file is used to build CUDA-Q Python wheels.
# Build with buildkit to get the wheels as output instead of the image.
#
# Usage:
# Must be built from the repo root with:
#   DOCKER_BUILDKIT=1 docker build -f docker/release/cudaq.wheel.Dockerfile . --output out

# NOTES:
# Building wheels for Mac; use CI build wheel instead. Good example: 
# - https://github.com/numpy/numpy/blob/main/pyproject.toml, and 
# - https://github.com/numpy/numpy/blob/main/.github/workflows/wheels.yml

ARG base_image=ghcr.io/nvidia/cuda-quantum-devdeps:manylinux-amd64-cu12.6-gcc11-main
FROM $base_image AS wheelbuild

ARG release_version=
ENV CUDA_QUANTUM_VERSION=$release_version

ARG workspace=.
ARG destination=cuda-quantum
ADD "$workspace" "$destination"

ARG python_version=3.10
RUN echo "Building MLIR bindings for python${python_version}" && \
    python${python_version} -m pip install --no-cache-dir numpy && \
    rm -rf "$LLVM_INSTALL_PREFIX/src" "$LLVM_INSTALL_PREFIX/python_packages" && \
    Python3_EXECUTABLE="$(which python${python_version})" \
    LLVM_PROJECTS='clang;mlir;python-bindings' \
    LLVM_CMAKE_CACHE=/cmake/caches/LLVM.cmake LLVM_SOURCE=/llvm-project \
    bash /scripts/build_llvm.sh -c Release -v 

# Build wheel using unified wheel build script
RUN cd /cuda-quantum && \
    PYTHON=python${python_version} \
    CUDA_VERSION=${CUDA_VERSION} \
    bash scripts/build_wheel.sh \
        -c $(echo ${CUDA_VERSION} | cut -d . -f1) \
        -o wheelhouse \
        -a assets \
        -v

FROM scratch
COPY --from=wheelbuild /cuda-quantum/wheelhouse/*manylinux*.whl . 
