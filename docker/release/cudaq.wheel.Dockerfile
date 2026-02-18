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
# Default empty stage for ccache data. CI overrides this with
# --build-context ccache-data=<path> to inject a pre-populated cache,
# while local/devcontainer builds get a harmless no-op (empty scratch).
FROM scratch AS ccache-data
FROM $base_image AS wheelbuild

ARG release_version=
ENV CUDA_QUANTUM_VERSION=$release_version

ARG workspace=.
ARG destination=cuda-quantum
ADD "$workspace" "$destination"

ARG python_version=3.10
ENV CCACHE_DIR=/root/.ccache
ENV CCACHE_BASEDIR=/cuda-quantum
ENV CCACHE_SLOPPINESS=include_file_mtime,include_file_ctime,time_macros,pch_defines
ENV CCACHE_LOGFILE=/root/.ccache/ccache.log
RUN --mount=from=ccache-data,target=/tmp/ccache-import,rw \
    if [ -d /tmp/ccache-import ] && [ "$(ls -A /tmp/ccache-import 2>/dev/null)" ]; then \
        echo "Importing ccache data..." && \
        mkdir -p /root/.ccache && cp -a /tmp/ccache-import/. /root/.ccache/ && \
        ccache -s 2>/dev/null || true && \
        ccache -z 2>/dev/null || true && \
        find /root/.ccache -type f | wc -l | tr -d ' ' > /root/.ccache/_restore_file_count.txt; \
    else \
        echo "No ccache data injected using empty scratch stage."; \
    fi
RUN echo "Building MLIR bindings for python${python_version}" && \
    CCACHE_DISABLE=1 python${python_version} -m pip install --no-cache-dir numpy && \
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
        -v && \
    echo "=== ccache stats ===" && (ccache -s 2>/dev/null || true) && \
    (ccache --print-stats 2>/dev/null || ccache -s 2>/dev/null) > /root/.ccache/_build_stats.txt

# Export ccache data so CI can extract it for persistence.
# Build with --target ccache-export --output type=local,dest=/tmp/ccache-out
FROM scratch AS ccache-export
COPY --from=wheelbuild /root/.ccache /ccache

FROM scratch
COPY --from=wheelbuild /cuda-quantum/wheelhouse/*manylinux*.whl . 
