# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This file validates that CUDA-Q can be built with realtime integration enabled
# in the CUDA manylinux build environment. It intentionally repeats the MLIR
# Python binding setup from cudaq.wheel.Dockerfile.
#  TODO: Refactor to avoid duplication between this and cudaq.wheel.Dockerfile.

ARG base_image=ghcr.io/nvidia/cuda-quantum-devdeps:manylinux-amd64-cu12.6-gcc12-main
# Default empty stage for ccache data. CI may override this with
# --build-context ccache-data=<path> to inject a pre-populated cache.
FROM scratch AS ccache-data
FROM $base_image AS realtime-integration
SHELL ["/bin/bash", "-c"]

ARG workspace=.
ARG destination=cuda-quantum
ADD "$workspace" "$destination"

# BEGIN duplicated MLIR Python setup from cudaq.wheel.Dockerfile.
ARG python_version=3.12
ENV CCACHE_DIR=/root/.ccache
ENV CCACHE_BASEDIR=/cuda-quantum
ENV CCACHE_SLOPPINESS=include_file_mtime,include_file_ctime,time_macros,pch_defines
ENV CCACHE_COMPILERCHECK=content
ENV CCACHE_LOGFILE=/root/.ccache/ccache.log

RUN --mount=from=ccache-data,target=/tmp/ccache-import,rw \
    if [ -d /tmp/ccache-import ] && [ "$(ls -A /tmp/ccache-import 2>/dev/null)" ]; then \
        echo "Importing ccache data..." && \
        mkdir -p /root/.ccache && cp -a /tmp/ccache-import/. /root/.ccache/ && \
        ccache -s 2>/dev/null || true && \
        ccache -z 2>/dev/null || true && \
        find /root/.ccache -type f | wc -l | tr -d ' ' > /root/.ccache/_restore_file_count.txt; \
    else \
        echo "No ccache data injected using empty scratch stage." && \
        mkdir -p /root/.ccache; \
    fi

RUN echo "Building MLIR bindings for python${python_version}" && \
    CCACHE_DISABLE=1 python${python_version} -m pip install --no-cache-dir numpy "nanobind>=2.12.0" && \
    rm -rf "$LLVM_INSTALL_PREFIX/src" "$LLVM_INSTALL_PREFIX/python_packages" && \
    Python3_EXECUTABLE="$(which python${python_version})" \
    LLVM_SOURCE=/llvm-project \
    LLVM_INSTALL_PREFIX="$LLVM_INSTALL_PREFIX" \
    bash /scripts/build_mlir_python_bindings.sh
# END duplicated MLIR Python setup from cudaq.wheel.Dockerfile.

# The manylinux wheel Python provides extension-module support but not an
# embeddable libpython target, so keep Python linkage in the wheel mode.
RUN set -euo pipefail; \
    cuda_version="${CUDA_VERSION}"; \
    python_exe="$(which python${python_version})"; \
    export CUDA_HOME="${CUDA_HOME:-${CUDA_INSTALL_PREFIX:-/usr/local/cuda-${cuda_version}}}"; \
    export CUDACXX="${CUDACXX:-${CUDA_HOME}/bin/nvcc}"; \
    export CUDAHOSTCXX="${CUDAHOSTCXX:-${CXX:-}}"; \
    if [ -z "${CUDAHOSTCXX}" ]; then echo "CUDAHOSTCXX or CXX must be set."; exit 1; fi; \
    realtime_prefix=/tmp/cudaq-realtime; \
    build_root=/tmp/build-realtime-integration; \
    rm -rf "${realtime_prefix}" "${build_root}"; \
    cmake -G Ninja \
      -S /cuda-quantum/realtime \
      -B "${build_root}/realtime" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER="${CUDACXX}" \
      -DCUDAQ_REALTIME_BUILD_TESTS=OFF \
      -DCUDAQ_REALTIME_BUILD_EXAMPLES=OFF \
      -DCUDAQ_REALTIME_ENABLE_HOLOLINK_TOOLS=OFF \
      -DCMAKE_INSTALL_PREFIX="${realtime_prefix}"; \
    cmake --build "${build_root}/realtime" --target install --parallel "$(nproc)"; \
    export CUDAQ_INSTALL_PREFIX=/tmp/cudaq-realtime-enabled; \
    export CUDAQ_BUILD_TESTS=OFF; \
    export CUDAQ_WERROR=ON; \
    cd /cuda-quantum; \
    git config --global --add safe.directory "*"; \
    bash scripts/build_cudaq.sh -v -B "${build_root}/cudaq" -- \
      -DCUDAQ_REALTIME_DIR="${realtime_prefix}" \
      -DSKBUILD=ON \
      -DPython_EXECUTABLE="${python_exe}" \
      -DPython3_EXECUTABLE="${python_exe}"; \
    cmake --build "${build_root}/cudaq" --parallel "$(nproc)"; \
    test -f "${realtime_prefix}/lib/libcudaq-realtime.so"; \
    test -f "${realtime_prefix}/lib/libcudaq-realtime-dispatch.a"; \
    test -f "${CUDAQ_INSTALL_PREFIX}/lib/libcudaq-device-call-runtime.so"; \
    echo "=== ccache stats ==="; \
    (ccache -s 2>/dev/null || true); \
    (ccache --print-stats 2>/dev/null || ccache -s 2>/dev/null) > /root/.ccache/_build_stats.txt
