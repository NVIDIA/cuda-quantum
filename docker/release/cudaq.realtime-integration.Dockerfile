# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This file validates that CUDA-Q can be built with realtime integration enabled
# from the normal CUDA-Q development dependencies image. CUDA is installed here
# so the same base devdeps image can be reused across CUDA variants.

ARG base_image=ghcr.io/nvidia/cuda-quantum-devdeps:gcc12-main
# Default empty stage for ccache data. CI may override this with
# --build-context ccache-data=<path> to inject a pre-populated cache.
FROM scratch AS ccache-data
FROM $base_image AS realtime-integration
SHELL ["/bin/bash", "-c"]

ARG cuda_version=12.6
ENV CUDA_VERSION=${cuda_version}

# Install the CUDA compiler and the development libraries that CUDA-Q commonly
# detects during source builds. Keep this image lighter than the full
# devcontainer by omitting MPI, cuQuantum, and cuTensor.
ARG cuda_packages="cuda-cudart cuda-nvrtc cuda-compiler libcublas libcublas-dev libcurand-dev libcusolver libcusparse-dev libnvjitlink cuda-nvml-dev"
ARG DEBIAN_FRONTEND=noninteractive
RUN set -euo pipefail; \
    if [ -n "${cuda_packages}" ]; then \
      if [ "$(echo "${CUDA_VERSION}" | cut -d "." -f1)" -lt 12 ]; then \
        cuda_packages=$(echo "${cuda_packages}" | tr ' ' '\n' | grep -v "libnvjitlink" | tr '\n' ' ' | sed 's/ *$//'); \
      fi; \
      arch_folder=$([ "$(uname -m)" = "aarch64" ] && echo sbsa || echo x86_64); \
      cuda_suffix=$(echo "${CUDA_VERSION}" | cut -d. -f1-2 | tr . -); \
      cuda_packages=$(echo "${cuda_packages}" | tr ' ' '\n' | xargs -I {} echo {}-${cuda_suffix}); \
      rm -f /etc/apt/sources.list.d/cuda.list; \
      apt-get update; \
      apt-get install -y --no-install-recommends ca-certificates wget g++; \
      wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${arch_folder}/cuda-keyring_1.1-1_all.deb"; \
      dpkg -i cuda-keyring_1.1-1_all.deb; \
      apt-get update; \
      apt-get install -y --no-install-recommends --allow-change-held-packages ${cuda_packages}; \
      rm cuda-keyring_1.1-1_all.deb; \
      apt-get autoremove -y --purge; \
      apt-get clean; \
      rm -rf /var/lib/apt/lists/*; \
    fi

# libibverbs is required by the realtime CPU RoCE transport
# (cudaq-realtime-cpu-transport). Installing it here ensures the realtime build
# below actually compiles that library, so the device_call `cpu_roce` channel
# (runtime/internal/device_call/CpuRoceChannel.cpp) is exercised by this
# build-only integration check instead of being silently skipped.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libibverbs-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


ENV CUDA_INSTALL_PREFIX="/usr/local/cuda-${CUDA_VERSION}"
ENV CUDA_HOME="$CUDA_INSTALL_PREFIX"
ENV CUDA_ROOT="$CUDA_INSTALL_PREFIX"
ENV CUDA_PATH="$CUDA_INSTALL_PREFIX"
ENV PATH="${CUDA_INSTALL_PREFIX}/lib64/:${CUDA_INSTALL_PREFIX}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_INSTALL_PREFIX}/lib64:${LD_LIBRARY_PATH}"

ARG workspace=.
ARG destination=cuda-quantum
ADD "$workspace" "$destination"

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

RUN set -euo pipefail; \
    export CUDA_HOME="${CUDA_HOME:-${CUDA_INSTALL_PREFIX:-/usr/local/cuda-${cuda_version}}}"; \
    export CUDACXX="${CUDACXX:-${CUDA_HOME}/bin/nvcc}"; \
    realtime_prefix=/tmp/cudaq-realtime; \
    build_root=/tmp/build-realtime-integration; \
    rm -rf "${realtime_prefix}" "${build_root}"; \
    unset CUDAHOSTCXX; \
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
    export CUDAHOSTCXX="${CXX:-}"; \
    if [ -z "${CUDAHOSTCXX}" ]; then echo "CXX must be set for the CUDA-Q build."; exit 1; fi; \
    cd /cuda-quantum; \
    git config --global --add safe.directory "*"; \
    bash scripts/build_cudaq.sh -v -B "${build_root}/cudaq" -- \
      -DCUDAQ_REALTIME_DIR="${realtime_prefix}"; \
    cmake --build "${build_root}/cudaq" --parallel "$(nproc)"; \
    test -f "${realtime_prefix}/lib/libcudaq-realtime.so"; \
    test -f "${realtime_prefix}/lib/libcudaq-realtime-dispatch.a"; \
    test -f "${CUDAQ_INSTALL_PREFIX}/lib/libcudaq-device-call-runtime.so"; \
    echo "=== ccache stats ==="; \
    (ccache -s 2>/dev/null || true); \
    (ccache --print-stats 2>/dev/null || ccache -s 2>/dev/null) > /root/.ccache/_build_stats.txt

# ---------------------------------------------------------------------------
# Optional build-only compile of the cpu_roce device_call test harness.
#
# Opt-in via build-arg build_device_call_tests=true, used by a dedicated CI job
# (see realtime_integration_ci.yml). Kept as a separate layer so the default
# integration check above stays exactly as PR #4565 designed it: build-only,
# with NO unit tests compiled. When enabled, this reuses the already-built
# cudaq tree, reconfigures it with tests ON, and compiles ONLY the two cpu_roce
# targets -- test_cpu_roce_device_call (the GoogleTest fixture) and
# cpu_roce_test_daemon. It deliberately does NOT build the shared-memory
# DeviceCallDispatchTester.cu (honoring #4565's intent that it not compile in
# CI) and never runs anything (no RDMA NIC is available in CI).
# ---------------------------------------------------------------------------
ARG build_device_call_tests=false
RUN set -euo pipefail; \
    if [ "${build_device_call_tests}" != "true" ]; then \
      echo "build_device_call_tests=false: skipping cpu_roce harness compile"; \
      exit 0; \
    fi; \
    export CUDA_HOME="${CUDA_HOME:-${CUDA_INSTALL_PREFIX:-/usr/local/cuda-${cuda_version}}}"; \
    export CUDACXX="${CUDACXX:-${CUDA_HOME}/bin/nvcc}"; \
    build_root=/tmp/build-realtime-integration; \
    echo "=== reconfiguring cudaq with tests ON (cpu_roce harness only) ==="; \
    cmake "${build_root}/cudaq" -DCUDAQ_BUILD_TESTS=ON; \
    echo "=== compiling cpu_roce harness (build-only, not run) ==="; \
    cmake --build "${build_root}/cudaq" \
      --target test_cpu_roce_device_call cpu_roce_test_daemon \
      --parallel "$(nproc)"; \
    test -f "${build_root}/cudaq/unittests/test_cpu_roce_device_call"; \
    test -f "${build_root}/cudaq/unittests/cpu_roce_test_daemon"; \
    echo "=== cpu_roce harness compiled successfully (not executed) ==="
