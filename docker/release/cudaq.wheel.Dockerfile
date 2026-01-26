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

# Configure the build based on the CUDA version
RUN cd /cuda-quantum && \
    . scripts/configure_build.sh && \
    case "${CUDA_VERSION%%.*}" in \
      12) cp pyproject.toml.cu12 pyproject.toml || true ;; \
      13) cp pyproject.toml.cu13 pyproject.toml || true ;; \
      *)  echo "Unsupported CUDA_VERSION=${CUDA_VERSION}"; exit 1 ;; \
    esac

# Create the README
# Note: conda channel 13.0.0 does not contain CUDA 13 (still 12.x), so we map to 13.0.2
RUN cd cuda-quantum && cat python/README.md.in > python/README.md && \
    package_name=cuda-quantum-cu$(echo ${CUDA_VERSION} | cut -d . -f1) && \
    cuda_version_requirement="\>= ${CUDA_VERSION}" && \
    cuda_version_conda=${CUDA_VERSION}.0 && \
    cuda_version_conda=${cuda_version_conda/13.0.0/13.0.2} && \
    for variable in package_name cuda_version_requirement cuda_version_conda deprecation_notice; do \
        sed -i "s/.{{[ ]*$variable[ ]*}}/${!variable}/g" python/README.md; \
    done && \
    if [ -n "$(cat python/README.md | grep -e '.{{.*}}')" ]; then \
        echo "Incomplete template substitutions in README." && \
        exit 1; \
    fi

# Build the wheel
RUN echo "Building wheel for python${python_version}." \
    && rm ~/.cache/pip -rf \
    && cd cuda-quantum && python=python${python_version} \
    # Find any external NVQIR simulator assets to be pulled in during wheel packaging.
    && export CUDAQ_EXTERNAL_NVQIR_SIMS=$(bash scripts/find_wheel_assets.sh assets) \
    && export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/assets" \
    &&  SETUPTOOLS_SCM_PRETEND_VERSION=${CUDA_QUANTUM_VERSION:-0.0.0} \
        CUDACXX="$CUDA_INSTALL_PREFIX/bin/nvcc" CUDAHOSTCXX=$CXX \
        $python -m build --wheel \
    && cudaq_major=$(echo ${CUDA_VERSION} | cut -d . -f1) \
    && cudart_libsuffix=$([ "$cuda_major" == "11" ] && echo "11.0" || echo "12") \
    && $python -m pip install --no-cache-dir auditwheel \
    && LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/_skbuild/lib" \
        $python -m auditwheel -v repair dist/cuda_quantum*linux_*.whl \
            --exclude libcustatevec.so.1 \
            --exclude libcutensornet.so.2 \
            --exclude libcudensitymat.so.0 \
            --exclude libcublas.so.$cudaq_major \
            --exclude libcublasLt.so.$cudaq_major \
            --exclude libcurand.so.10 \
            --exclude libcusolver.so.11 \
            --exclude libcusparse.so.$cudaq_major \
            --exclude libcutensor.so.2 \
            --exclude libnvToolsExt.so.1 \ 
            --exclude libcudart.so.$cudart_libsuffix \
            --exclude libnvidia-ml.so.1 \
            --exclude libcuda.so.1

FROM scratch
COPY --from=wheelbuild /cuda-quantum/wheelhouse/*manylinux*.whl . 
