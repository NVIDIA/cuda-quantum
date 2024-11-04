# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
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

ARG base_image=ghcr.io/nvidia/cuda-quantum-devdeps:manylinux-amd64-cu11.8-gcc11-main
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

# Patch the pyproject.toml file to change the CUDA version if needed
RUN if [ "${CUDA_VERSION#12.}" != "${CUDA_VERSION}" ]; then \
        sed -i "s/-cu11/-cu12/g" cuda-quantum/pyproject.toml && \
        sed -i -E "s/(nvidia-cublas-cu[0-9]* ~= )[0-9\.]*/\1${CUDA_VERSION}/g" cuda-quantum/pyproject.toml; \
        sed -i -E "s/(nvidia-cuda-runtime-cu[0-9]* ~= )[0-9\.]*/\1${CUDA_VERSION}/g" cuda-quantum/pyproject.toml; \
    fi

# Build the wheel
RUN echo "Building wheel for python${python_version}." \
    && rm ~/.cache/pip -rf \
    && cd cuda-quantum && python=python${python_version} \
    # Find any external NVQIR simulator assets to be pulled in during wheel packaging.
    && export CUDAQ_EXTERNAL_NVQIR_SIMS=$(bash scripts/find_wheel_assets.sh assets) \
    && export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/assets" \
    && export CUQUANTUM_INSTALL_PREFIX=/usr/local/cuquantum \
    && export CUTENSOR_INSTALL_PREFIX=/usr/local/cutensor \
    && bash scripts/configure_build.sh install-cuquantum \
    && bash scripts/configure_build.sh install-cutensor \
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
            --exclude libcublas.so.$cudaq_major \
            --exclude libcublasLt.so.$cudaq_major \
            --exclude libcusolver.so.$cudaq_major \
            --exclude libcutensor.so.2 \
            --exclude libnvToolsExt.so.1 \ 
            --exclude libcudart.so.$cudart_libsuffix

FROM scratch
COPY --from=wheelbuild /cuda-quantum/wheelhouse/*manylinux*.whl . 

# cd /cuda-quantum/python/sdist
# echo $CUDA_QUANTUM_VERSION > /cuda-quantum/python/sdist/_version.txt
# cp -y /cuda-quantum/_version.py /cuda-quantum/python/sdist/_version.py
# cp /cuda-quantum/LICENSE /cuda-quantum/python/sdist/LICENSE
# cp /cuda-quantum/NOTICE /cuda-quantum/python/sdist/NOTICE
# cp /cuda-quantum/CITATION.cff /cuda-quantum/python/sdist/CITATION.cff
# CUDAQ_META_WHEEL_BUILD=1 python3.10 -m build . --sdist
# python3.10 -m pip install --extra-index-url $extraurl dist/cudaq-*.tar.gz
# python3.10 -m pip install pip3-autoremove
# /opt/_internal/cpython-3.10.15/bin/pip-autoremove cudaq
# rm -rf cudaq.egg-info/ dist