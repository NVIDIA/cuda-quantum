# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This file is used to build CUDA Quantum Python wheels.
# Build with buildkit to get the wheels as output instead of the image.
#
# Usage:
# Must be built from the repo root with:
#   DOCKER_BUILDKIT=1 docker build -f docker/build/cudaq.wheel.Dockerfile . --output out

ARG base_image=ghcr.io/nvidia/cuda-quantum-devdeps:manylinux-x86_64-main
FROM $base_image as wheelbuild

ARG release_version=
ENV CUDA_QUANTUM_VERSION=$release_version

ARG workspace=.
ARG destination=cuda-quantum
ADD "$workspace" "$destination"

ARG assets=./assets
COPY "$assets/" "$workspace/assets/"
ENV CUDAQ_EXTERNAL_NVQIR_SIMS=""

ARG python_version=3.10
RUN echo "Finding external NVQIR simulators." \
    && \
    for config_file in `find $workspace/assets/wheels_$(uname -m)/*.config -maxdepth 0 -type f`; \
        do \
            current_dir="$(dirname "${config_file}")" \
            && current_dir=$(cd $current_dir; pwd) \
            && target_name=${config_file##*/} \
            && target_name=${target_name%.config} \
            && \
                if [ -n "$CUDAQ_EXTERNAL_NVQIR_SIMS" ]; then \
                    export CUDAQ_EXTERNAL_NVQIR_SIMS="$CUDAQ_EXTERNAL_NVQIR_SIMS;$current_dir/libnvqir-$target_name.so;$current_dir/$target_name.config"; \
                else \
                    export CUDAQ_EXTERNAL_NVQIR_SIMS="$current_dir/libnvqir-$target_name.so;$current_dir/$target_name.config"; \
                fi \
        done \
    && echo "External NVQIR simulators: '$CUDAQ_EXTERNAL_NVQIR_SIMS'" \
    && echo "Building wheel for python${python_version}." \
    && cd cuda-quantum && python=python${python_version} \
    && $python -m pip install --no-cache-dir \
        cmake auditwheel \
        cuquantum-cu11==23.6.0 \
    && cuquantum_location=`$python -m pip show cuquantum-cu11 | grep -e 'Location: .*$'` \
    && export CUQUANTUM_INSTALL_PREFIX="${cuquantum_location#Location: }/cuquantum" \
    && ln -s $CUQUANTUM_INSTALL_PREFIX/lib/libcustatevec.so.1 $CUQUANTUM_INSTALL_PREFIX/lib/libcustatevec.so \
    && ln -s $CUQUANTUM_INSTALL_PREFIX/lib/libcutensornet.so.2 $CUQUANTUM_INSTALL_PREFIX/lib/libcutensornet.so \
    &&  SETUPTOOLS_SCM_PRETEND_VERSION=${CUDA_QUANTUM_VERSION:-0.0.0} \
        CUDAQ_BUILD_SELFCONTAINED=ON \
        CUDACXX="$CUDA_INSTALL_PREFIX/bin/nvcc" CUDAHOSTCXX=$CXX \
        $python -m build --wheel \
    && LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/_skbuild/lib:$workspace/assets/wheels_$(uname -m)/lib" \
        $python -m auditwheel -v repair dist/cuda_quantum-*linux_*.whl \
            --exclude libcustatevec.so.1 \
            --exclude libcutensornet.so.2 \
            --exclude libcublas.so.11 \
            --exclude libcublasLt.so.11 \
            --exclude libcusolver.so.11 \
            --exclude libcutensor.so.1 \
            --exclude libnvToolsExt.so.1 \ 
            --exclude libcudart.so.11.0 

FROM scratch
COPY --from=wheelbuild /cuda-quantum/wheelhouse/*manylinux*.whl . 
