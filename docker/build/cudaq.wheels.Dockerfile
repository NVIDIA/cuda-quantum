# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This Dockerfile is the endpoint for the manylinux workflow.

# Under construction: 
# We will have to run auditwheel on the output wheel to change 
# its distribution name to manylinux. Without much time to debug,
# this returns an error about missing the libcudaq-spin .so file.
# I belive this could be due to the fact that I've removed the static
# libz linking, so my next steps will be to add those back in and check
# again.

# DOCKER_BUILDKIT=1 docker build -t nvidia/cudaq_manylinux_build -f docker/wheel/Dockerfile . --output out
FROM docker.io/nvidia/cudaq_manylinux_deps:no-zlib as buildStage 

ARG release_version=
ENV CUDA_QUANTUM_VERSION=$release_version

ARG workspace=.
ARG destination=cuda-quantum
ADD "$workspace" "$destination"

RUN installed_versions=$(for python in `ls /usr/local/bin/python*`; do \
        $python --version | cut -d ' ' -f 2 | egrep -o '^3\.[0-9]+'; \
    done | sort -V) && \
    valid_version=$(for v in $installed_versions; do \
        comp=$(echo -e "$v\n3.8" | sort -V); \
        if [ "$comp" != "${comp#3.8}" ]; then echo python$v; fi \
    done) && \
    # We need a newer cmake version than what is available with dnf. 
    dnf remove -y cmake && python3.10 -m pip install cmake; \
    cd cuda-quantum && \
    for python in $valid_version; do \
        echo "Building wheel for $python."; \
        CUDAQ_BUILD_SELFCONTAINED=ON LLVM_INSTALL_PREFIX="$LLVM_INSTALL_PREFIX" \
        $python -m build --wheel; \
        $python -m pip install auditwheel; \
        $python docker/wheel/auditwheel -v repair dist/cuda_quantum-*-linux_*.whl; \
    done

# Use this with DOCKER_BUILDKIT=1
FROM scratch
COPY --from=buildStage /cuda-quantum/wheelhouse/*manylinux*.whl . 
