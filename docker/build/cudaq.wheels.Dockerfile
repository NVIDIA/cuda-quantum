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
#   DOCKER_BUILDKIT=1 docker build -f docker/build/cudaq.wheels.Dockerfile . --output out

ARG base_image=ghcr.io/nvidia/cuda-quantum-devdeps:manylinux
FROM $base_image as wheelbuild

ARG release_version=
ENV CUDA_QUANTUM_VERSION=$release_version
ENV SETUPTOOLS_SCM_PRETEND_VERSION=$CUDA_QUANTUM_VERSION

ARG workspace=.
ARG destination=cuda-quantum
ADD "$workspace" "$destination"

RUN installed_versions=$(for python in `ls /usr/local/bin/python*`; do \
        $python --version | cut -d ' ' -f 2 | egrep -o '^3\.10'; \
    done | sort -V) && \
    valid_version=$(for v in $installed_versions; do \
        comp=$(echo -e "$v\n3.8" | sort -V); \
        if [ "$comp" != "${comp#3.8}" ]; then echo python$v; fi \
    done) && \
    # We need a newer cmake version than what is available with dnf. 
    python3.10 -m pip install cmake; \
    cd cuda-quantum && \
    for python in $valid_version; do \
        echo "Building wheel for $python."; \
        CUDAQ_BUILD_SELFCONTAINED=ON LLVM_INSTALL_PREFIX="$LLVM_INSTALL_PREFIX" \
        $python -m build --wheel; \
        $python -m pip install auditwheel; \
        LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/_skbuild/lib" \
        $python -m auditwheel -v repair dist/cuda_quantum-*linux_*.whl; \
    done

FROM scratch
COPY --from=wheelbuild /cuda-quantum/wheelhouse/*manylinux*.whl . 
