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

ARG python_version=3.10
RUN echo "Building wheel for python${python_version}." \
    && cd cuda-quantum && python=python${python_version} \
    && $python -m pip install cmake auditwheel \
    && CUDAQ_BUILD_SELFCONTAINED=ON $python -m build --wheel \
    && LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/_skbuild/lib" \
        $python -m auditwheel -v repair dist/cuda_quantum-*linux_*.whl

FROM scratch
COPY --from=wheelbuild /cuda-quantum/wheelhouse/*manylinux*.whl . 
