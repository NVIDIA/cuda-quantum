# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This Dockerfile is the endpoint for the manylinux workflow.
# It will pull in the manylinux dependency image from `deps/Dockerfile`,
# then pulls down CUDA-Quantum and calls `scripts/build_wheel.sh`.

# Under construction: 
# We will have to run auditwheel on the output wheel to change 
# its distribution name to manylinux. Without much time to debug,
# this returns an error about missing the libcudaq-spin .so file.
# I belive this could be due to the fact that I've removed the static
# libz linking, so my next steps will be to add those back in and check
# again.

# DOCKER_BUILDKIT=1 docker build -t nvidia/cudaq_manylinux_build -f docker/wheel/Dockerfile . --output out
FROM docker.io/nvidia/cudaq_manylinux_deps:local as buildStage 

ARG workspace=.
ARG destination=cuda-quantum
ADD "$workspace" "$destination"

RUN cd cuda-quantum && bash scripts/build_wheel.sh && \
    if [ ! "$?" -eq "0" ]; then exit 1; fi
RUN cd cuda-quantum \
    && python3.10 docker/wheel/auditwheel -v repair dist/cuda_quantum-*-linux_x86_64.whl

# Use this with DOCKER_BUILDKIT=1
FROM scratch
COPY --from=buildStage /cuda-quantum/wheelhouse/*manylinux*x86_64.whl . 
