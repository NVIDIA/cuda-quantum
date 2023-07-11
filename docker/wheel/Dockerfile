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

# DOCKER_BUILDKIT=1 docker build -t nvidia/cudaq_manylinux_test . --output out
FROM nvidia/cudaq_manylinux_deps as buildStage 

ARG workspace=.
ARG destination=cuda-quantum
ADD "$workspace" "$destination"

RUN cd cuda-quantum \
    && export LLVM_DIR=/opt/llvm \
    && export CUDAQ_CPR_INSTALL=/cpr/install \
    && bash scripts/build_wheel.sh
#RUN cd cuda-quantum \
#    && python3.10 docker/wheel/auditwheel -v repair dist/cuda_quantum-0.3.0-cp310-cp310-linux_x86_64.whl

# Also in: _skbuild/linux-x86_64-3.10/cmake-build/lib/
# No longer needed (?): LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/cuda-quantum/_skbuild/linux-x86_64-3.10/cmake-install/lib/python3.10/site-packages/cudaq/lib/" 
# python3.10 docker/wheel/auditwheel -v repair dist/cuda_quantum-0.3.0-cp310-cp310-linux_x86_64.whl

# Use this with DOCKER_BUILDKIT=1
#FROM scratch AS exportStage 
#COPY --from=buildStage /cuda-quantum/dist/*.whl . 

# FIXME GETTING AN OUTPUT
# Fixed-up wheel written to /cuda-quantum/wheelhouse/cuda_quantum-0.3.0-cp310-cp310-manylinux_2_28_x86_64.whl

# TODO: 
# - check we can install it on the manylinux compatible oses
# - figure out what to do with the libgomp1 dependency
# - check that other simulators work as well...
# - find out if we can avoid the --user flag