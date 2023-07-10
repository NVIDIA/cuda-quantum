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

# DOCKER_BUILDKIT=1 docker build -t nvidia/cudaq_manylinux_test --network host . --output out
FROM nvidia/cudaq_manylinux_deps as buildStage 

# Pull down CUDA-Quantum to build the wheel off of.
RUN ls && ls && ls && git clone -b wheel https://github.com/NVIDIA/cuda-quantum \
    && cd cuda-quantum && git -c submodule.tpls/llvm.update=none submodule update --init --recursive \
    && rm -rf /root/.ssh 

RUN ls
RUN cd cuda-quantum \
    && export LLVM_DIR=/opt/llvm \
    && export CUDAQ_CPR_INSTALL=/cpr/install \
    && bash scripts/build_wheel.sh \
    && docker/wheel/auditwheel -v repair dist/cuda_quantum-0.0.3-cp310-cp310-linux_x86_64.whl
    # TODO: Pass the output wheel off to uploading. Might actually run this step from the file
    # that we actually test the wheel in, instead of at this point.

# Use this with DOCKER_BUILDKIT=1
FROM scratch AS exportStage 
COPY --from=buildStage /cuda-quantum/dist/*.whl . 