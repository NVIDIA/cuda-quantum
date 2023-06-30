# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# DOCKER_BUILDKIT=1 docker build -t nvidia/cudaq_manylinux_test --network host . --output out
FROM nvidia/cudaq_manylinux_deps as buildStage 

# GET cudaq
RUN ls && ls && ls && git clone -b wheel https://github.com/NVIDIA/cuda-quantum \
    && cd cuda-quantum && git -c submodule.tpls/llvm.update=none submodule update --init --recursive \
    && rm -rf /root/.ssh 

RUN ls
RUN cd cuda-quantum  
    && export LLVM_DIR=/opt/llvm \
    && export CUDAQ_CPR_INSTALL=/cpr/install \
    && python3.10 -m pip install pytest auditwheel scikit-build \
    && bash scripts/build_wheel.sh 
    && docker/wheel/auditwheel -v repair dist/cuda_quantum-0.0.3-cp310-cp310-linux_x86_64.whl

# Use this with DOCKER_BUILDKIT=1
FROM scratch AS exportStage 
COPY --from=buildStage /cuda-quantum/dist/*.whl . 