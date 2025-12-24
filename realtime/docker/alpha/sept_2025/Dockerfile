# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Make sure cudaq-qclink-main.tar.gz is in cwd. 
# 
# run 
# docker build --network host -t cudaq-qclink-alpha . 
#

# ARG base_image=ghcr.io/nvidia/cuda-quantum-devdeps:ext-amd64-cu12.0-gcc11-main
ARG base_image=ghcr.io/nvidia/cuda-quantum-dev:ext-cu12.0-gcc11-main
ARG TARGETARCH=

FROM $base_image
WORKDIR /workspaces

ADD cudaq-qclink-main.tar.gz .

RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

RUN cd cudaq-qclink-main && mkdir build && cd build && cmake .. -G Ninja -DCUDAQ_QCLINK_INCLUDE_TESTS=ON \
  && LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH} ninja