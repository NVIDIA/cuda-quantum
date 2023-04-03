# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Usage:
# Build from the repo root with
#   docker build -t ghcr.io/nvidia/cuda-quantum -f docker/build/cudaq.Dockerfile .
#
# If a custom build_environment is used, then the build environment must 
# 1) have all the necessary build dependendencies installed
# 2) define the LLVM_INSTALL_PREFIX environment variable indicating where the 
#    the LLVM binaries that CUDA Quantum depends on are installed
# 3) set the CC and CXX environment variable to use the same compiler toolchain
#    as the LLVM dependencies have been built with.

ARG build_environment=ghcr.io/nvidia/cuda-quantum-devdeps:llvm-latest
FROM build_environment as cudaqbuild
ADD ../../ /workspaces/cuda-quantum

# The CUDA Quantum build automatically detects whether GPUs are available and will 
# only include any GPU based components if they are. We use an argument to override
# this behavior and force building GPU components even if no GPU is detected. This is
# particularly useful for docker images since GPUs may not be accessible during build.
ARG force_compile_gpu_components=false
ARG configuration=Release

RUN CUDAQ_INSTALL_PREFIX=/opt/nvidia/cudaq FORCE_COMPILE_GPU_COMPONENTS=$force_compile_gpu_components \ 
    bash /workspaces/cuda-quantum/scripts/build_cudaq.sh -c $configuration -v

# Build a container with CUDA Quantum installed without any of the dev dependencies.
FROM ubuntu:22.04
ENV SHELL=/bin/bash LANG=C.UTF-8 LC_ALL=C.UTF-8

# When a dialogue box would be needed during install, assume default configurations.
# Set here to avoid setting it for all install commands. 
# Given as arg to make sure that this value is only set during build but not in the launched container.
ARG DEBIAN_FRONTEND=noninteractive
RUN apt update && apt-get install -y --no-install-recommends \
        ca-certificates openssl wget git sudo vim \
    && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/* 

# Install CUDA Quantum runtime dependencies.

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip libpython3-dev \
        libstdc++-11-dev \
        libcurl4-openssl-dev libssl-dev \
    && apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir numpy
    && ln -s /bin/python3 /bin/python \

ENV CPLUS_INCLUDE_PATH="$CPLUS_INCLUDE_PATH:/usr/include/c++/11/:/usr/include/x86_64-linux-gnu/c++/11"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda-11.8/lib64:/usr/local/cuda-11.8/extras/CUPTI/lib64"

# Copy over the CUDA Quantum installation, the necessary compiler tools, 
# as well as additional readmes and samples that are distributed with the image.

COPY --from=cudaqbuild /opt/llvm/bin/clang++ /opt/llvm/bin/clang++
COPY --from=cudaqbuild /opt/llvm/lib/clang /opt/llvm/lib/clang
COPY --from=cudaqbuild /opt/llvm/bin/llc /opt/llvm/bin/llc
COPY --from=cudaqbuild /opt/nvidia/cudaq /opt/nvidia/cudaq
COPY --from=cudaqbuild /workspaces/cuda-quantum/docs/sphinx/examples/ /home/cudaq/examples/
COPY --from=cudaqbuild /workspaces/cuda-quantum/docker/release/README.md /home/cudaq/README.md

ENV CUDA_QUANTUM_VERSION=0.3.0
ENV CUDA_QUANTUM_PATH=/opt/nvidia/cudaq
ENV PATH "${PATH}:/opt/nvidia/cudaq/bin:/opt/llvm/bin"
ENV PYTHONPATH "${PYTHONPATH}:/opt/nvidia/cudaq"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/nvidia/cudaq/lib"

# Create cudaq user

ARG COPYRIGHT_NOTICE="=========================\n\
   NVIDIA CUDA Quantum   \n\
=========================\n\n\
CUDA Quantum Version ${CUDA_QUANTUM_VERSION}\n\n\
Copyright (c) 2023 NVIDIA Corporation & Affiliates \n\
All rights reserved.\n"
RUN echo "$COPYRIGHT_NOTICE" > "$CUDA_QUANTUM_PATH/Copyright.txt"
RUN echo 'cat "$CUDA_QUANTUM_PATH/Copyright.txt"' > /etc/profile.d/welcome.sh

RUN useradd -m cudaq && echo "cudaq:cuda-quantum" | chpasswd && adduser cudaq sudo
RUN chown -R cudaq /home/cudaq && chgrp -R cudaq /home/cudaq

USER cudaq
WORKDIR /home/cudaq
ENTRYPOINT ["bash", "-l"]
