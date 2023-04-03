# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This file builds an image that contains a CUDA Quantum installation and all necessary runtime 
# dependencies for using CUDA Quantum.
#
# This image requires specifing an image as argument that contains a CUDA Quantum installation
# along with its development dependencies. This file then copies that installation into a more
# minimal runtime environment. The installation location of CUDA Quantum in the dev image can be 
# defined by by passing the argument CUDAQ_INSTALL_PREFIX, and the installation location of the 
# LLVM dependencies by passing LLVM_INSTALL_PREFIX.
#
# Usage:
# Set the following environment variables on the docker build host:
# CUDA_QUANTUM_VERSION, CUDAQ_INSTALL_PREFIX, LLVM_INSTALL_PREFIX
#
# Then build the image from the repo root with:
#   docker build -t ghcr.io/nvidia/cuda-quantum:$CUDA_QUANTUM_VERSION -f docker/build/devdeps.Dockerfile . \
#   --build-arg CUDA_QUANTUM_VERSION --build-arg CUDAQ_INSTALL_PREFIX --build-arg LLVM_INSTALL_PREFIX \
#   --build-arg cuda_quantum_dev_image=$dev_image --build-arg dev_tag=$dev_tag
# 
# The variable $dev_image defines the CUDA Quantum dev image to use, and the variable $dev_tag defines
# the tag of that image.

ARG cuda_quantum_dev_image=nvidia/cuda-quantum-dev
ARG dev_tag=llvm-latest
FROM nvidia/cuda-quantum-dev:$dev_tag as cudaqbuild

FROM ubuntu:22.04
ENV SHELL=/bin/bash LANG=C.UTF-8 LC_ALL=C.UTF-8

ARG CUDA_QUANTUM_VERSION=0.3.0
ARG CUDAQ_INSTALL_PREFIX=/opt/nvidia/cudaq
ARG LLVM_INSTALL_PREFIX=/opt/llvm

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
    && python3 -m pip install --no-cache-dir numpy \
    && ln -s /bin/python3 /bin/python

ENV CPLUS_INCLUDE_PATH="$CPLUS_INCLUDE_PATH:/usr/include/c++/11/:/usr/include/x86_64-linux-gnu/c++/11"

# Copy over the CUDA Quantum installation, and the necessary compiler tools.

COPY --from=cudaqbuild "$LLVM_INSTALL_PREFIX/bin/clang++" "$LLVM_INSTALL_PREFIX/bin/clang++"
COPY --from=cudaqbuild "$LLVM_INSTALL_PREFIX/lib/clang" "$LLVM_INSTALL_PREFIX/lib/clang"
COPY --from=cudaqbuild "$LLVM_INSTALL_PREFIX/bin/llc" "$LLVM_INSTALL_PREFIX/bin/llc"
COPY --from=cudaqbuild "$CUDAQ_INSTALL_PREFIX" "$CUDAQ_INSTALL_PREFIX"

ENV CUDA_QUANTUM_VERSION=$CUDA_QUANTUM_VERSION
ENV CUDA_QUANTUM_PATH="$CUDAQ_INSTALL_PREFIX"
ENV PATH "${PATH}:$CUDAQ_INSTALL_PREFIX/bin:/opt/llvm/bin"
ENV PYTHONPATH "${PYTHONPATH}:$CUDAQ_INSTALL_PREFIX"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CUDAQ_INSTALL_PREFIX/lib"

# Include additional readmes and samples that are distributed with the image.

ADD ../../docs/sphinx/examples/ /home/cudaq/examples/
ADD ../../docker/release/README.md /home/cudaq/README.md

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
