# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This file builds an image that contains a CUDA-Q installation and all necessary runtime 
# dependencies for using CUDA-Q.
#
# This image requires specifing an image as argument that contains a CUDA-Q installation
# along with its development dependencies. This file then copies that installation into a more
# minimal runtime environment. 
# A suitable base image can be obtained by building docker/build/cudaq.dev.Dockerfile.
#
# Usage:
# Must be built from the repo root with:
#   docker build -t nvcr.io/nvidia/nightly/cuda-quantum:latest-base -f docker/release/cudaq.Dockerfile .
# 
# The build argument cudaqdev_image defines the CUDA-Q dev image that contains the CUDA
# Quantum build. This Dockerfile copies the built components into the base_image. The specified
# base_image must contain the necessary CUDA-Q runtime dependencies.

ARG base_image=ubuntu:22.04
ARG cudaqdev_image=ghcr.io/nvidia/cuda-quantum-dev:latest
FROM $cudaqdev_image as cudaqbuild

# Unfortunately, there is no way to use the environment variables defined in the dev image
# to determine where to copy files from. See also e.g. https://github.com/moby/moby/issues/37345
# The rather ugly work around to achieve encapsulation is to make a copy here were we have
# access to the environment variables, so that the hardcoded paths in this file don't need to 
# match the paths in the dev image.
RUN mkdir /usr/local/cudaq_assets && cd /usr/local/cudaq_assets && \
    mkdir -p llvm/bin && mkdir -p llvm/lib && mkdir cuquantum && \
    mv "$LLVM_INSTALL_PREFIX/bin/"clang* "/usr/local/cudaq_assets/llvm/bin/" && rm -rf "/usr/local/cudaq_assets/llvm/bin/"clang-format* && \
    mv "$LLVM_INSTALL_PREFIX/lib/"clang* "/usr/local/cudaq_assets/llvm/lib/" && \
    mv "$LLVM_INSTALL_PREFIX/bin/llc" "/usr/local/cudaq_assets/llvm/bin/llc" && \
    mv "$LLVM_INSTALL_PREFIX/bin/lld" "/usr/local/cudaq_assets/llvm/bin/lld" && \
    mv "$LLVM_INSTALL_PREFIX/bin/ld.lld" "/usr/local/cudaq_assets/llvm/bin/ld.lld" && \
    if [ -d "$CUQUANTUM_INSTALL_PREFIX" ]; then mv "$CUQUANTUM_INSTALL_PREFIX"/* "/usr/local/cudaq_assets/cuquantum"; fi && \
    if [ "$CUDAQ_INSTALL_PREFIX" != "/usr/local/cudaq" ]; then mv "$CUDAQ_INSTALL_PREFIX" "/usr/local/cudaq"; fi

FROM $base_image
SHELL ["/bin/bash", "-c"]
ENV SHELL=/bin/bash LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV UCX_LOG_LEVEL=error

# When a dialogue box would be needed during install, assume default configurations.
# Set here to avoid setting it for all install commands. 
# Given as arg to make sure that this value is only set during build but not in the launched container.
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates wget git sudo vim \
    && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/* 

# Install CUDA-Q runtime dependencies.

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip libstdc++-12-dev \
    && apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir numpy \
    && ln -s /bin/python3 /bin/python
    RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ python3-dev \
    # Ref: https://github.com/qutip/qutip/issues/2412
    && python3 -m pip install --no-cache-dir notebook==7.1.3 "qutip<5" matplotlib \
    && apt-get remove -y gcc g++ python3-dev \
    && apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy over the CUDA-Q installation, and the necessary compiler tools.

ARG release_version=
ENV CUDA_QUANTUM_VERSION=$release_version
ENV CUDA_QUANTUM_PATH="/opt/nvidia/cudaq"

COPY --from=cudaqbuild "/usr/local/cudaq/" "$CUDA_QUANTUM_PATH"
COPY --from=cudaqbuild "/usr/local/cudaq_assets" "$CUDA_QUANTUM_PATH/assets"

# For now, the CUDA-Q build hardcodes certain paths and hence expects to find its 
# dependencies in specific locations. While a relocatable installation of CUDA-Q should 
# be a good/better option in the future, for now we make sure to copy the dependencies to the 
# expected locations. The CUDQ Quantum installation contains an xml file that lists these.
ADD ./scripts/migrate_assets.sh "$CUDA_QUANTUM_PATH/bin/migrate_assets.sh"
RUN bash "$CUDA_QUANTUM_PATH/bin/migrate_assets.sh" -s "$CUDA_QUANTUM_PATH/assets" \
    && rm -rf "$CUDA_QUANTUM_PATH/assets" \
    && rm "$CUDA_QUANTUM_PATH/bin/migrate_assets.sh"

ENV PATH "${PATH}:$CUDA_QUANTUM_PATH/bin"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CUDA_QUANTUM_PATH/lib"
ENV CPLUS_INCLUDE_PATH="$CPLUS_INCLUDE_PATH:$CUDA_QUANTUM_PATH/include"
# Better alternative to setting the PYTHONPATH, since the PYTHONPATH is generally not preserved when running as sudo.
RUN echo "$CUDA_QUANTUM_PATH" > /usr/local/lib/python$(python --version | egrep -o "([0-9]{1,}\.)+[0-9]{1,}" | cut -d '.' -f -2)/dist-packages/cudaq.pth

# Some tools related to shell handling.

ARG COPYRIGHT_NOTICE="=========================\n\
      NVIDIA CUDA-Q      \n\
=========================\n\n\
Version: ${CUDA_QUANTUM_VERSION}\n\n\
Copyright (c) 2024 NVIDIA Corporation & Affiliates \n\
All rights reserved.\n\n\
To run a command as administrator (user `root`), use `sudo <command>`.\n"
RUN echo -e "$COPYRIGHT_NOTICE" > "$CUDA_QUANTUM_PATH/Copyright.txt"
RUN echo 'cat "$CUDA_QUANTUM_PATH/Copyright.txt"' > /etc/profile.d/welcome.sh

# See also https://github.com/microsoft/vscode-remote-release/issues/4781
RUN env | egrep -v "^(HOME=|USER=|MAIL=|LC_ALL=|LS_COLORS=|LANG=|HOSTNAME=|PWD=|TERM=|SHLVL=|LANGUAGE=|_=)" \
        >> /etc/environment

# Create cudaq user

# Create new user `cudaq` with admin rights, and disable password and gecos, 
# see also https://askubuntu.com/a/1195288/635348.
RUN adduser --disabled-password --gecos '' cudaq && adduser cudaq sudo \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers \
    && mkdir -p /home/cudaq/.ssh && mkdir -p /var/run/sshd
ENV PATH="$PATH:/home/cudaq/.local/bin"

ADD ./docs/sphinx/examples/ /home/cudaq/examples/
ADD ./docker/release/README.md /home/cudaq/README.md
RUN mv /home/cudaq/examples/python/tutorials /home/cudaq/tutorials \
    && chown -R cudaq /home/cudaq && chgrp -R cudaq /home/cudaq

USER cudaq
WORKDIR /home/cudaq

# Run apt-get update to ensure that apt-get knows about CUDA packages
# if the base image has added the CUDA keyring.
# If we don't do that, then apt-get will get confused if some CUDA
# components are already installed but not all of them.
RUN sudo apt-get update

ENTRYPOINT ["bash", "-l"]
