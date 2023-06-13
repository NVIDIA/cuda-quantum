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
# minimal runtime environment. 
# A suitable dev image can be obtained by building docker/build/cudaqdev.Dockerfile.
#
# Usage:
# Must be built from the repo root with:
#   docker build -t ghcr.io/nvidia/cuda-quantum:latest -f docker/release/cudaq.Dockerfile .
# 
# The build argument dev_image defines the CUDA Quantum dev image to use, and the argument
# dev_tag defines the tag of that image.

ARG dev_image=nvidia/cuda-quantum-dev
ARG dev_tag=latest
FROM $dev_image:$dev_tag as cudaqbuild

# Unfortunately, there is no way to use the environment variables defined in the dev image
# to determine where to copy files from. See also e.g. https://github.com/moby/moby/issues/37345
# The rather ugly work around to achieve encapsulation is to make a copy here were we have
# access to the environment variables, so that the hardcoded paths in this file don't need to 
# match the paths in the dev image.
RUN if [ "$LLVM_INSTALL_PREFIX" != "/usr/local/llvm" ]; then mv "$LLVM_INSTALL_PREFIX" /usr/local/llvm; fi
RUN if [ "$CUDAQ_INSTALL_PREFIX" != "/usr/local/cudaq" ]; then mv "$CUDAQ_INSTALL_PREFIX" /usr/local/cudaq; fi
RUN mkdir -p /usr/local/cuquantum && \
    if [ "$CUQUANTUM_INSTALL_PREFIX" != "/usr/local/cuquantum" ] && [ -d "$CUQUANTUM_INSTALL_PREFIX" ]; then \
        mv "$CUQUANTUM_INSTALL_PREFIX"/* /usr/local/cuquantum; \
    fi
    
FROM ubuntu:22.04
SHELL ["/bin/bash", "-c"]
ENV SHELL=/bin/bash LANG=C.UTF-8 LC_ALL=C.UTF-8

# When a dialogue box would be needed during install, assume default configurations.
# Set here to avoid setting it for all install commands. 
# Given as arg to make sure that this value is only set during build but not in the launched container.
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates openssl wget git sudo vim \
    && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/* 

# Install CUDA Quantum runtime dependencies.

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip libpython3-dev \
        libstdc++-12-dev \
        libcurl4-openssl-dev libssl-dev \
    && apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir numpy \
    && ln -s /bin/python3 /bin/python

ENV CPLUS_INCLUDE_PATH="$CPLUS_INCLUDE_PATH:/usr/include/c++/11/:/usr/include/x86_64-linux-gnu/c++/11"

# Copy over the CUDA Quantum installation, and the necessary compiler tools.

ARG release_version=
ENV CUDA_QUANTUM_VERSION=$release_version
ENV CUDA_QUANTUM_PATH="/opt/nvidia/cudaq"

COPY --from=cudaqbuild "/usr/local/llvm/bin/clang++" "$CUDA_QUANTUM_PATH/llvm/bin/clang++"
COPY --from=cudaqbuild "/usr/local/llvm/lib/clang" "$CUDA_QUANTUM_PATH/llvm/lib/clang"
COPY --from=cudaqbuild "/usr/local/llvm/bin/llc" "$CUDA_QUANTUM_PATH/llvm/bin/llc"
COPY --from=cudaqbuild "/usr/local/llvm/bin/lld" "$CUDA_QUANTUM_PATH/llvm/bin/lld"
COPY --from=cudaqbuild "/usr/local/llvm/bin/ld.lld" "$CUDA_QUANTUM_PATH/llvm/bin/ld.lld"
COPY --from=cudaqbuild "/usr/local/cuquantum/" "$CUDA_QUANTUM_PATH/cuquantum/"
COPY --from=cudaqbuild "/usr/local/cudaq/" "$CUDA_QUANTUM_PATH"

ENV PATH "${PATH}:$CUDA_QUANTUM_PATH/bin"
ENV PYTHONPATH "${PYTHONPATH}:$CUDA_QUANTUM_PATH"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CUDA_QUANTUM_PATH/lib"

# Install additional runtime dependencies for optional components if present.

RUN if [ -n "$(ls -A $CUDA_QUANTUM_PATH/cuquantum)" ]; then \
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb \
        && dpkg -i cuda-keyring_1.0-1_all.deb \
        && apt-get update && apt-get install -y --no-install-recommends cuda-runtime-11-8 \
        && rm cuda-keyring_1.0-1_all.deb \
        && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/*; \
    fi

ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda-11.8/lib64:/usr/local/cuda-11.8/extras/CUPTI/lib64"

# For now, the CUDA Quantum build hardcodes certain paths and hence expects to find its 
# dependencies in specific locations. While a relocatable installation of CUDA Quantum should 
# be a good/better option in the future, for now we make sure to copy the dependencies to the 
# expected locations. The CUDQ Quantum installation contains an xml file that lists these.
RUN rdom () { local IFS=\> ; read -d \< E C ;} && \
    while rdom; do \
        if [ "$E" = "LLVM_INSTALL_PREFIX" ]; then \
            mkdir -p "$C" && mv "$CUDA_QUANTUM_PATH/llvm"/* "$C"; \
        elif [ "$E" = "CUQUANTUM_INSTALL_PREFIX" ] && [ -n "$(ls -A $CUDA_QUANTUM_PATH/cuquantum)" ]; then \
            mkdir -p "$C" && mv "$CUDA_QUANTUM_PATH/cuquantum"/* "$C"; \
        fi \
    done < "$CUDA_QUANTUM_PATH/build_config.xml"

# Include additional readmes and samples that are distributed with the image.

ADD ./docs/sphinx/examples/ /home/cudaq/examples/
ADD ./docker/release/README.md /home/cudaq/README.md

ARG COPYRIGHT_NOTICE="=========================\n\
   NVIDIA CUDA Quantum   \n\
=========================\n\n\
Version: ${CUDA_QUANTUM_VERSION}\n\n\
Copyright (c) 2023 NVIDIA Corporation & Affiliates \n\
All rights reserved.\n"
RUN echo -e "$COPYRIGHT_NOTICE" > "$CUDA_QUANTUM_PATH/Copyright.txt"
RUN echo 'cat "$CUDA_QUANTUM_PATH/Copyright.txt"' > /etc/profile.d/welcome.sh

# Create cudaq user

RUN useradd -m cudaq && echo "cudaq:cuda-quantum" | chpasswd && adduser cudaq sudo
RUN chown -R cudaq /home/cudaq && chgrp -R cudaq /home/cudaq

USER cudaq
WORKDIR /home/cudaq
ENTRYPOINT ["bash", "-l"]
