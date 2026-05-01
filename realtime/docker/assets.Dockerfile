# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This file builds the CUDA Quantum Realtime binaries from scratch such that 
# they can be used on a range of Linux systems, provided the requirements documented in 
# the data center installation guide are satisfied.
#
# Usage:
# Must be built from the repo root with:
#   docker build -t ghcr.io/nvidia/cudaq-realtime-assets:amd64-cu12 -f realtime/docker/assets.Dockerfile .

# [Operating System]
ARG base_image=amd64/almalinux:8
FROM ${base_image} AS assets
SHELL ["/bin/bash", "-c"]
ARG cuda_version=13.0
ENV CUDA_VERSION=${cuda_version}

# When a dialogue box would be needed during install, assume default configurations.
# Set here to avoid setting it for all install commands. 
# Given as arg to make sure that this value is only set during build but not in the launched container.
ARG DEBIAN_FRONTEND=noninteractive
RUN dnf install -y --nobest --setopt=install_weak_deps=False \
        'dnf-command(config-manager)' && \
    dnf config-manager --enable powertools

ADD scripts/configure_build.sh /cuda-quantum/scripts/configure_build.sh

# [Prerequisites]
ARG PYTHON=python3.11
RUN dnf install -y --nobest --setopt=install_weak_deps=False ${PYTHON}

# [Build Dependencies]
RUN dnf install -y --nobest --setopt=install_weak_deps=False wget git unzip

## [CUDA]
RUN source /cuda-quantum/scripts/configure_build.sh install-cuda
## [Compiler Toolchain]
RUN source /cuda-quantum/scripts/configure_build.sh install-gcc

# [>ToolchainConfiguration]
ENV GCC_TOOLCHAIN="/opt/rh/gcc-toolset-12/root/usr/"
ENV CXX="${GCC_TOOLCHAIN}/bin/g++"
ENV CC="${GCC_TOOLCHAIN}/bin/gcc"
ENV CUDACXX=/usr/local/cuda/bin/nvcc
ENV CUDAHOSTCXX="${GCC_TOOLCHAIN}/bin/g++"
# [<ToolchainConfiguration]

## [nvComp] 
# For Hololink
RUN dnf -y install nvcomp pkgconfig

ENV PATH="${PATH}:/usr/local/cuda/bin" 

# [CMake]
# Hololink requires a newer CMake version
ARG CMAKE_VERSION=3.31.11
RUN wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-$(uname -m).sh -O cmake-install.sh && \
    bash cmake-install.sh --skip-licence --exclude-subdir --prefix=/usr/local
   
# [Holoscan SDK]
ARG HOLOSCAN_SDK_VERSION=4.0.0.1
ENV HOLOSCAN_SDK_INSTALL_PREFIX=/opt/nvidia/holoscan

RUN wget https://developer.download.nvidia.com/compute/holoscan/redist/holoscan/linux-$([ "$(uname -m)" == "aarch64" ] && echo sbsa || echo x86_64)/holoscan-linux-$([ "$(uname -m)" == "aarch64" ] && echo sbsa || echo x86_64)-${HOLOSCAN_SDK_VERSION}_cuda$(echo ${CUDA_VERSION} | cut -d . -f1)-archive.tar.xz -O holoscan.tar.xz && \
    mkdir -p $HOLOSCAN_SDK_INSTALL_PREFIX && \
    tar xf  holoscan.tar.xz --strip-components 1 -C $HOLOSCAN_SDK_INSTALL_PREFIX
    

# [DOCA]
ARG DOCA_VERSION=3.3.0
RUN wget https://www.mellanox.com/downloads/DOCA/DOCA_v${DOCA_VERSION}/host/doca-host-${DOCA_VERSION}-088000_26.01_rhel8.$(uname -m).rpm -O doca-host.rpm && \
    rpm -i doca-host.rpm && \
    dnf clean all && \
    dnf -y install epel-release && \
    crb enable && \
    dnf -y install doca-all doca-sdk-gpunetio doca-sdk-gpunetio-devel

## [CUDAQ Realtime Source]
ADD realtime /cuda-quantum/realtime

# [HSB]
ARG cuda_native_arg="80-real;90"
ENV CUDA_NATIVE_ARCH=${cuda_native_arg}
ARG hsb_version="2.6.0-EA2"
# Build HSB
RUN cd / && git clone -b ${hsb_version} https://github.com/nvidia-holoscan/holoscan-sensor-bridge.git && cd holoscan-sensor-bridge && \
    cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DHOLOLINK_BUILD_ONLY_NATIVE=OFF -DHOLOLINK_BUILD_PYTHON=OFF -DHOLOLINK_BUILD_TESTS=OFF -DHOLOLINK_BUILD_TOOLS=OFF -DHOLOLINK_BUILD_EXAMPLES=OFF -DHOLOLINK_BUILD_EMULATOR=OFF && \
    cmake --build build --target roce_receiver gpu_roce_transceiver hololink_core

# [CUDAQ Realtime]
RUN cd /cuda-quantum/realtime && \
    mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCUDAQ_REALTIME_BUILD_TESTS=ON -DCUDAQ_REALTIME_ENABLE_HOLOLINK_TOOLS=ON -DHOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR=/holoscan-sensor-bridge -DHOLOSCAN_SENSOR_BRIDGE_BUILD_DIR=/holoscan-sensor-bridge/build/ -DCMAKE_INSTALL_PREFIX=/realtime_assets && \
    make -j$(nproc) install

# [Install makeself]
RUN git clone --filter=tree:0 https://github.com/megastep/makeself /makeself && \
    cd /makeself && git checkout release-2.5.0 && \
    ln -s /makeself/makeself.sh /usr/local/bin/makeself && \
    ln -s /makeself/makeself-header.sh /usr/local/bin/makeself-header.sh

# [Build realtime installer]
# Set install prefix to match where build_installer.sh expects it
ENV CUDAQ_REALTIME_INSTALL_PREFIX=/realtime_assets
# Run installer build script
RUN bash /cuda-quantum/realtime/scripts/build_installer.sh -c $(echo $CUDA_VERSION | cut -d . -f1)   

FROM scratch
COPY --from=assets out/install_cuda_quantum_realtime_* .
