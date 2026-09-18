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
# ninja-build comes from powertools, enabled above; the HSB build needs it.
RUN dnf install -y --nobest --setopt=install_weak_deps=False wget git unzip ninja-build

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
# For HSB
RUN dnf -y install nvcomp pkgconfig

ENV PATH="${PATH}:/usr/local/cuda/bin" 

# [CMake]
ARG CMAKE_VERSION=4.4.3
RUN wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-$(uname -m).sh -O cmake-install.sh && \
    bash cmake-install.sh --skip-license --exclude-subdir --prefix=/usr/local

# [Holoscan SDK]
# Installed by the code the dev containers use, so the two cannot drift. Copied on its own
# so that an edit elsewhere under realtime/ does not redo this, DOCA and the HSB build.
ADD realtime/scripts/deps_common.sh /cuda-quantum/realtime/scripts/deps_common.sh

# Empty ARG, never ENV: the script reads ${HOLOSCAN_SDK_VERSION:-...}, so blank keeps its
# default while a --build-arg overrides it. An ENV would win over the build arg.
ARG HOLOSCAN_SDK_VERSION=
ENV HOLOSCAN_SDK_INSTALL_PREFIX=/opt/nvidia/holoscan

RUN . /cuda-quantum/realtime/scripts/deps_common.sh && \
    cudaq_realtime_install_holoscan

# [DOCA]
# Registered against the repository doca-host would otherwise deliver offline, by the same
# code the dev containers use. epel and crb supply dependencies of doca-all.
ARG DOCA_VERSION=
RUN dnf -y install epel-release && \
    crb enable && \
    . /cuda-quantum/realtime/scripts/deps_common.sh && \
    cudaq_realtime_add_doca_repo && \
    dnf -y install doca-all doca-sdk-gpunetio doca-sdk-gpunetio-devel

## [CUDAQ Realtime Source]
ADD realtime /cuda-quantum/realtime
# Needed by realtime/unittests; the standalone build has no top-level cmake dir.
ADD cmake/modules/CUDAQGtestDiscovery.cmake /cuda-quantum/cmake/modules/CUDAQGtestDiscovery.cmake

# [HSB]
# Built by the code the dev containers use, which also derives the CUDA architectures
# from the toolkit here. HSB_ROOT stays an ENV: build_realtime.sh reads it below.
ENV HSB_ROOT=/holoscan-sensor-bridge
ARG CUDAQ_REALTIME_HSB_REPO=
ARG CUDAQ_REALTIME_HSB_REF=
RUN . /cuda-quantum/realtime/scripts/deps_common.sh && \
    cudaq_realtime_build_hsb

# [CUDAQ Realtime]
# Set install prefix to match where build_installer.sh expects it
ENV CUDAQ_REALTIME_INSTALL_PREFIX=/realtime_assets
RUN cd /cuda-quantum/realtime && \
    bash scripts/build_realtime.sh

# [Install makeself]
RUN git clone --filter=tree:0 https://github.com/megastep/makeself /makeself && \
    cd /makeself && git checkout release-2.5.0 && \
    ln -s /makeself/makeself.sh /usr/local/bin/makeself && \
    ln -s /makeself/makeself-header.sh /usr/local/bin/makeself-header.sh

# [Build realtime installer]
RUN bash /cuda-quantum/realtime/scripts/build_installer.sh -c $(echo $CUDA_VERSION | cut -d . -f1)   

FROM scratch
COPY --from=assets out/install_cuda_quantum_realtime_* .
