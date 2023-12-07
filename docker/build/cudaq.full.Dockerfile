# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

FROM amd64/almalinux:8
ARG DEBIAN_FRONTEND=noninteractive

# Build dependencies (can be removed after CUDA Quantum is built):
ARG PYTHON=python3.11
RUN dnf install -y --nobest --setopt=install_weak_deps=False \
        'dnf-command(config-manager)' \
        wget git unzip \
        ${PYTHON}-devel perl-core \
        # needed only for OpenMPI build
        autoconf libtool flex
RUN ${PYTHON} -m ensurepip && ${PYTHON} -m pip install \
        # needed only for building and running CUDA Quantum tests
        pytest lit fastapi uvicorn pydantic requests llvmlite

# Runtime dependencies of CUDA Quantum:
RUN dnf install -y --nobest --setopt=install_weak_deps=False ${PYTHON}
RUN ${PYTHON} -m pip install --no-cache-dir numpy \
        # optional dependencies for certain application libraries
        scipy==1.10.1 openfermionpyscf==0.5

# The following environment variables *must* be set during the build;
# Their value can be chosen freely, but with the build as it is currently,
# the path during the build needs to match the path during runtime.
# A fully self-contained build that is relocatable is not yet supported 
# for some component.
ENV CUDAQ_INSTALL_PREFIX=/usr/local/cudaq
ENV CUQUANTUM_INSTALL_PREFIX=/usr/local/cuquantum
ENV CUTENSOR_INSTALL_PREFIX=/usr/local/cutensor
ENV LLVM_INSTALL_PREFIX=/usr/local/llvm
ENV OPENSSL_INSTALL_PREFIX=/usr/local/openssl
ENV BLAS_PATH=/usr/local/blas
ENV MPI_PATH=/usr/local/openmpi
ENV MPI_HOME="${MPI_PATH}"

# CUDA dependencies:
# The environment variables DISTRIBUTION and CUDA_ARCH_FOLDER are used to download
# the correct CUDA and cuQuantum packages. Update them as needed for your platform.
# Go to https://developer.download.nvidia.com/compute/cuda/repos/ to find a 
# distribution and architecture that matches your platform.
ARG CUDA_VERSION=11.8
ENV DISTRIBUTION=rhel8 CUDA_ARCH_FOLDER=x86_64
RUN dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/${DISTRIBUTION}/${CUDA_ARCH_FOLDER}/cuda-${DISTRIBUTION}.repo
RUN dnf install -y --nobest --setopt=install_weak_deps=False \
        cuda-toolkit-$(echo ${CUDA_VERSION} | tr . -)

# cuQuantum dependencies:
# Each version of CUDA Quantum is compatible only with a specific cuQuantum version.
# Please check the CUDA Quantum documentation or release notes to confirm that the 
# installed cuQuantum version is compatible with the CUDA Quantum version you want to build.
ARG CUQUANTUM_VERSION=23.10.0.6
RUN cuquantum_archive=cuquantum-linux-${CUDA_ARCH_FOLDER}-${CUQUANTUM_VERSION}_cuda$(echo ${CUDA_VERSION} | cut -d . -f1)-archive.tar.xz; \
    wget "https://developer.download.nvidia.com/compute/cuquantum/redist/cuquantum/linux-${CUDA_ARCH_FOLDER}/${cuquantum_archive}"; \
    mkdir -p "$CUQUANTUM_INSTALL_PREFIX" && tar xf "${cuquantum_archive}" --strip-components 1 -C "$CUQUANTUM_INSTALL_PREFIX" && rm -rf "${cuquantum_archive}"

# cuTensor dependencies:
# The cuTensor library is not included in the CUDA installation above and is needed to
# build some of the simulator backends. Please check the cuQuantum documentation to ensure
# you choose a version that is compatible with the used cuQuantum version.
ARG CUTENSOR_VERSION=1.7.0.1
RUN cutensor_archive=libcutensor-linux-${CUDA_ARCH_FOLDER}-${CUTENSOR_VERSION}-archive.tar.xz; \
    wget "https://developer.download.nvidia.com/compute/cutensor/redist/libcutensor/linux-${CUDA_ARCH_FOLDER}/${cutensor_archive}"; \
    mkdir -p "$CUTENSOR_INSTALL_PREFIX" && tar xf "${cutensor_archive}" --strip-components 1 -C "$CUTENSOR_INSTALL_PREFIX"; \
    mv "$CUTENSOR_INSTALL_PREFIX"/lib/$(echo ${CUDA_VERSION} | cut -d . -f1)/* $CUTENSOR_INSTALL_PREFIX/lib/; \
    ls -d $CUTENSOR_INSTALL_PREFIX/lib/*/ | xargs rm -rf && rm -rf "${cutensor_archive}"

# Compiler toolchain:
# We use GCC 11 to build CUDA Quantum and its LLVM/MLIR dependencies.
# The compiler toolchain used for the build needs to support C++20 and
# be a supported host compiler for the installed CUDA version.
ARG GCC_VERSION=11
RUN dnf install -y --nobest --setopt=install_weak_deps=False \
        gcc-toolset-${GCC_VERSION}

# Build environment:
# Independent on which compiler toolchain you installed, the environment
# variables CC and CXX *must* be set for the CUDA Quantum build.
# If the CUDA compiler is not found when building CUDA Quantum, some
# components and backends will be omitted. To use GPU-acceleration in 
# CUDA Quantum, make sure to set CUDACXX to your CUDA compiler.
# A Fortran compiler is needed (only) to build the OpenSSL dependency.
ENV CXX=/opt/rh/gcc-toolset-11/root/usr/bin/g++
ENV CC=/opt/rh/gcc-toolset-11/root/usr/bin/gcc
ENV FC=/opt/rh/gcc-toolset-11/root/usr/bin/gfortran
ENV CUDACXX=/usr/local/cuda-${CUDA_VERSION}/bin/nvcc

# MPI dependencies:
# To work with all CUDA Quantum backends, a CUDA-aware OpenMPI installation
# is required. The following installation should be taken as giving an outline
# for a minimal installation that works, but to make best use of MPI, we
# recommend a more fully featured installation that fits your system.
# CUDA Quantum supports different MPI implementations via plugin support.
# A plugin for OpenMPI and MPICH is included with CUDA Quantum. To use a different
# MPI implementation, you will need to implement the plugin as defined in our documentation.
ARG OPENMPI_VERSION=4.1.4
RUN mkdir /openmpi-project && cd /openmpi-project && \
    git init && git remote add origin https://github.com/open-mpi/ompi && \
    git fetch origin --depth=1 v${OPENMPI_VERSION} && git reset --hard FETCH_HEAD
RUN cd /openmpi-project && ./autogen.pl && \
    CUDA_PATH=/usr/local/cuda-${CUDA_VERSION} && \
    PATH="$(dirname $CC):$PATH" \
    LDFLAGS=-Wl,--as-needed \
    ./configure --prefix="$MPI_PATH" \
                --disable-getpwuid --disable-static \
                --disable-debug --disable-mem-debug --disable-event-debug \
                --disable-mem-profile --disable-memchecker \
                --enable-mpi-fortran=none \
                --without-verbs \
                --with-cuda="$CUDA_PATH" && \
    make -j$(nproc) && make -j$(nproc) install && \
    cd / && rm -rf /openmpi-project

# CUDA Quantum build from source:
# This file is written for a specific version/commit of CUDA Quantum. 
# Make sure to checkout the commit that contains the version of this file you are using.
# The CUDA Quantum build will compile or omit optional components automatically depending
# on whether the necessary pre-requisites are found in the build environment.
# Please check the build log to confirm that all desired components have been built.
ARG CUDA_QUANTUM_COMMIT=fa4ac125bef8bfd65861724e20ceecf7c15389a5
RUN git clone --filter=tree:0 https://github.com/nvidia/cuda-quantum /cuda-quantum && \
    cd /cuda-quantum && git checkout ${CUDA_QUANTUM_COMMIT}
RUN cd /cuda-quantum && \
    FORCE_COMPILE_GPU_COMPONENTS=true CUDAQ_WERROR=false \
    bash scripts/build_cudaq.sh -u -v
    # && $CUQUANTUM_INSTALL_PREFIX/distributed_interfaces/ && bash activate_mpi.sh

ENV PATH="${CUDAQ_INSTALL_PREFIX}/bin:${PATH}"
ENV PYTHONPATH="${CUDAQ_INSTALL_PREFIX}:${PYTHONPATH}"

# Removing build dependencies
RUN ${PYTHON} -m pip uninstall -y pytest lit fastapi uvicorn pydantic requests llvmlite
RUN dnf remove -y wget git 'dnf-command(config-manager)' ${PYTHON}-devel perl-core autoconf libtool flex

# Maybe not needed?
#ENV CUQUANTUM_PATH="${CUQUANTUM_INSTALL_PREFIX}"
#ENV CUTENSOR_PATH="${CUTENSOR_INSTALL_PREFIX}"
#ENV LLVM_PATH="${LLVM_INSTALL_PREFIX}"
#ENV OPENSSL_PATH="${OPENSSL_INSTALL_PREFIX}"
#ENV CUDA_QUANTUM_PATH="$CUDAQ_INSTALL_PREFIX"
