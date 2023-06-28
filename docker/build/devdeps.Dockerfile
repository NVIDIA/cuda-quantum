# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This file builds the development environment that contains the necessary development 
# dependencies for building and testing CUDA Quantum. This does not include the CUDA, OpenMPI 
# and other dependencies that some of the simulator backends require. These backends
# will be omitted from the build if this environment is used.
#
# Usage:
# Must be built from the repo root with:
#   docker build -t ghcr.io/nvidia/cuda-quantum-devdeps:${toolchain}-latest -f docker/build/devdeps.Dockerfile --build-arg toolchain=$toolchain .
#
# The variable $toolchain indicates which compiler toolchain to build the LLVM libraries with. 
# The toolchain used to build the LLVM binaries that CUDA Quantum depends on must be used to build
# CUDA Quantum. This image sets the CC and CXX environment variables to use that toolchain. 
# Currently, llvm (default), clang16, clang15, gcc12, and gcc11 are supported. To use a different 
# toolchain, add support for it to the install_toolchain.sh script. If the toolchain is set to llvm, 
# then the toolchain will be built from source.

# Build additional tools needed for CUDA Quantum documentation generation.
FROM ubuntu:22.04 as doxygenbuild
RUN apt-get update && apt-get install -y wget unzip make cmake flex bison gcc g++ python3 \
    && wget https://github.com/doxygen/doxygen/archive/9a5686aeebff882ebda518151bc5df9d757ea5f7.zip -q -O repo.zip \
    && unzip repo.zip && mv doxygen* repo && rm repo.zip \
    && cmake -G "Unix Makefiles" repo && cmake --build . --target install --config Release \
    && rm -rf repo && apt-get remove -y wget unzip make cmake flex bison gcc g++ python3 \
    && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/*

FROM ubuntu:22.04 as llvmbuild
SHELL ["/bin/bash", "-c"]

ARG llvm_commit
ARG toolchain=llvm

# When a dialogue box would be needed during install, assume default configurations.
# Set here to avoid setting it for all install commands. 
# Given as arg to make sure that this value is only set during build but not in the launched container.
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates openssl apt-utils \
    && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/* 

# Install prerequisites for building LLVM
RUN apt-get update && apt-get install -y --no-install-recommends \
        ninja-build cmake python3 \
    && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/* 

# Clone the LLVM source code
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && mkdir /llvm-project && cd /llvm-project && git init \
    && git remote add origin https://github.com/llvm/llvm-project \
    && git fetch origin --depth=1 $llvm_commit && git reset --hard FETCH_HEAD \
    && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/* 

# Build the the LLVM libraries and compiler toolchain needed to build CUDA Quantum;
# The safest option to avoid any compatibility issues is to build an application using these libraries 
# with the same compiler toolchain that the libraries were compiled with.
# Since the llvm libraries needed to build CUDA Quantum include the compiler toolchain, we can build 
# CUDA Quantum itself with that compiler as well. This is done when llvm is specified as the desired
# toolchain. For more information about compatibility between different C++ compilers, see e.g.
# - Itanium C++ ABI and C++ Standard Library implementations
# - https://libcxx.llvm.org/
# - https://clang.llvm.org/docs/MSVCCompatibility.html
# - https://clang.llvm.org/docs/Toolchain.html
# - https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html
# - https://gcc.gnu.org/onlinedocs/gcc/Code-Gen-Options.html#Code%20Gen%20Options
# - https://gcc.gnu.org/onlinedocs/gcc/C_002b_002b-Dialect-Options.html#C_002b_002b-Dialect-Options
ADD ./scripts/install_toolchain.sh /scripts/install_toolchain.sh
ADD ./scripts/build_llvm.sh /scripts/build_llvm.sh
RUN LLVM_INSTALL_PREFIX=/opt/llvm LLVM_SOURCE=/llvm-project \
        source scripts/install_toolchain.sh -e /opt/llvm/bootstrap -t ${toolchain}
RUN source /opt/llvm/bootstrap/init_command.sh && \
    LLVM_INSTALL_PREFIX=/opt/llvm \
        bash /scripts/build_llvm.sh -s /llvm-project -c Release -v \
    && rm -rf /llvm-project 

# We use a newer version of cmake that is only available via the Kitware apt repository.
FROM ubuntu:22.04 as cmakebuild
RUN apt-get update && apt-get install -y wget unzip make gcc g++ libssl-dev \
    && wget https://github.com/Kitware/CMake/releases/download/v3.26.3/cmake-3.26.3.zip -q \
    && unzip cmake-3.26.3.zip && rm cmake-3.26.3.zip \
    && cd cmake-3.26.3 && ./bootstrap --prefix=/usr/local/cmake-3.26/ \
    && make -j$(nproc) && make -j$(nproc) install && cd .. \
    && rm -rf cmake-3.26.3 && apt-get remove -y wget unzip make gcc g++ libssl-dev \
    && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/*

FROM ubuntu:22.04
SHELL ["/bin/bash", "-c"]

# When a dialogue box would be needed during install, assume default configurations.
# Set here to avoid setting it for all install commands. 
# Given as arg to make sure that this value is only set during build but not in the launched container.
ARG DEBIAN_FRONTEND=noninteractive
ENV HOME=/home SHELL=/bin/bash LANG=C.UTF-8 LC_ALL=C.UTF-8

# Copy over doxygen.
COPY --from=doxygenbuild /usr/local/bin/doxygen /usr/local/bin/doxygen
ENV PATH="${PATH}:/usr/local/bin"

# Copy over the llvm build dependencies.
COPY --from=llvmbuild /opt/llvm /opt/llvm
ENV LLVM_INSTALL_PREFIX=/opt/llvm
ENV PATH="$PATH:$LLVM_INSTALL_PREFIX/bin/"

# Install the C++ standard library. We could alternatively build libc++ 
# as part of the LLVM build and compile against that instead of libstdc++.
RUN apt-get update && apt-get install -y --no-install-recommends libstdc++-12-dev \
    && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install the C/C++ compiler toolchain with which the LLVM dependencies have
# been built. CUDA Quantum needs to be built with that same toolchain. We use
# a wrapper script so that the path that we set CC and CXX to is independent 
# on the installed toolchain. Unfortunately, a symbolic link won't work.
# Using update-alternatives for c++ and cc could maybe be a better option.
RUN source "$LLVM_INSTALL_PREFIX/bootstrap/init_command.sh" \
    && echo -e '#!/bin/bash\n"'$CC'" "$@"' > "$LLVM_INSTALL_PREFIX/bootstrap/cc" \
    && echo -e '#!/bin/bash\n"'$CXX'" "$@"' > "$LLVM_INSTALL_PREFIX/bootstrap/cxx" \
    && chmod +x "$LLVM_INSTALL_PREFIX/bootstrap/cc" \
    && chmod +x "$LLVM_INSTALL_PREFIX/bootstrap/cxx"
ENV CC="$LLVM_INSTALL_PREFIX/bootstrap/cc"
ENV CXX="$LLVM_INSTALL_PREFIX/bootstrap/cxx"

# Install additional dependencies required to build and test CUDA Quantum.
COPY --from=cmakebuild /usr/local/cmake-3.26/ /usr/local/cmake-3.26/
ENV PATH="${PATH}:/usr/local/cmake-3.26/bin"
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ninja-build libcurl4-openssl-dev libssl-dev \
        python3 python3-pip libpython3-dev \
        libblas-dev \
    && python3 -m pip install --no-cache-dir \
        lit pytest numpy \
        fastapi uvicorn pydantic llvmlite \
        openfermionpyscf \
    && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install additional tools for CUDA Quantum documentation generation.
RUN python3 -m pip install --no-cache-dir \
    sphinx==5.3.0 sphinx_rtd_theme==1.2.0 sphinx-reredirects==0.1.2 \
    enum-tools[sphinx] breathe==4.34.0 myst-parser==1.0.0
