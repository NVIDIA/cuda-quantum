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

# Install prerequisites for building LLVM.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ninja-build cmake python3 \
    && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/* 

# Clone the LLVM source code.
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

# Todo: 
# - remove http://apt.llvm.org/jammy/ in the install_toolchain.sh and use
#   FROM llvmbuild as prereqs
# - uncomment the source /opt/llvm/bootstrap/init_command.sh below
FROM ubuntu:22.04 as prereqs
SHELL ["/bin/bash", "-c"]
COPY --from=llvmbuild /opt/llvm /opt/llvm
ADD ./scripts/install_prerequisites.sh /scripts/install_prerequisites.sh
ADD ./scripts/install_toolchain.sh /scripts/install_toolchain.sh
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && export LLVM_INSTALL_PREFIX=/opt/llvm \
    && export BLAS_INSTALL_PREFIX=/usr/local/blas \
    && export OPENSSL_INSTALL_PREFIX=/usr/local/openssl \
    # Making sure that anything that is build from source when installing additional
    # prerequisites is built using the same toolchain as CUDA Quantum by default.
    # && source /opt/llvm/bootstrap/init_command.sh \
    && bash /scripts/install_prerequisites.sh \
    && apt-get remove -y ca-certificates \
    && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/*

# Pre-built binaries for doxygen are only available for x86_64.
FROM ubuntu:22.04 as doxygenbuild
RUN if [ "$(uname -m)" != "x86_64" ]; then \
        apt-get update && apt-get install -y wget unzip make cmake flex bison gcc g++ python3 \
        # Fixed commit corresponding to release 1.9.7
        && wget https://github.com/doxygen/doxygen/archive/6a2ce4d18b5af1ca501bcf585e4c8e2b2b353b0f.zip -q -O repo.zip \
        && unzip repo.zip && mv doxygen* repo && rm repo.zip \
        && cmake -G "Unix Makefiles" repo && cmake --build . --target install --config Release \
        && rm -rf repo && apt-get remove -y wget unzip make cmake flex bison gcc g++ python3 \
        && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/*; \
    else \
        apt-get update && apt-get install --no-install-recommends -y wget ca-certificates \
        && wget https://www.doxygen.nl/files/doxygen-1.9.7.linux.bin.tar.gz \
        && tar xf doxygen-1.9.7* && mv doxygen-1.9.7/bin/* /usr/local/bin/ && rm -rf doxygen-1.9.7* \
        && apt-get remove -y wget ca-certificates \
        && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/*; \
    fi

FROM ubuntu:22.04
SHELL ["/bin/bash", "-c"]

# When a dialogue box would be needed during install, assume default configurations.
# Set here to avoid setting it for all install commands. 
# Given as arg to make sure that this value is only set during build but not in the launched container.
ARG DEBIAN_FRONTEND=noninteractive
ENV HOME=/home SHELL=/bin/bash LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0

# Copy over the llvm build dependencies.
COPY --from=llvmbuild /opt/llvm /opt/llvm
ENV LLVM_INSTALL_PREFIX=/opt/llvm
ENV PATH="$PATH:$LLVM_INSTALL_PREFIX/bin/"

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

# Install the C++ standard library. We could alternatively build libc++ 
# as part of the LLVM build and compile against that instead of libstdc++.
RUN apt-get update && apt-get install -y --no-install-recommends libstdc++-12-dev \
    && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy over additional prerequisites.
ENV BLAS_INSTALL_PREFIX=/usr/local/blas
ENV OPENSSL_INSTALL_PREFIX=/usr/local/openssl
ENV OPENSSL_ROOT_DIR="$OPENSSL_INSTALL_PREFIX"
COPY --from=prereqs /usr/local/blas "$BLAS_INSTALL_PREFIX"
COPY --from=prereqs /usr/local/openssl "$OPENSSL_INSTALL_PREFIX"

# Install additional tools for CUDA Quantum documentation generation.
COPY --from=doxygenbuild /usr/local/bin/doxygen /usr/local/bin/doxygen
ENV PATH="${PATH}:/usr/local/bin"
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip pandoc \
    && python3 -m pip install --no-cache-dir \
        ipython==8.15.0 pandoc==2.3 sphinx==5.3.0 sphinx_rtd_theme==1.2.0 sphinx-reredirects==0.1.2 \
        sphinx-copybutton==0.5.2 enum-tools[sphinx] breathe==4.34.0 nbsphinx==0.9.2 sphinx_gallery==0.13.0 \
     myst-parser==1.0.0
        

# Install additional dependencies required to build and test CUDA Quantum.
RUN apt-get update && apt-get install --no-install-recommends -y wget ca-certificates \
    && wget https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4-linux-$(uname -m).tar.gz \
    && tar xf cmake-3.26.4* && mv cmake-3.26.4-linux-$(uname -m)/ /usr/local/cmake-3.26/ && rm -rf cmake-3.26.4* \
    # NOTE: apt-get remove -y ca-certificates also remove python3-pip.
    && apt-get remove -y wget ca-certificates \
    && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/*
ENV PATH="${PATH}:/usr/local/cmake-3.26/bin"
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ninja-build libcurl4-openssl-dev \
        python3 python3-pip libpython3-dev \
    && python3 -m pip install --no-cache-dir \
        lit pytest numpy \
        fastapi uvicorn pydantic requests llvmlite \
        scipy==1.10.1 openfermionpyscf==0.5 \
    && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/*
