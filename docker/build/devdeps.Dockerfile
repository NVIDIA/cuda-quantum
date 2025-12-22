# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This file builds the development environment that contains the necessary development 
# dependencies for building and testing CUDA-Q. This does not include the CUDA, OpenMPI 
# and other dependencies that some of the simulator backends require. These backends
# will be omitted from the build if this environment is used.
#
# Usage:
# Must be built from the repo root with:
#   docker build -t ghcr.io/nvidia/cuda-quantum-devdeps:${toolchain}-latest -f docker/build/devdeps.Dockerfile --build-arg toolchain=$toolchain .
#
# The variable $toolchain indicates which compiler toolchain to build the LLVM libraries with. 
# The toolchain used to build the LLVM binaries that CUDA-Q depends on must be used to build
# CUDA-Q. This image sets the CC and CXX environment variables to use that toolchain. 
# Currently, clang16, clang15, gcc12, and gcc11 are supported. To use a different 
# toolchain, add support for it to the install_toolchain.sh script. If the toolchain is set to llvm, 
# then the toolchain will be built from source.

# [Operating System]
ARG base_image=ubuntu:24.04

# [CUDA-Q Dependencies]
FROM ${base_image} AS prereqs
SHELL ["/bin/bash", "-c"]
ARG toolchain=gcc11

# When a dialogue box would be needed during install, assume default configurations.
# Set here to avoid setting it for all install commands. 
# Given as arg to make sure that this value is only set during build but not in the launched container.
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/*

## [Prerequisites]
RUN apt-get update && apt-get install -y --no-install-recommends python3 && \
    apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/*

## [Environment Variables]
ENV CUDAQ_INSTALL_PREFIX=/usr/local/cudaq
ENV CUQUANTUM_INSTALL_PREFIX=/usr/local/cuquantum
ENV CUTENSOR_INSTALL_PREFIX=/usr/local/cutensor
ENV LLVM_INSTALL_PREFIX=/usr/local/llvm
ENV BLAS_INSTALL_PREFIX=/usr/local/blas
ENV ZLIB_INSTALL_PREFIX=/usr/local/zlib
ENV OPENSSL_INSTALL_PREFIX=/usr/local/openssl
ENV CURL_INSTALL_PREFIX=/usr/local/curl
ENV AWS_INSTALL_PREFIX=/usr/local/aws
# TODO: eliminate the need for this
ENV PIP_BREAK_SYSTEM_PACKAGES=1

## [Build Dependencies]
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget git unzip \
        python3-dev python3-pip && \
    python3 -m pip install --no-cache-dir numpy --break-system-packages && \
    apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/*
ADD scripts/install_toolchain.sh /cuda-quantum/scripts/install_toolchain.sh
RUN source /cuda-quantum/scripts/install_toolchain.sh \
        -e "$LLVM_INSTALL_PREFIX/bootstrap" -t ${toolchain}

## [Source Dependencies]
ADD scripts/install_prerequisites.sh /cuda-quantum/scripts/install_prerequisites.sh
ADD scripts/build_llvm.sh /cuda-quantum/scripts/build_llvm.sh
ADD cmake/caches/LLVM.cmake /cuda-quantum/cmake/caches/LLVM.cmake
ADD tpls/customizations/llvm /cuda-quantum/tpls/customizations/llvm
ADD .gitmodules /cuda-quantum/.gitmodules
ADD .git/modules/tpls/pybind11/HEAD /.git_modules/tpls/pybind11/HEAD
ADD .git/modules/tpls/llvm/HEAD /.git_modules/tpls/llvm/HEAD

# This is initializing the .git index sufficiently so that we can 
# check out the correct commits based on the submodule commit. 
RUN cd /cuda-quantum && git init && \
    git config -f .gitmodules --get-regexp '^submodule\..*\.path$' | \
    while read path_key local_path; do \
        if [ -f "/.git_modules/$local_path/HEAD" ]; then \
            url_key=$(echo $path_key | sed 's/\.path/.url/') && \
            url=$(git config -f .gitmodules --get "$url_key") && \
            git update-index --add --cacheinfo 160000 \
            $(cat /.git_modules/$local_path/HEAD) $local_path; \
        fi; \
    done && git submodule init && git submodule
# Build compiler-rt (only) since it is needed for code coverage tools
RUN LLVM_PROJECTS='clang;lld;mlir;python-bindings;compiler-rt' \
    bash /cuda-quantum/scripts/install_prerequisites.sh -t ${toolchain}

## [Dev Dependencies]
RUN if [ "$(uname -m)" == "x86_64" ]; then \
        # Pre-built binaries for doxygen are (only) available for x86_64.
        wget https://www.doxygen.nl/files/doxygen-1.9.7.linux.bin.tar.gz && \
        tar xf doxygen-1.9.7* && mv doxygen-1.9.7/bin/* /usr/local/bin/ && rm -rf doxygen-1.9.7*; \
    else \
        apt-get update && apt-get install -y --no-install-recommends make cmake flex bison g++ && \
        # Fixed commit corresponding to release 1.9.7
        wget https://github.com/doxygen/doxygen/archive/6a2ce4d18b5af1ca501bcf585e4c8e2b2b353b0f.zip -q -O repo.zip && \
        unzip repo.zip && mv doxygen* repo && rm repo.zip && \
        cmake -G "Unix Makefiles" repo && cmake --build . --target install --config Release && \
        rm -rf repo && apt-get remove -y make cmake flex bison g++ && \
        apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/*; \
    fi

# [CUDA-Q Dev Environment]
FROM ${base_image}
SHELL ["/bin/bash", "-c"]

# When a dialogue box would be needed during install, assume default configurations.
# Set here to avoid setting it for all install commands. 
# Given as arg to make sure that this value is only set during build but not in the launched container.
ARG DEBIAN_FRONTEND=noninteractive
ENV HOME=/home SHELL=/bin/bash LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0

# Copy over the llvm build dependencies.
COPY --from=prereqs /usr/local/llvm /usr/local/llvm
ENV LLVM_INSTALL_PREFIX=/usr/local/llvm
ENV PATH="$PATH:$LLVM_INSTALL_PREFIX/bin/"

# Install the C/C++ compiler toolchain with which the LLVM dependencies have
# been built. CUDA-Q needs to be built with that same toolchain. We use
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

# Copy over additional prerequisites.
ENV BLAS_INSTALL_PREFIX=/usr/local/blas
ENV ZLIB_INSTALL_PREFIX=/usr/local/zlib
ENV OPENSSL_INSTALL_PREFIX=/usr/local/openssl
ENV CURL_INSTALL_PREFIX=/usr/local/curl
ENV AWS_INSTALL_PREFIX=/usr/local/aws
COPY --from=prereqs /usr/local/blas "$BLAS_INSTALL_PREFIX"
COPY --from=prereqs /usr/local/zlib "$ZLIB_INSTALL_PREFIX"
COPY --from=prereqs /usr/local/openssl "$OPENSSL_INSTALL_PREFIX"
COPY --from=prereqs /usr/local/curl "$CURL_INSTALL_PREFIX"
COPY --from=prereqs /usr/local/aws "$AWS_INSTALL_PREFIX"

# Install additional dependencies required to build and test CUDA-Q.
RUN apt-get update && apt-get install --no-install-recommends -y wget ca-certificates \
    && wget https://github.com/Kitware/CMake/releases/download/v3.28.4/cmake-3.28.4-linux-$(uname -m).tar.gz \
    && tar xf cmake-3.28.4* && mv cmake-3.28.4-linux-$(uname -m)/ /usr/local/cmake-3.28/ && rm -rf cmake-3.28.4* \
    # NOTE: removing ca-certificates also remove python3-pip.
    && apt-get remove -y wget ca-certificates \
    && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/*
ENV PATH="${PATH}:/usr/local/cmake-3.28/bin"
RUN apt-get update && apt-get install -y --no-install-recommends \
        git gdb ninja-build file lldb \
        python3 python3-pip libpython3-dev \
    && python3 -m pip install --no-cache-dir --break-system-packages \
        lit==18.1.4 pytest==8.3.0 numpy==1.26.4 requests==2.31.0 \
        fastapi==0.111.0 uvicorn==0.29.0 pydantic==2.7.1 llvmlite==0.42.0 \
        pyspelling==2.10 pymdown-extensions==10.8.1 yapf \
        scipy==1.11.4 openfermionpyscf==0.5 h5py==3.12.1 \
    && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install additional tools for CUDA-Q documentation generation.
COPY --from=prereqs /usr/local/bin/doxygen /usr/local/bin/doxygen
ENV PATH="${PATH}:/usr/local/bin"
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip pandoc aspell aspell-en \
    && python3 -m pip install --no-cache-dir --break-system-packages \
        ipython==8.15.0 pandoc==2.3 sphinx==5.3.0 sphinx_rtd_theme==1.2.0 sphinx-reredirects==0.1.2 \
        sphinx-copybutton==0.5.2 sphinx_inline_tabs==2023.4.21 enum-tools[sphinx] breathe==4.34.0 \
        nbsphinx==0.9.2 sphinx_gallery==0.13.0 myst-parser==1.0.0 ipykernel==6.29.4 notebook==7.3.2 \
        ipywidgets==8.1.5 sphinx-tags==0.4
