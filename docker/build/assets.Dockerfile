# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This file builds the CUDA Quantum binaries from scratch such that they can be
# used on a range of Linux systems, provided the requirements documented in 
# the data center installation guide are satisfied.
#
# Usage:
# Must be built from the repo root with:
#   docker build -t ghcr.io/nvidia/cuda-quantum-assets:latest -f docker/build/assets.Dockerfile .

# [Operating System]
ARG base_image=amd64/almalinux:8
FROM ${base_image} as prereqs
SHELL ["/bin/bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
RUN dnf install -y --nobest --setopt=install_weak_deps=False \
        'dnf-command(config-manager)' && \
    dnf config-manager --enable powertools
ADD scripts/configure_build.sh /cuda-quantum/scripts/configure_build.sh

# [Prerequisites]
ARG PYTHON=python3.11
RUN dnf install -y --nobest --setopt=install_weak_deps=False ${PYTHON} glibc-static

# [Build Dependencies]
RUN dnf install -y --nobest --setopt=install_weak_deps=False wget git unzip

## [CUDA]
RUN source /cuda-quantum/scripts/configure_build.sh install-cuda
## [Compiler Toolchain]
RUN source /cuda-quantum/scripts/configure_build.sh install-gcc

# [CUDA Quantum]
ADD scripts/configure_build.sh /cuda-quantum/scripts/configure_build.sh
ADD scripts/install_prerequisites.sh /cuda-quantum/scripts/install_prerequisites.sh
ADD scripts/install_toolchain.sh /cuda-quantum/scripts/install_toolchain.sh
ADD scripts/build_llvm.sh /cuda-quantum/scripts/build_llvm.sh
ADD .gitmodules /cuda-quantum/.gitmodules
ADD .git/modules/tpls/pybind11/HEAD /.git_modules/tpls/pybind11/HEAD
ADD .git/modules/tpls/llvm/HEAD /.git_modules/tpls/llvm/HEAD

# This is a hack so that we do not need to rebuild the prerequisites 
# whenever we pick up a new CUDA Quantum commit (which is always in CI).
ARG install_before_build=prereqs
RUN cd /cuda-quantum && git init && \
    git config -f .gitmodules --get-regexp '^submodule\..*\.path$' | \
    while read path_key local_path; do \
        if [ -f "/.git_modules/$local_path/HEAD" ]; then \
            url_key=$(echo $path_key | sed 's/\.path/.url/') && \
            url=$(git config -f .gitmodules --get "$url_key") && \
            git update-index --add --cacheinfo 160000 \
            $(cat /.git_modules/$local_path/HEAD) $local_path; \
        fi; \
    done && git submodule init && git submodule && \
    source scripts/configure_build.sh install-$install_before_build

FROM prereqs as build
# Checking out a CUDA Quantum commit is suboptimal, since the source code
# version must match this file. At the same time, adding the entire current
# directory will always rebuild CUDA Quantum, so instead just checking out
# the required folders here. 
ADD .git/index /cuda-quantum/.git/index
ADD .git/modules/ /cuda-quantum/.git/modules/

ADD "cmake" /cuda-quantum/cmake
ADD "docs/CMakeLists.txt" /cuda-quantum/docs/CMakeLists.txt
ADD "docs/sphinx/examples" /cuda-quantum/docs/sphinx/examples
ADD "docs/sphinx/snippets" /cuda-quantum/docs/sphinx/snippets
ADD "include" /cuda-quantum/include
ADD "lib" /cuda-quantum/lib
ADD "runtime" /cuda-quantum/runtime
ADD "scripts/build_cudaq.sh" /cuda-quantum/scripts/build_cudaq.sh
ADD "scripts/migrate_assets.sh" /cuda-quantum/scripts/migrate_assets.sh
ADD "scripts/cudaq_set_env.sh" /cuda-quantum/scripts/cudaq_set_env.sh
ADD "targettests" /cuda-quantum/targettests
ADD "test" /cuda-quantum/test
ADD "tools" /cuda-quantum/tools
ADD "tpls/customizations" /cuda-quantum/tpls/customizations
ADD "tpls/json" /cuda-quantum/tpls/json
ADD "unittests" /cuda-quantum/unittests
ADD "utils" /cuda-quantum/utils
ADD "CMakeLists.txt" /cuda-quantum/CMakeLists.txt
ADD "LICENSE" /cuda-quantum/LICENSE
ADD "NOTICE" /cuda-quantum/NOTICE

ARG release_version=
ENV CUDA_QUANTUM_VERSION=$release_version

RUN cd /cuda-quantum && source scripts/configure_build.sh && \
    ## [>CUDAQuantumBuild]
    CUDAQ_WERROR=false \
    CUDAQ_PYTHON_SUPPORT=OFF \
    CUDAHOSTCXX="$CXX" \
    CUDAQ_ENABLE_STATIC_LINKING=true \
    LDFLAGS='-static-libgcc -static-libstdc++' \
    LLVM_PROJECTS='clang;lld;mlir' \
    bash scripts/build_cudaq.sh -uv
    ## [<CUDAQuantumBuild]

## [Tests]
FROM build
RUN if [ ! -x "$(command -v nvidia-smi)" ] || [ -z "$(nvidia-smi | egrep -o "CUDA Version: ([0-9]{1,}\.)+[0-9]{1,}")" ]; then \
        excludes="--label-exclude gpu_required"; \
    fi && cd /cuda-quantum && \
    # FIXME: Disabled nlopt doesn't seem to work properly
    # tracked in https://github.com/NVIDIA/cuda-quantum/issues/1103
    excludes+=" --exclude-regex NloptTester|ctest-nvqpp|ctest-targettests" && \
    ctest --output-on-failure --test-dir build $excludes

ENV CUDAQ_CPP_STD="c++17"
ENV PATH="${PATH}:/usr/local/cuda/bin" 

RUN python3 -m ensurepip --upgrade && python3 -m pip install lit && \
    dnf install -y --nobest --setopt=install_weak_deps=False file which
RUN cd /cuda-quantum && source scripts/configure_build.sh && \
    "$LLVM_INSTALL_PREFIX/bin/llvm-lit" -v build/test \
        --param nvqpp_site_config=build/test/lit.site.cfg.py
