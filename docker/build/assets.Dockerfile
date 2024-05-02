# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
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
RUN dnf install -y --nobest --setopt=install_weak_deps=False ${PYTHON}

# [Build Dependencies]
RUN dnf install -y --nobest --setopt=install_weak_deps=False wget git unzip

## [CUDA]
RUN source /cuda-quantum/scripts/configure_build.sh install-cuda
## [Compiler Toolchain]
RUN source /cuda-quantum/scripts/configure_build.sh install-gcc

# [CUDA-Q]
ADD scripts/configure_build.sh /cuda-quantum/scripts/configure_build.sh
ADD scripts/install_prerequisites.sh /cuda-quantum/scripts/install_prerequisites.sh
ADD scripts/install_toolchain.sh /cuda-quantum/scripts/install_toolchain.sh
ADD scripts/build_llvm.sh /cuda-quantum/scripts/build_llvm.sh
ADD cmake/caches/LLVM.cmake /cuda-quantum/cmake/caches/LLVM.cmake
ADD .gitmodules /cuda-quantum/.gitmodules
ADD .git/modules/tpls/pybind11/HEAD /.git_modules/tpls/pybind11/HEAD
ADD .git/modules/tpls/llvm/HEAD /.git_modules/tpls/llvm/HEAD

# This is a hack so that we do not need to rebuild the prerequisites 
# whenever we pick up a new CUDA-Q commit (which is always in CI).
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
RUN cd /cuda-quantum && source scripts/configure_build.sh && \
    CUDAHOSTCXX="$CXX" \
    LLVM_PROJECTS='clang;flang;lld;mlir;runtimes' \
    bash scripts/install_prerequisites.sh -t llvm

# Validate that the built toolchain and libraries have no GCC dependencies.
RUN source /cuda-quantum/scripts/configure_build.sh && \
    shared_libraries=$(find "${LLVM_INSTALL_PREFIX}" -name '*.so') && \
    executables=$(find "${LLVM_INSTALL_PREFIX}" -executable -type f) && \
    if [ -n "$(ldd ${shared_libraries} ${executables} | grep gcc)" ]; then \
        echo -e "\e[01;31mThe produced toolchain and libraries depend on GCC libraries.\e[0m" >&2; \
        exit 1; \
    fi

# Checking out a CUDA-Q commit is suboptimal, since the source code
# version must match this file. At the same time, adding the entire current
# directory will always rebuild CUDA-Q, so we instead just add only
# the necessary files for the build to each build stage.
ADD .git/index /cuda-quantum/.git/index
ADD .git/modules/ /cuda-quantum/.git/modules/

## [C++ support]
FROM prereqs as cpp_build
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
    # IMPORTANT:
    # Make sure that the variables and arguments configured here match
    # the ones in the install_prerequisites.sh invocation in the prereqs stage!
    ## [>CUDAQuantumBuild]
    CUDAQ_WERROR=false \
    CUDAQ_PYTHON_SUPPORT=OFF \
    CUDAHOSTCXX="$CXX" \
    LLVM_PROJECTS='clang;flang;lld;mlir;runtimes' \
    bash scripts/build_cudaq.sh -t llvm -v
    ## [<CUDAQuantumBuild]

## [Python support]
FROM prereqs as python_build
ADD "pyproject.toml" /cuda-quantum/pyproject.toml
ADD "python" /cuda-quantum/python
ADD "cmake" /cuda-quantum/cmake
ADD "include" /cuda-quantum/include
ADD "lib" /cuda-quantum/lib
ADD "runtime" /cuda-quantum/runtime
ADD "tools" /cuda-quantum/tools
ADD "tpls/customizations" /cuda-quantum/tpls/customizations
ADD "tpls/json" /cuda-quantum/tpls/json
ADD "utils" /cuda-quantum/utils
ADD "CMakeLists.txt" /cuda-quantum/CMakeLists.txt
ADD "LICENSE" /cuda-quantum/LICENSE
ADD "NOTICE" /cuda-quantum/NOTICE

ARG release_version=
ENV SETUPTOOLS_SCM_PRETEND_VERSION=$release_version

ARG PYTHON=python3.11
RUN dnf install -y --nobest --setopt=install_weak_deps=False ${PYTHON}-devel && \
    ${PYTHON} -m ensurepip --upgrade && \
    ${PYTHON} -m pip install numpy build auditwheel patchelf

RUN cd /cuda-quantum && source scripts/configure_build.sh && \
    # IMPORTANT:
    # Make sure that the invocation of the install_prerequisites.sh script here matches
    # the ones in the install_prerequisites.sh invocation in the prereqs stage!
    LLVM_INSTALL_PREFIX="$(mktemp -d)" && \
    ## [>CUDAQuantumPythonBuild]
    bash scripts/install_prerequisites.sh -t llvm && \
    CUDAHOSTCXX="$CXX" \
    python3 -m build --wheel
    ## [<CUDAQuantumPythonBuild]

# The '[a-z]*linux_[^\.]*' is meant to catch things like:
# - manylinux_2_28_x86_64,
# - manylinux_2_28_aarch64,
# - linux_x86_64, etc.
# If input is linux_<ARCH>, then choose manylinux_2_28_<ARCH> output
RUN echo "Patching up wheel using auditwheel..." && \
    ## [>CUDAQuantumWheel]
    CUDAQ_WHEEL="$(find . -name 'cuda_quantum*.whl')" && \
    MANYLINUX_PLATFORM="$(echo ${CUDAQ_WHEEL} | grep -o '[a-z]*linux_[^\.]*' | sed -re 's/^linux_/manylinux_2_28_/')" && \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$(pwd)/_skbuild/lib" \ 
    python3 -m auditwheel -v repair ${CUDAQ_WHEEL} \
        --plat ${MANYLINUX_PLATFORM} \
        --exclude libcublas.so.11 \
        --exclude libcublasLt.so.11 \
        --exclude libcusolver.so.11 \
        --exclude libcutensor.so.1 \
        --exclude libcutensornet.so.2 \
        --exclude libcustatevec.so.1 \
        --exclude libcudart.so.11.0 
    ## [<CUDAQuantumWheel]

## [Tests]
FROM cpp_build
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
        --param nvqpp_site_config=build/test/lit.site.cfg.py && \
    "$LLVM_INSTALL_PREFIX/bin/llvm-lit" -v build/targettests \
        --param nvqpp_site_config=build/targettests/lit.site.cfg.py


# Tests for the Python wheel are run post-installation.
COPY --from=python_build /wheelhouse /cuda_quantum/wheelhouse