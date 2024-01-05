# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This file builds a self-extractable CUDA Quantum archive that can be installed
# on a compatible Linux host system; see also https://makeself.io/.
#
# Usage:
# Must be built from the repo root with:
#   DOCKER_BUILDKIT=1 docker build -f docker/release/installer.Dockerfile . --output out

# [Operating System]
ARG base_image=amd64/almalinux:8
FROM ${base_image} as cudaqbuild
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
## [cuQuantum]
RUN source /cuda-quantum/scripts/configure_build.sh install-cuquantum
## [cuTensor]
RUN source /cuda-quantum/scripts/configure_build.sh install-cutensor
## [Compiler Toolchain]
RUN source /cuda-quantum/scripts/configure_build.sh install-gcc

# [CUDA Quantum]
ADD scripts/configure_build.sh /cuda-quantum/scripts/configure_build.sh
ADD scripts/install_prerequisites.sh /cuda-quantum/scripts/install_prerequisites.sh
ADD scripts/install_toolchain.sh /cuda-quantum/scripts/install_toolchain.sh
ADD scripts/build_llvm.sh /cuda-quantum/scripts/build_llvm.sh
ADD .gitmodules /cuda-quantum/.gitmodules
ADD .git/modules/tpls/pybind11/HEAD /.git_modules/tpls/pybind11/HEAD
ADD .git/modules/tpls/pybind11/refs /.git_modules/tpls/pybind11/refs
ADD .git/modules/tpls/llvm/HEAD /.git_modules/tpls/llvm/HEAD
ADD .git/modules/tpls/llvm/refs /.git_modules/tpls/llvm/refs

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
            $(cat /.git_modules/$local_path/HEAD) $local_path && \
            mv /.git_modules/$local_path/refs .git/modules/$local_path/refs && \
            mv /.git_modules/$local_path/packed-refs .git/modules/$local_path/packed-refs; \
        fi; \
    done && git submodule init && \
    source scripts/configure_build.sh install-$install_before_build

# Checking out a CUDA Quantum commit is suboptimal, since the source code
# version must match this file. At the same time, adding the entire current
# directory will always rebuild CUDA Quantum, so instead just checking out
# the required folders here. 
ARG CUDA_QUANTUM_REPO=.
ADD .git/index /cuda-quantum/.git/index
ADD .git/modules/ /cuda-quantum/.git/modules/

ADD "$CUDAQ_REPO_ROOT/cmake" /cuda-quantum/cmake
ADD "$CUDAQ_REPO_ROOT/docs/CMakeLists.txt" /cuda-quantum/docs/CMakeLists.txt
ADD "$CUDAQ_REPO_ROOT/docs/sphinx/examples" /cuda-quantum/docs/sphinx/examples
ADD "$CUDAQ_REPO_ROOT/docs/sphinx/snippets" /cuda-quantum/docs/sphinx/snippets
ADD "$CUDAQ_REPO_ROOT/include" /cuda-quantum/include
ADD "$CUDAQ_REPO_ROOT/lib" /cuda-quantum/lib
ADD "$CUDAQ_REPO_ROOT/runtime" /cuda-quantum/runtime
ADD "$CUDAQ_REPO_ROOT/scripts/build_cudaq.sh" /cuda-quantum/scripts/build_cudaq.sh
ADD "$CUDAQ_REPO_ROOT/targettests" /cuda-quantum/targettests
ADD "$CUDAQ_REPO_ROOT/test" /cuda-quantum/test
ADD "$CUDAQ_REPO_ROOT/tools" /cuda-quantum/tools
ADD "$CUDAQ_REPO_ROOT/tpls/customizations" /cuda-quantum/tpls/customizations
ADD "$CUDAQ_REPO_ROOT/tpls/json" /cuda-quantum/tpls/json
ADD "$CUDAQ_REPO_ROOT/unittests" /cuda-quantum/unittests
ADD "$CUDAQ_REPO_ROOT/utils" /cuda-quantum/utils
ADD "$CUDAQ_REPO_ROOT/CMakeLists.txt" /cuda-quantum/CMakeLists.txt
ADD "$CUDAQ_REPO_ROOT/LICENSE" /cuda-quantum/LICENSE
ADD "$CUDAQ_REPO_ROOT/NOTICE" /cuda-quantum/NOTICE

ARG release_version=
ENV CUDA_QUANTUM_VERSION=$release_version

RUN cd /cuda-quantum && source scripts/configure_build.sh && \
    ## [>CUDAQuantumBuild]
    CUDAQ_PYTHON_SUPPORT=OFF CUDAQ_WERROR=false \
    CUDAQ_ENABLE_STATIC_LINKING=true \
    LDFLAGS='-static-libgcc -static-libstdc++' \
    LLVM_PROJECTS='clang;lld;mlir' \
    bash scripts/build_cudaq.sh -uv
    ## [<CUDAQuantumBuild]

## [Build Tests]
RUN if [ ! -x "$(command -v nvidia-smi)" ] || [ -z "$(nvidia-smi | egrep -o "CUDA Version: ([0-9]{1,}\.)+[0-9]{1,}")" ]; then \
        excludes="--label-exclude gpu_required"; \
    fi && cd /cuda-quantum && \
    ctest --output-on-failure --test-dir build -E ctest-nvqpp $excludes
# FIXME: Not yet working due to failure to find span
#RUN python3 -m ensurepip --upgrade && python3 -m pip install lit && \
#    cd /cuda-quantum && source scripts/configure_build.sh && \
#    "$LLVM_INSTALL_PREFIX/bin/llvm-lit" -v --param nvqpp_site_config=build/test/lit.site.cfg.py build/test

# [Build Assets]
ADD "$CUDAQ_REPO_ROOT/scripts/migrate_assets.sh" /cuda-quantum/scripts/migrate_assets.sh
RUN source /cuda-quantum/scripts/configure_build.sh && \
    archive=/cuda_quantum && mkdir -p "${archive}" && \
    cp "/cuda-quantum/scripts/migrate_assets.sh" "${archive}/install.sh" && \
    mv "${CUDAQ_INSTALL_PREFIX}/build_config.xml" "${archive}/build_config.xml" && \
    mv "${CUDAQ_INSTALL_PREFIX}" "${archive}" && \
    mv "${CUQUANTUM_INSTALL_PREFIX}" "${archive}" && \
    mv "${CUTENSOR_INSTALL_PREFIX}" "${archive}" && \
    mkdir -p "${archive}/llvm/bin" && mkdir -p "${archive}/llvm/lib" && \
    mv "${LLVM_INSTALL_PREFIX}/bin/"clang* "${archive}/llvm/bin/" && rm -rf "${archive}/llvm/bin/"clang-format* && \
    mv "${LLVM_INSTALL_PREFIX}/lib/"clang* "${archive}/llvm/lib/" && \
    mv "${LLVM_INSTALL_PREFIX}/bin/llc" "${archive}/llvm/bin/llc" && \
    mv "${LLVM_INSTALL_PREFIX}/bin/lld" "${archive}/llvm/bin/lld" && \
    mv "${LLVM_INSTALL_PREFIX}/bin/ld.lld" "${archive}/llvm/bin/ld.lld"

RUN git clone --filter=tree:0 https://github.com/megastep/makeself /makeself && \
    cd /makeself && git checkout release-2.5.0 && \
    ./makeself.sh --gzip --license /cuda-quantum/LICENSE \
        /cuda_quantum cuda_quantum.$(uname -m) \
        "CUDA Quantum toolkit for heterogeneous quantum-classical workflows" \
        ./install.sh

FROM scratch
COPY --from=cudaqbuild /makeself/cuda_quantum.* . 
