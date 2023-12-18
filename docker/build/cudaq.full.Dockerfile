# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# [Operating System]
FROM amd64/almalinux:8
SHELL ["/bin/bash", "-c"]
ARG DEBIAN_FRONTEND=noninteractive
RUN dnf install -y --nobest --setopt=install_weak_deps=False \
        'dnf-command(config-manager)' && \
    dnf config-manager --enable powertools
ADD scripts/configure_build.sh /cuda-quantum/scripts/configure_build.sh

# [Prerequisites]
ARG PYTHON=python3.11
RUN dnf install -y --nobest --setopt=install_weak_deps=False ${PYTHON} glibc-static
RUN ${PYTHON} -m ensurepip && ${PYTHON} -m pip install numpy

# [Build Dependencies]
RUN dnf install -y --nobest --setopt=install_weak_deps=False \
        ${PYTHON}-devel perl-core \
        wget git unzip
RUN ${PYTHON} -m pip install \
        pytest lit fastapi uvicorn pydantic requests llvmlite \
        scipy==1.10.1 openfermionpyscf==0.5

## [CUDA]
RUN source /cuda-quantum/scripts/configure_build.sh install-cuda
## [cuQuantum]
RUN source /cuda-quantum/scripts/configure_build.sh install-cuquantum
## [cuTensor]
RUN source /cuda-quantum/scripts/configure_build.sh install-cutensor
## [Compiler Toolchain]
RUN source /cuda-quantum/scripts/configure_build.sh install-gcc

# [CUDA Quantum Build]
ADD scripts/configure_build.sh /cuda-quantum/scripts/configure_build.sh
ADD scripts/install_prerequisites.sh /cuda-quantum/scripts/install_prerequisites.sh
ADD scripts/install_toolchain.sh /cuda-quantum/scripts/install_toolchain.sh
ADD scripts/build_llvm.sh /cuda-quantum/scripts/build_llvm.sh
ADD .gitmodules /cuda-quantum/.gitmodules
ADD .git/modules/tpls/pybind11/HEAD /.git_modules/tpls/pybind11/HEAD
ADD .git/modules/tpls/llvm/HEAD /.git_modules/tpls/llvm/HEAD
ADD .git/modules/tpls/pybind11/refs /.git_modules/tpls/pybind11/refs
ADD .git/modules/tpls/llvm/refs /.git_modules/tpls/llvm/refs
ADD .git/modules/tpls/pybind11/packed-refs /.git_modules/tpls/pybind11/packed-refs
ADD .git/modules/tpls/llvm/packed-refs /.git_modules/tpls/llvm/packed-refs

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

ARG CUDA_QUANTUM_COMMIT=68844992ddc2f34c9678dd78dc4c98931eebf9fa
RUN rm -rf /cuda-quantum && \
    git clone --filter=tree:0 https://github.com/nvidia/cuda-quantum /cuda-quantum && \
    cd /cuda-quantum && git checkout ${CUDA_QUANTUM_COMMIT} && \
    rm -rf /cuda-quantum/scripts
ADD scripts /cuda-quantum/scripts
RUN cd /cuda-quantum && source scripts/configure_build.sh \
    ## [>CUDAQuantum]
    FORCE_COMPILE_GPU_COMPONENTS=true \
    CUDAQ_ENABLE_STATIC_LINKING=true \
    CUDAQ_WERROR=false \
    LD_LIBRARY_PATH="$CUTENSOR_INSTALL_PREFIX/lib:$LD_LIBRARY_PATH" \
    bash scripts/build_cudaq.sh -uv
    ## [<CUDAQuantum]

# [Build Artifacts]
RUN mkdir /artifacts && source /cuda-quantum/scripts/configure_build.sh && \
    #mv /usr/local/openssl artifacts \
    mv "${CUDAQ_INSTALL_PREFIX}" artifacts && \
    mv "${LLVM_INSTALL_PREFIX}" artifacts && \
    mv "${CUQUANTUM_INSTALL_PREFIX}" artifacts && \
    mv "${CUTENSOR_INSTALL_PREFIX}" artifacts

RUN git clone --filter=tree:0 https://github.com/megastep/makeself /makeself && \
    cd /makeself && git checkout release-2.5.0

RUN cd /makeself && \
    makeself.sh --gzip --license /cuda-quantum/LICENSE \
        /artifacts cuda_quantum \
        "CUDA Quantum toolkit for heterogeneous quantum-classical workflows" \
        startup_script [script_args]