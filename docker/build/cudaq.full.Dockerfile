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
        'dnf-command(config-manager)'

# [Repo Content]
ADD scripts/configure_build.sh /cuda-quantum/scripts/configure_build.sh

# [Prerequisites]
ARG PYTHON=python3.11
RUN dnf install -y --nobest --setopt=install_weak_deps=False ${PYTHON}
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
ARG CUDA_QUANTUM_COMMIT=cec13c0429cb23ccbaa67a601d1ebd1fdf653611
RUN rm -rf /cuda-quantum && \
    git clone --filter=tree:0 https://github.com/nvidia/cuda-quantum /cuda-quantum && \
    cd /cuda-quantum && git checkout ${CUDA_QUANTUM_COMMIT} && \
    rm -rf /cuda-quantum/scripts
ADD scripts /cuda-quantum/scripts

ARG install_before_build=prereqs
RUN cd /cuda-quantum && source scripts/configure_build.sh install-$install_before_build
RUN cd /cuda-quantum && source scripts/configure_build.sh \
    ## [>CUDAQuantum]
    FORCE_COMPILE_GPU_COMPONENTS=true CUDAQ_WERROR=false \
    CUDAQ_BUILD_SELFCONTAINED=TRUE \
    LD_LIBRARY_PATH="$CUTENSOR_INSTALL_PREFIX/lib:$LD_LIBRARY_PATH" \
    bash scripts/build_cudaq.sh -uv
    ## [<CUDAQuantum]

