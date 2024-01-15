# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This file builds a self-extractable CUDA Quantum archive that can be installed
# on a compatible Linux host system; see also https://makeself.io/.
# A suitable base image can be obtained by building docker/build/assets.Dockerfile.
#
# Usage:
# Must be built from the repo root with:
#   DOCKER_BUILDKIT=1 docker build -f docker/release/installer.Dockerfile . --output out

ARG base_image=ghcr.io/nvidia/cuda-quantum-assets:amd64-gcc11-main
ARG additional_components=none

FROM $base_image as additional_components_none
ONBUILD RUN echo "No additional components included."
FROM $base_image as additional_components_assets
ONBUILD COPY assets /assets
RUN source /cuda-quantum/scripts/configure_build.sh && \
    for folder in `find /assets/*$(uname -m)/* -maxdepth 0 -type d`; \
    do bash /cuda-quantum/scripts/migrate_assets.sh -s "$folder" && rm -rf "$folder"; \
    done

# [Installer]
FROM additional_components_${additional_components} as assets
RUN git clone --filter=tree:0 https://github.com/megastep/makeself /makeself && \
    cd /makeself && git checkout release-2.5.0

## [Installation Scripts]
RUN cp /cuda-quantum/scripts/migrate_assets.sh install.sh && \
    # Note: Generally, the idea is to set the necessary environment variables
    # to make CUDA Quantum discoverable in login shells and for all users. 
    # Non-login shells should inherit them from the original login shell. 
    # If we cannot modify /etc/profile, we instead modify ~/.bashrc, which 
    # is always executed by all interactive non-login shells.
    # The reason for this is that bash is a bit particular when it comes to user
    # level profiles for login-shells in the sense that there isn't one specific
    # file that is guaranteed to execute; it first looks for .bash_profile, 
    # then for .bash_login and .profile, and *only* the first file it finds is 
    # executed. Hence, the reliable and non-disruptive way to configure 
    # environment variables at the user level is to instead edit .bashrc.
    echo -e '\n\n\
    . "${CUDA_QUANTUM_PATH}/set_env.sh" \n\
    if [ -f /etc/profile ] && [ -w /etc/profile ]; then \n\
        echo "Configuring CUDA Quantum environment variables in /etc/profile." \n\
        sed "/^\s*\(#\|$\)/d" "${CUDA_QUANTUM_PATH}/set_env.sh" | sed "/^CUDAQ_INSTALL_PATH=.*/ s@@CUDAQ_INSTALL_PATH=${CUDA_QUANTUM_PATH}@" >> /etc/profile \n\
    else \n\
        echo "Configuring CUDA Quantum environment variables in ~/.bashrc." \n\
        sed "/^\s*\(#\|$\)/d" "${CUDA_QUANTUM_PATH}/set_env.sh" | sed "/^CUDAQ_INSTALL_PATH=.*/ s@@CUDAQ_INSTALL_PATH=${CUDA_QUANTUM_PATH}@" >> ~/.bashrc \n\
    fi \n\
    if [ -f /etc/zprofile ] && [ -w /etc/zprofile ]; then \n\
        echo "Configuring CUDA Quantum environment variables in /etc/zprofile." \n\
        sed "/^\s*\(#\|$\)/d" "${CUDA_QUANTUM_PATH}/set_env.sh" | sed "/^CUDAQ_INSTALL_PATH=.*/ s@@CUDAQ_INSTALL_PATH=${CUDA_QUANTUM_PATH}@" >> /etc/zprofile \n\
    fi \n\n\
    if [ -d "${MPI_PATH}" ] && [ -n "$(ls -A "${MPI_PATH}"/* 2> /dev/null)" ] && [ -x "$(command -v "${CUDA_QUANTUM_PATH}/bin/nvq++")" ]; then \n\
        bash "${CUDA_QUANTUM_PATH}/distributed_interfaces/activate_custom_mpi.sh" \n\
        chmod a+rX "${CUDA_QUANTUM_PATH}/distributed_interfaces/libcudaq_distributed_interface_mpi.so" \n\
    fi \n' >> install.sh

## [Content]
RUN source /cuda-quantum/scripts/configure_build.sh && \
    chmod a+x install.sh "${CUDAQ_INSTALL_PREFIX}/set_env.sh" && \
    mkdir cuda_quantum_assets && mv install.sh cuda_quantum_assets/install.sh && \
    ## [>CUDAQuantumAssets]
    mkdir -p cuda_quantum_assets/llvm/bin && mkdir -p cuda_quantum_assets/llvm/lib && \
    mv "${LLVM_INSTALL_PREFIX}/bin/"clang* cuda_quantum_assets/llvm/bin/ && \
    mv "${LLVM_INSTALL_PREFIX}/lib/"clang* cuda_quantum_assets/llvm/lib/ && \
    mv "${LLVM_INSTALL_PREFIX}/bin/llc" cuda_quantum_assets/llvm/bin/llc && \
    mv "${LLVM_INSTALL_PREFIX}/bin/lld" cuda_quantum_assets/llvm/bin/lld && \
    mv "${LLVM_INSTALL_PREFIX}/bin/ld.lld" cuda_quantum_assets/llvm/bin/ld.lld && \
    mv "${CUTENSOR_INSTALL_PREFIX}" cuda_quantum_assets && \
    mv "${CUQUANTUM_INSTALL_PREFIX}" cuda_quantum_assets && \
    mv "${CUDAQ_INSTALL_PREFIX}/build_config.xml" cuda_quantum_assets/build_config.xml && \
    mv "${CUDAQ_INSTALL_PREFIX}" cuda_quantum_assets
    ## [<CUDAQuantumAssets]

## [Self-extracting Archive]
RUN bash /makeself/makeself.sh --gzip --license /cuda-quantum/LICENSE \
        /cuda_quantum_assets install_cuda_quantum.$(uname -m) \
        "CUDA Quantum toolkit for heterogeneous quantum-classical workflows" \
        bash install.sh -t /opt/nvidia/cudaq

FROM scratch
COPY --from=assets install_cuda_quantum.* . 
