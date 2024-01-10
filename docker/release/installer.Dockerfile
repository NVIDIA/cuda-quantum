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
ENV CUDAQ_INSTALL_PATH=/opt/nvidia/cudaq
RUN echo 'export CUDA_QUANTUM_PATH="${CUDA_QUANTUM_PATH:-'"${CUDAQ_INSTALL_PATH}"'}"' > set_env.sh && \
    echo 'export PATH="${PATH:+$PATH:}${CUDA_QUANTUM_PATH}/bin"' >> set_env.sh && \
    echo 'export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}${CUDA_QUANTUM_PATH}/lib"' >> set_env.sh && \
    echo 'export CPLUS_INCLUDE_PATH="${CPLUS_INCLUDE_PATH:+$CPLUS_INCLUDE_PATH:}${CUDA_QUANTUM_PATH}/include:${CPLUS_INCLUDE_PATH}"' >> set_env.sh
RUN cp /cuda-quantum/scripts/migrate_assets.sh install.sh && \
    echo -e '\n\n\
    this_file_dir=`dirname "$(readlink -f "${BASH_SOURCE[0]}")"` \n\
    source "$this_file_dir/set_env.sh" \n\
    if [ -f /etc/profile ] && [ -w /etc/profile ]; then \n\
        cat "$this_file_dir/set_env.sh" >> /etc/profile \n\
    fi \n\
    if [ -f /etc/zprofile ] && [ -w /etc/zprofile ]; then \n\
        cat "$this_file_dir/set_env.sh" >> /etc/zprofile \n\
    fi \n\n\
    if [ -d "${MPI_PATH}" ] && [ -n "$(ls -A "${MPI_PATH}"/* 2> /dev/null)" ] && [ -x "$(command -v "${CUDA_QUANTUM_PATH}/bin/nvq++")" ]; then \n\
        bash "${CUDA_QUANTUM_PATH}/distributed_interfaces/activate_custom_mpi.sh" \n\
    fi \n' >> install.sh

## [Content]
RUN source /cuda-quantum/scripts/configure_build.sh && \
    archive=/cuda_quantum && mkdir -p "${archive}" && \
    chmod a+x install.sh && chmod a+x set_env.sh && \
    mv install.sh "${archive}/install.sh" && \
    mv set_env.sh "${archive}/set_env.sh" && \
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

## [Self-extracting Archive]
RUN bash /makeself/makeself.sh --gzip --license /cuda-quantum/LICENSE \
        /cuda_quantum install_cuda_quantum.$(uname -m) \
        "CUDA Quantum toolkit for heterogeneous quantum-classical workflows" \
        bash install.sh -t "${CUDAQ_INSTALL_PATH}"

FROM scratch
COPY --from=assets install_cuda_quantum.* . 
