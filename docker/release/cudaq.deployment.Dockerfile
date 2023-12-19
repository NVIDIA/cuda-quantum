# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

FROM amd64/almalinux:8 as mpibuild
SHELL ["/bin/bash", "-c"]
ARG DEBIAN_FRONTEND=
RUN dnf install -y --nobest --setopt=install_weak_deps=False \
        'dnf-command(config-manager)'

# [OpenMPI Prerequisites]
ADD scripts/configure_build.sh /cuda-quantum/scripts/configure_build.sh
RUN source /cuda-quantum/scripts/configure_build.sh install-cuda
RUN source /cuda-quantum/scripts/configure_build.sh install-gcc
RUN dnf install -y --nobest --setopt=install_weak_deps=False \
        autoconf libtool flex make wget

# [OpenMPI Build]
RUN source /cuda-quantum/scripts/configure_build.sh build-openmpi

FROM ubuntu:22.04 as ubuntu
SHELL ["/bin/bash", "-c"]
ARG DEBIAN_FRONTEND=noninteractive

# [Prerequisites]
ARG PYTHON=python3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
        ${PYTHON} lib${PYTHON} $(echo ${PYTHON} | cut -d . -f 1)-pip  wget \
    && ${PYTHON} -m pip install --no-cache-dir numpy

# [Runtime Dependencies]
RUN apt-get install -y --no-install-recommends libstdc++-11-dev
RUN cuda_packages="cuda-cudart-11-8 cuda-nvtx-11-8 libcusolver-11-8 libcublas-11-8" && \
    if [ -n "$cuda_packages" ]; then \
        arch_folder=$([ "$(uname -m)" == "aarch64" ] && echo sbsa || echo x86_64) \
        && wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/$arch_folder/cuda-keyring_1.0-1_all.deb" \
        && dpkg -i cuda-keyring_1.0-1_all.deb \
        && apt-get update && apt-get install -y --no-install-recommends $cuda_packages \
        && rm cuda-keyring_1.0-1_all.deb \
        && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/*; \
    fi

# [CUDA Quantum]
ADD out/cuda_quantum.* .
RUN ./cuda_quantum.$(uname -m) --accept

ENV CUDA_QUANTUM_PATH=/opt/nvidia/cudaq
ENV PATH="${CUDA_QUANTUM_PATH}/bin:${PATH}"
ENV PYTHONPATH="${CUDA_QUANTUM_PATH}:${PYTHONPATH}"
#ENV LD_LIBRARY_PATH="${CUQUANTUM_PATH}/lib:$CUTENSOR_PATH/lib:$LD_LIBRARY_PATH"
ENV CPLUS_INCLUDE_PATH="${CUDA_QUANTUM_PATH}/include:${CPLUS_INCLUDE_PATH}"

# FIXME: REMOVE
ENV NVQPP_LD_PATH=ld
RUN apt-get update && apt-get install -y --no-install-recommends gcc-11
ADD runtime/cudaq/distributed/builtin/activate_custom_mpi.sh ${CUDA_QUANTUM_PATH}/distributed_interfaces/activate_custom_mpi.sh

# [Enable MPI]
COPY --from=mpibuild /usr/local/openmpi/ /usr/local/openmpi
RUN MPI_PATH=/usr/local/openmpi \
    bash "${CUDA_QUANTUM_PATH}/distributed_interfaces/activate_custom_mpi.sh"

# [Validate]
ADD scripts/validate_container.sh validate.sh
ADD docs/sphinx/examples examples

FROM amd64/almalinux:8 as almalinux
SHELL ["/bin/bash", "-c"]
ARG DEBIAN_FRONTEND=noninteractive
RUN dnf install -y --nobest --setopt=install_weak_deps=False \
        'dnf-command(config-manager)'

# [Prerequisites]
ARG PYTHON=python3.11
RUN dnf install -y --nobest --setopt=install_weak_deps=False ${PYTHON}
RUN ${PYTHON} -m ensurepip && ${PYTHON} -m pip install numpy

# [Runtime Dependencies]
ADD scripts/configure_build.sh /cuda-quantum/scripts/configure_build.sh
RUN source /cuda-quantum/scripts/configure_build.sh install-gcc
RUN source /cuda-quantum/scripts/configure_build.sh install-cudart
#RUN dnf install -y --nobest --setopt=install_weak_deps=False openssl
        # libpython3-dev libcurl4-openssl-dev 

# [CUDA Quantum]
ADD out/cuda_quantum.* .
RUN ./cuda_quantum.$(uname -m) --accept

ENV CUDA_QUANTUM_PATH=/opt/nvidia/cudaq
ENV PATH="${CUDA_QUANTUM_PATH}/bin:${PATH}"
ENV PYTHONPATH="${CUDA_QUANTUM_PATH}:${PYTHONPATH}"
#ENV LD_LIBRARY_PATH="${CUQUANTUM_PATH}/lib:$CUTENSOR_PATH/lib:$LD_LIBRARY_PATH"
ENV CPLUS_INCLUDE_PATH="${CUDA_QUANTUM_PATH}/include:${CPLUS_INCLUDE_PATH}"

# FIXME: REMOVE
ENV NVQPP_LD_PATH=ld
ADD runtime/cudaq/distributed/builtin/activate_custom_mpi.sh ${CUDA_QUANTUM_PATH}/distributed_interfaces/activate_custom_mpi.sh

# [Enable MPI]
COPY --from=mpibuild /usr/local/openmpi/ /usr/local/openmpi
RUN MPI_PATH=/usr/local/openmpi \
    bash "${CUDA_QUANTUM_PATH}/distributed_interfaces/activate_custom_mpi.sh"

# [Validate]
ADD scripts/validate_container.sh validate.sh
ADD docs/sphinx/examples examples
#RUN bash validate.sh
