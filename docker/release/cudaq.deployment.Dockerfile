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
        'dnf-command(config-manager)' && \
    dnf config-manager --enable powertools

ADD scripts/configure_build.sh /cuda-quantum/scripts/configure_build.sh
RUN source /cuda-quantum/scripts/configure_build.sh install-cuda
RUN dnf install -y --nobest --setopt=install_weak_deps=False \
        autoconf libtool flex make git

ENV OPENMPI_VERSION=4.1.4
RUN mkdir ~/.openmpi-project && cd ~/.openmpi-project && \
    git init && git remote add origin https://github.com/open-mpi/ompi && \
    git fetch origin --depth=1 v${OPENMPI_VERSION} && git reset --hard FETCH_HEAD

RUN source /cuda-quantum/scripts/configure_build.sh install-gcc && \
    cd ~/.openmpi-project && ./autogen.pl && \
    PATH="$(dirname $CC):$PATH" LDFLAGS=-Wl,--as-needed \
    ./configure \
        --prefix="/usr/local/openmpi" \
        --disable-getpwuid --disable-static \
        --disable-debug --disable-mem-debug --disable-event-debug \
        --disable-mem-profile --disable-memchecker \
        --without-verbs \
        --with-cuda=/usr/local/cuda && \
    make -j$(nproc) && make -j$(nproc) install && \
    cd - && rm -rf ~/.openmpi-project

FROM ubuntu:22.04 as ubuntu
SHELL ["/bin/bash", "-c"]
ARG DEBIAN_FRONTEND=noninteractive

# [Prerequisites]
ARG PYTHON=python3
RUN apt-get update && apt-get install -y --no-install-recommends \
        ${PYTHON} ${PYTHON}-pip \
    && python3 -m pip install --no-cache-dir numpy

# [Runtime Dependencies]
RUN apt-get install -y --no-install-recommends libstdc++-12-dev
RUN cuda_packages="cuda-cudart-11-8 cuda-nvtx-11-8 libcusolver-11-8" && \
    if [ -n "$cuda_packages" ]; then \
        arch_folder=$([ "$(uname -m)" == "aarch64" ] && echo sbsa || echo x86_64) \
        && wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/$arch_folder/cuda-keyring_1.0-1_all.deb" \
        && dpkg -i cuda-keyring_1.0-1_all.deb \
        && apt-get update && apt-get install -y --no-install-recommends $cuda_packages \
        && rm cuda-keyring_1.0-1_all.deb \
        && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/*; \
    fi

# [CUDA Quantum]
#ENV CUDA_QUANTUM_PATH=/opt/nvidia/cudaq
#ENV CUQUANTUM_PATH=/opt/nvidia/cuquantum
#ENV CUTENSOR_PATH=/opt/nvidia/cutensor
#ENV OPENSSL_PATH=/usr/local/openssl
#COPY --from=cudaqbuild "/usr/local/cudaq/" "${CUDA_QUANTUM_PATH}"
#COPY --from=cudaqbuild "/usr/local/cuquantum/" "${CUQUANTUM_PATH}"
#COPY --from=cudaqbuild "/usr/local/cutensor/" "${CUTENSOR_PATH}"
#COPY --from=cudaqbuild "/usr/local/openssl/" "${OPENSSL_PATH}"
ADD out/cuda_quantum.* .
RUN ./cuda_quantum.$(uname -m) --accept

ENV PATH="${CUDA_QUANTUM_PATH}/bin:${PATH}"
ENV PYTHONPATH="${CUDA_QUANTUM_PATH}:${PYTHONPATH}"
#ENV LD_LIBRARY_PATH="${CUQUANTUM_PATH}/lib:$CUTENSOR_PATH/lib:$LD_LIBRARY_PATH"
ENV CPLUS_INCLUDE_PATH="${CUDA_QUANTUM_PATH}/include:${CPLUS_INCLUDE_PATH}"

# [Enable MPI]
ENV MPI_PATH=/usr/local/openmpi
COPY --from=mpibuild "/usr/local/openmpi/" "${MPI_PATH}"
RUN "${CUDA_QUANTUM_PATH}/bin/activate_mpi.sh"

FROM amd64/almalinux:8 as almalinux
SHELL ["/bin/bash", "-c"]
ARG DEBIAN_FRONTEND=noninteractive

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
#ENV CUDA_QUANTUM_PATH=/opt/nvidia/cudaq
#ENV CUQUANTUM_PATH=/opt/nvidia/cuquantum
#ENV CUTENSOR_PATH=/opt/nvidia/cutensor
#ENV OPENSSL_PATH=/usr/local/openssl
#COPY --from=cudaqbuild "/usr/local/cudaq/" "${CUDA_QUANTUM_PATH}"
#COPY --from=cudaqbuild "/usr/local/cuquantum/" "${CUQUANTUM_PATH}"
#COPY --from=cudaqbuild "/usr/local/cutensor/" "${CUTENSOR_PATH}"
#COPY --from=cudaqbuild "/usr/local/openssl/" "${OPENSSL_PATH}"
ADD out/cuda_quantum.* .
RUN ./cuda_quantum.$(uname -m) --accept

ENV PATH="${CUDA_QUANTUM_PATH}/bin:${PATH}"
ENV PYTHONPATH="${CUDA_QUANTUM_PATH}:${PYTHONPATH}"
#ENV LD_LIBRARY_PATH="${CUQUANTUM_PATH}/lib:$CUTENSOR_PATH/lib:$LD_LIBRARY_PATH"
ENV CPLUS_INCLUDE_PATH="${CUDA_QUANTUM_PATH}/include:${CPLUS_INCLUDE_PATH}"

# [Enable MPI]
ENV MPI_PATH=/usr/local/openmpi
COPY --from=mpibuild "/usr/local/openmpi/" "${MPI_PATH}"
RUN "${CUDA_QUANTUM_PATH}/bin/activate_mpi.sh"

