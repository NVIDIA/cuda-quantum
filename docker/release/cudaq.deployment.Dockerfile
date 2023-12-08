# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

FROM amd64/almalinux:8 as mpibuild
ADD scripts/configure_build.sh /cuda-quantum/scripts/configure_build.sh

RUN source /cuda-quantum/scripts/configure_build.sh install-cuda
RUN dnf install -y --nobest --setopt=install_weak_deps=False \
        autoconf libtool flex make

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

FROM cudaq-build as cudaqbuild

FROM amd64/almalinux:8
ARG DEBIAN_FRONTEND=noninteractive

# [Prerequisites]
ARG PYTHON=python3.11
RUN dnf install -y --nobest --setopt=install_weak_deps=False ${PYTHON}
RUN ${PYTHON} -m ensurepip && ${PYTHON} -m pip install numpy

# [Runtime Dependencies]
RUN source /cuda-quantum/scripts/configure_build.sh install-gcc
RUN source /cuda-quantum/scripts/configure_build.sh install-cudart
#RUN dnf install -y --nobest --setopt=install_weak_deps=False openssl
        # libpython3-dev libcurl4-openssl-dev 

# [CUDA Quantum]
ENV CUDA_QUANTUM_PATH=/opt/nvidia/cudaq
ENV CUQUANTUM_PATH=/opt/nvidia/cuquantum
ENV CUTENSOR_PATH=/opt/nvidia/cutensor
#ENV OPENSSL_PATH=/usr/local/openssl
COPY --from=cudaqbuild "/usr/local/cudaq/" "${CUDA_QUANTUM_PATH}"
COPY --from=cudaqbuild "/usr/local/cuquantum/" "${CUQUANTUM_PATH}"
COPY --from=cudaqbuild "/usr/local/cutensor/" "${CUTENSOR_PATH}"
#COPY --from=cudaqbuild "/usr/local/openssl/" "${OPENSSL_PATH}"

ENV PATH="${CUDA_QUANTUM_PATH}/bin:${PATH}"
ENV PYTHONPATH="${CUDA_QUANTUM_PATH}:${PYTHONPATH}"
ENV LD_LIBRARY_PATH="${CUQUANTUM_PATH}/lib:$CUTENSOR_PATH/lib:$LD_LIBRARY_PATH"

# [Enable MPI]
ENV MPI_PATH=/usr/local/openmpi
COPY --from=mpibuild "/usr/local/openmpi/" "${MPI_PATH}"
RUN "${CUDA_QUANTUM_PATH}/activate_mpi.sh"

