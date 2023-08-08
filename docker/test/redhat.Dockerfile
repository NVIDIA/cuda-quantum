# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

ARG os_version=ubi9:9.2
ARG python_version=3.11

FROM redhat/$os_version

ARG DEBIAN_FRONTEND=noninteractive
RUN dnf install -y --nobest --setopt=install_weak_deps=False \
        python${python_version}.x86_64
RUN python${python_version} -m ensurepip --upgrade \
    && python${python_version} -m pip install numpy pytest

ARG cuda_quantum_wheel=cuda_quantum-0.0.0-cp310-cp310-manylinux_2_28_x86_64.whl
ARG pip_install_flags="--user"

COPY $cuda_quantum_wheel /tmp/$cuda_quantum_wheel
COPY docs/sphinx/examples/python /tmp/examples/
COPY python/tests /tmp/tests/

RUN python${python_version} -m pip install ${pip_install_flags} /tmp/$cuda_quantum_wheel