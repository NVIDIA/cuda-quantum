# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

ARG os_version=leap:15.5
FROM opensuse/$os_version

ARG python_version=3.11
ARG pip_install_flags="--user"
ARG preinstalled_modules="numpy pytest nvidia-cublas-cu11"

ARG DEBIAN_FRONTEND=noninteractive
RUN zypper clean --all && zypper --non-interactive up --no-recommends \
    && zypper --non-interactive in --no-recommends \
        python$(echo ${python_version} | tr -d .) \
    && python${python_version} -m ensurepip --upgrade
RUN if [ -n "$preinstalled_modules" ]; then \
        echo $preinstalled_modules | xargs python${python_version} -m pip install; \
    fi

ARG optional_dependencies=
ARG cuda_quantum_wheel=cuda_quantum-0.0.0-cp39-cp39-manylinux_2_28_x86_64.whl
COPY $cuda_quantum_wheel /tmp/$cuda_quantum_wheel
COPY docs/sphinx/examples/python /tmp/examples/
COPY python/tests /tmp/tests/

RUN python${python_version} -m pip install ${pip_install_flags} /tmp/$cuda_quantum_wheel
RUN if [ -n "$optional_dependencies" ]; then python${python_version} -m pip install cuda-quantum[$optional_dependencies]; fi
