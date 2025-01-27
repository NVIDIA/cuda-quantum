# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

ARG base_image=redhat/ubi9:9.2
FROM ${base_image}

ARG python_version=3.11
ARG pip_install_flags="--user"
ARG preinstalled_modules="numpy pytest nvidia-cublas-cu12"

ARG DEBIAN_FRONTEND=noninteractive
RUN dnf install -y --nobest --setopt=install_weak_deps=False \
        python${python_version} \
    && python${python_version} -m ensurepip --upgrade
RUN if [ -n "$preinstalled_modules" ]; then \
        echo $preinstalled_modules | xargs python${python_version} -m pip install; \
    fi

ARG optional_dependencies=
ARG cuda_quantum_wheel=cuda_quantum_cu12-0.0.0-cp311-cp311-manylinux_2_28_x86_64.whl
COPY $cuda_quantum_wheel /tmp/$cuda_quantum_wheel
COPY docs/sphinx/examples/python /tmp/examples/
COPY docs/sphinx/applications/python /tmp/applications/
COPY docs/sphinx/targets/python /tmp/targets/
COPY docs/sphinx/snippets/python /tmp/snippets/
COPY python/tests /tmp/tests/
COPY python/README*.md /tmp/

RUN python${python_version} -m pip install ${pip_install_flags} /tmp/$cuda_quantum_wheel
RUN if [ -n "$optional_dependencies" ]; then \
        cudaq_package=$(echo $cuda_quantum_wheel | cut -d '-' -f1 | tr _ -) && \
        python${python_version} -m pip install ${pip_install_flags} $cudaq_package[$optional_dependencies]; \
    fi
