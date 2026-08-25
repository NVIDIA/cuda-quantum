# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

ARG base_image=fedora:41
FROM ${base_image}

ARG python_version=3.10
ARG pip_install_flags="--user"
ARG preinstalled_modules="numpy pytest nvidia-cublas-cu12"

ARG DEBIAN_FRONTEND=noninteractive

RUN dnf install -y --refresh --setopt=install_weak_deps=False expat \
    && dnf install -y --nobest --setopt=install_weak_deps=False wget \
        python$(echo $python_version | tr -d .) \
    && python${python_version} -m ensurepip --upgrade
RUN if [ -n "$preinstalled_modules" ]; then \
        echo $preinstalled_modules | xargs python${python_version} -m pip install; \
    fi

ARG optional_dependencies=
ARG cuda_quantum_wheel=cuda_quantum_cu12-0.0.0-cp310-cp310-manylinux_2_28_x86_64.whl
COPY $cuda_quantum_wheel /tmp/$cuda_quantum_wheel
COPY docs/sphinx/examples/python /tmp/examples/
COPY docs/sphinx/applications/python /tmp/applications/
COPY docs/sphinx/targets/python /tmp/targets/
COPY docs/sphinx/snippets/python /tmp/snippets/
COPY python/tests /tmp/tests/
COPY python/README*.md /tmp/

# Working around issue https://github.com/pypa/pip/issues/11153.
# Retry download to a file (not a pipe) to survive transient/truncated fetches.
RUN for i in 1 2 3; do \
        wget --tries=3 --retry-connrefused --waitretry=5 --timeout=30 \
            https://github.com/rapidsai/gha-tools/releases/latest/download/tools.tar.gz -O /tmp/tools.tar.gz \
        && gzip -t /tmp/tools.tar.gz && tar -xzf /tmp/tools.tar.gz -C /usr/local/bin && break \
        || { echo "gha-tools download attempt $i failed; retrying..."; sleep 5; }; \
    done && rm -f /tmp/tools.tar.gz && \
    RAPIDS_PIP_EXE="python${python_version} -m pip" \
    /usr/local/bin/rapids-pip-retry install ${pip_install_flags} /tmp/$cuda_quantum_wheel
RUN if [ -n "$optional_dependencies" ]; then \
        cudaq_package=$(echo $cuda_quantum_wheel | cut -d '-' -f1 | tr _ -) && \
        RAPIDS_PIP_EXE="python${python_version} -m pip" \
        /usr/local/bin/rapids-pip-retry install ${pip_install_flags} $cudaq_package[$optional_dependencies]; \
    fi
