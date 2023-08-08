# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

ARG os_version=12
FROM debian:$os_version

ARG python_version=3.11
ARG pip_install_flags="--user"
ARG preinstalled_modules="numpy pytest"

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python${python_version} python${python_version}-venv
RUN python${python_version} -m venv /usr/local/venvs/cudaq \
    && source /usr/local/venvs/cudaq/bin/activate
RUN if [ -n "$preinstalled_modules" ]; then \
        echo $preinstalled_modules | xargs python${python_version} -m pip install; \
    fi

ARG cuda_quantum_wheel=cuda_quantum-0.0.0-cp310-cp310-manylinux_2_28_x86_64.whl
COPY $cuda_quantum_wheel /tmp/$cuda_quantum_wheel
COPY docs/sphinx/examples/python /tmp/examples/
COPY python/tests /tmp/tests/

RUN cudaq/bin/python${python_version} -m pip install ${pip_install_flags} /tmp/$cuda_quantum_wheel