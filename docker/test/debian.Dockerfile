# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

ARG os_version=12
FROM debian:$os_version
SHELL ["/bin/bash", "-c"]

ARG python_version=3.11
ARG pip_install_flags=""
ARG preinstalled_modules="numpy pytest nvidia-cublas-cu11"

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python${python_version} python${python_version}-venv

# We need to make sure the virtual Python environment remains
# activated for all subsequent commands.
ENV VIRTUAL_ENV=/opt/venv
RUN python${python_version} -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN if [ -n "$preinstalled_modules" ]; then \
        echo $preinstalled_modules | xargs python${python_version} -m pip install; \
    fi

ARG optional_dependencies=
ARG cuda_quantum_wheel=cuda_quantum-0.0.0-cp311-cp311-manylinux_2_28_x86_64.whl
COPY $cuda_quantum_wheel /tmp/$cuda_quantum_wheel
COPY docs/sphinx/examples/python /tmp/examples/
COPY python/tests /tmp/tests/

RUN if [ -n "$pip_install_flags" ]; then \
        # We can't install with a --user flag in a virtual environment unless we enable this.
        sed -i 's/include-system-site-packages = false/include-system-site-packages = true/' $VIRTUAL_ENV/pyvenv.cfg; \
    fi
RUN python${python_version} -m pip install ${pip_install_flags} /tmp/$cuda_quantum_wheel
RUN if [ -n "$optional_dependencies" ]; then python${python_version} -m pip install cuda-quantum[$optional_dependencies]; fi
