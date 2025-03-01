# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

ARG base_image=ubuntu:22.04
FROM ${base_image}

ARG python_version=3.10
ARG pip_install_flags="--user"
ARG preinstalled_modules="numpy pytest nvidia-cublas-cu12"

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
ARG cuda_quantum_wheel=cuda_quantum_cu12-0.0.0-cp310-cp310-manylinux_2_28_x86_64.whl
COPY $cuda_quantum_wheel /tmp/$cuda_quantum_wheel
COPY docs/sphinx/examples/python /tmp/examples/
COPY docs/sphinx/applications/python /tmp/applications/
COPY docs/sphinx/targets/python /tmp/targets/
COPY docs/sphinx/snippets/python /tmp/snippets/
COPY python/tests /tmp/tests/
COPY python/README*.md /tmp/

RUN sed -ie 's/include-system-site-packages\s*=\s*false/include-system-site-packages = true/g' "$VIRTUAL_ENV/pyvenv.cfg"

RUN python${python_version} -m pip install ${pip_install_flags} /tmp/$cuda_quantum_wheel
RUN if [ -n "$optional_dependencies" ]; then \
        cudaq_package=$(echo $cuda_quantum_wheel | cut -d '-' -f1 | tr _ -) && \
        python${python_version} -m pip install ${pip_install_flags} $cudaq_package[$optional_dependencies]; \
    fi
