# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

ARG base_image=ubuntu:24.04
FROM ${base_image}

ARG python_version=3.10
ARG pip_install_flags="--user"
ARG preinstalled_modules="numpy pytest nvidia-cublas-cu12"

ARG DEBIAN_FRONTEND=noninteractive
# Tolerate transient apt mirror failures.
RUN echo 'Acquire::Retries "5";' > /etc/apt/apt.conf.d/80-retries \
    && echo 'Acquire::Retries::Delay::Maximum "30";' >> /etc/apt/apt.conf.d/80-retries
RUN apt-get update && apt-get install -y --no-install-recommends wget \
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
