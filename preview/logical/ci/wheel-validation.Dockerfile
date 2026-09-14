# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Multi-distro validation image for the cudaq-logical wheel.
#
# One recipe covers every distro in the python matrix of
# .github/workflows/config/validation_config.json (ubuntu, debian, fedora,
# redhat/ubi -- opensuse only appears in the cpp matrix, so there is no zypper
# branch). The distro-specific setup below follows
# docker/test/wheels/{ubuntu,fedora,redhat}.Dockerfile; the install under test
# is the cudaq-logical wheel, resolved together with its CUDA-Q runtime
# dependency from the wheels copied into /tmp/dist.

ARG base_image=ubuntu:22.04
FROM ${base_image}

ARG python_version=3.11
ARG cudaq_version
ARG cudaq_logical_version
ARG pip_install_flags=""
ARG preinstalled_modules="pytest"

ARG DEBIAN_FRONTEND=noninteractive

# On apt distros the wheel is tested inside a virtual Python environment that
# must remain activated for all subsequent commands (mirroring
# docker/test/wheels/ubuntu.Dockerfile). On dnf distros no venv is created
# and these variables harmlessly point at a non-existent directory.
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Distro-specific Python setup. The apt branch follows
# docker/test/wheels/ubuntu.Dockerfile, the dnf branch
# docker/test/wheels/{fedora,redhat}.Dockerfile -- package names differ
# between fedora (python311) and redhat/ubi (python3.11).
RUN if command -v apt-get >/dev/null 2>&1; then \
        # Tolerate transient apt mirror failures.
        echo 'Acquire::Retries "5";' > /etc/apt/apt.conf.d/80-retries \
        && echo 'Acquire::Retries::Delay::Maximum "30";' >> /etc/apt/apt.conf.d/80-retries \
        && apt-get update && apt-get install -y --no-install-recommends wget \
            python${python_version} python${python_version}-venv \
        && python${python_version} -m venv "$VIRTUAL_ENV" \
        # A --user pip install inside a virtual environment only works with
        # system site packages enabled (ENABLE_USER_SITE: True).
        && sed -ie 's/include-system-site-packages\s*=\s*false/include-system-site-packages = true/g' "$VIRTUAL_ENV/pyvenv.cfg"; \
    elif command -v dnf >/dev/null 2>&1; then \
        if grep -q fedora /etc/os-release; then \
            dnf install -y --refresh --setopt=install_weak_deps=False expat \
            && dnf install -y --nobest --setopt=install_weak_deps=False wget \
                python$(echo $python_version | tr -d .); \
        else \
            dnf install -y --nobest --setopt=install_weak_deps=False wget \
                python${python_version}; \
        fi \
        && python${python_version} -m ensurepip --upgrade; \
    else \
        echo "Unsupported base image: needs apt-get or dnf." >&2; \
        exit 1; \
    fi

RUN if [ -n "$preinstalled_modules" ]; then \
        echo $preinstalled_modules | xargs python${python_version} -m pip install; \
    fi

# wheel-validation/ holds the cudaq-logical wheel under test plus the
# matching CUDA-Q runtime wheel and cudaq metapackage it depends on.
COPY wheel-validation/ /tmp/dist/
# The test suite reads files across the cudaq-logical source tree
# (python/tests/cudaq/logical/test_*.py resolve the tree root via parents[4]),
# so the full preview/logical tree must be present, not just the tests.
COPY preview/logical/ /tmp/cudaq-logical/

# Working around issue https://github.com/pypa/pip/issues/11153.
# Retry download to a file (not a pipe) to survive transient/truncated fetches.
RUN for i in 1 2 3; do \
        wget --tries=3 --retry-connrefused --waitretry=5 --timeout=30 \
            https://github.com/rapidsai/gha-tools/releases/latest/download/tools.tar.gz -O /tmp/tools.tar.gz \
        && gzip -t /tmp/tools.tar.gz && tar -xzf /tmp/tools.tar.gz -C /usr/local/bin && break \
        || { echo "gha-tools download attempt $i failed; retrying..."; sleep 5; }; \
    done && rm -f /tmp/tools.tar.gz && \
    RAPIDS_PIP_EXE="python${python_version} -m pip" \
    /usr/local/bin/rapids-pip-retry install ${pip_install_flags} --find-links /tmp/dist \
        "cudaq==$cudaq_version" \
        "cudaq-logical==$cudaq_logical_version"
