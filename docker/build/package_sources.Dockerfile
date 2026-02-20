# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
#
# Builds an Ubuntu 24.04 image containing source code for a set of apt and pip
# packages plus the repo's tpls/ (third-party library) source. Tpls are cloned
# at build time using .gitmodules and a lock file (commit + path per line) via
# git clone --no-checkout --filter=tree:0 + fetch + checkout.
#
# Build from repo root with package-source-diff/ and tpls_commits.lock (or generate with scripts/generate_tpls_lock.sh):
#   docker build -t package-sources:latest -f docker/build/package_sources.Dockerfile .
#
# Expects in build context:
#   package-source-diff/apt_packages.txt   - one apt package name per line
#   package-source-diff/pip_packages.txt   - one pip package==version per line
#   tpls_commits.lock                      - "<commit> <path>" per submodule (same as install_prerequisites.sh -l)
#   .gitmodules                            - submodule paths and URLs
#   scripts/clone_tpls_from_lock.sh        - clone script
#   NOTICE, LICENSE                        - attribution

FROM ubuntu:24.04

SHELL ["/bin/bash", "-c"]
ARG DEBIAN_FRONTEND=noninteractive

# Install deps for fetching apt source, pip sdists, and cloning tpls
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    dpkg-dev \
    git \
    python3 \
    python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Enable source repositories (Ubuntu 24.04 DEB822 format)
RUN if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then \
      sed -i 's/^Types: deb$/Types: deb deb-src/' /etc/apt/sources.list.d/ubuntu.sources; \
    else \
      sed -i '/^# deb-src/s/^# //' /etc/apt/sources.list 2>/dev/null || true; \
    fi
RUN apt-get update

ENV SOURCES_ROOT=/sources
RUN mkdir -p "${SOURCES_ROOT}/apt" "${SOURCES_ROOT}/pip" "${SOURCES_ROOT}/tpls"

# Copy .gitmodules, tpls lock file, and clone script; then clone each tpl at pinned commit
COPY .gitmodules /tmp/.gitmodules
COPY tpls_commits.lock /tmp/tpls_commits.lock
COPY scripts/clone_tpls_from_lock.sh /tmp/clone_tpls_from_lock.sh
RUN SOURCES_ROOT="${SOURCES_ROOT}" GITMODULES=/tmp/.gitmodules lock_file=/tmp/tpls_commits.lock \
    bash /tmp/clone_tpls_from_lock.sh

# Copy attribution
COPY NOTICE LICENSE "${SOURCES_ROOT}/"

# Copy package lists (workflow writes these into package-source-diff/)
COPY package-source-diff/apt_packages.txt /tmp/apt_packages.txt
COPY package-source-diff/pip_packages.txt /tmp/pip_packages.txt

# Fetch apt source for each package (failures expected: not all packages have source in this image's repos)
RUN apt-get update && \
    cd "${SOURCES_ROOT}/apt" && \
    : > "${SOURCES_ROOT}/apt_failed_packages.txt" && \
    while IFS= read -r pkg || [ -n "$pkg" ]; do \
      [ -z "$pkg" ] && continue; \
      apt-get source -y "$pkg" || echo "$pkg" >> "${SOURCES_ROOT}/apt_failed_packages.txt"; \
    done < /tmp/apt_packages.txt; \
    rm -f /tmp/apt_packages.txt

# Fetch pip sdists (allow failures for binary-only packages; use system pip, do not upgrade)
RUN : > "${SOURCES_ROOT}/pip_failed_packages.txt" && \
    while IFS= read -r spec || [ -n "$spec" ]; do \
      [ -z "$spec" ] && continue; \
      python3 -m pip download --no-binary :all: -d "${SOURCES_ROOT}/pip" "$spec" 2>/dev/null || echo "$spec" >> "${SOURCES_ROOT}/pip_failed_packages.txt"; \
    done < /tmp/pip_packages.txt; \
    rm -f /tmp/pip_packages.txt

RUN echo "apt_failed_packages.txt: $(cat ${SOURCES_ROOT}/apt_failed_packages.txt)"
    rm -f ${SOURCES_ROOT}/apt_failed_packages.txt
RUN echo "pip_failed_packages.txt: $(cat ${SOURCES_ROOT}/pip_failed_packages.txt)"
    rm -f ${SOURCES_ROOT}/pip_failed_packages.txt

# Summary
RUN echo "apt: $(find ${SOURCES_ROOT}/apt -maxdepth 1 -type d 2>/dev/null | wc -l) dirs" && \
    echo "pip: $(find ${SOURCES_ROOT}/pip -maxdepth 1 -type f \( -name '*.tar.gz' -o -name '*.zip' \) 2>/dev/null | wc -l) sdists" && \
    echo "tpls: $(find ${SOURCES_ROOT}/tpls -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l) libraries"

WORKDIR ${SOURCES_ROOT}
