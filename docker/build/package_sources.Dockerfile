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
# base_image is the base image to use for the build.
#
# Expects in build context:
#   package-source-diff/apt_packages_cudaq.txt   - one apt package name per line (cudaq)
#   package-source-diff/pip_packages_cudaq.txt   - one pip package==version per line (cudaq)
#   package-source-diff/apt_packages_cudaqx.txt - one apt package name per line (cudaqx)
#   package-source-diff/pip_packages_cudaqx.txt - one pip package==version per line (cudaqx)
#   tpls_commits.lock                      - "<commit> <path>" per submodule (same as install_prerequisites.sh -l)
#   .gitmodules                            - submodule paths and URLs
#   scripts/clone_tpls_from_lock.sh        - clone script
#   NOTICE, LICENSE                        - attribution

ARG base_image=ubuntu:24.04
FROM ${base_image}

SHELL ["/bin/bash", "-c"]
ARG DEBIAN_FRONTEND=noninteractive

# Install deps for fetching apt source, pip sdists, and cloning tpls
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    curl \
    dpkg-dev \
    git \
    jq \
    python3 \
    python3-pip \
    unzip \
    && python3 -m pip install --upgrade unearth --break-system-packages \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install necessary repository for librdmac1
RUN apt-get update && apt-get install -y --no-install-recommends gnupg wget \
    && wget -qO - "https://www.mellanox.com/downloads/ofed/RPM-GPG-KEY-Mellanox" | apt-key add - \
    && mkdir -p /etc/apt/sources.list.d && wget -q -nc --no-check-certificate -P /etc/apt/sources.list.d "https://linux.mellanox.com/public/repo/mlnx_ofed/5.3-1.0.0.1/ubuntu20.04/mellanox_mlnx_ofed.list" \
    && echo 'deb-src http://linux.mellanox.com/public/repo/mlnx_ofed/5.3-1.0.0.1/ubuntu20.04/$(ARCH) ./' >> /etc/apt/sources.list.d/mellanox_mlnx_ofed.list \
    && apt-get update -y

# Enable source repositories (Ubuntu 24.04 DEB822 format)
RUN if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then \
      sed -i 's/^Types: deb$/Types: deb deb-src/' /etc/apt/sources.list.d/ubuntu.sources; \
    else \
      sed -i '/^# deb-src/s/^# //' /etc/apt/sources.list 2>/dev/null || true; \
    fi
RUN apt-get update

ENV SOURCES_ROOT=/sources
RUN mkdir -p "${SOURCES_ROOT}/apt" "${SOURCES_ROOT}/pip" "${SOURCES_ROOT}/tpls" "${SOURCES_ROOT}/scripts"

ENV SCRIPTS_DIR=${SOURCES_ROOT}/scripts

# Copy .gitmodules, tpls lock file, clone script, package lists, and pip sdist fetcher
COPY .gitmodules "${SCRIPTS_DIR}"/.gitmodules
COPY tpls_commits.lock "${SCRIPTS_DIR}"/tpls_commits.lock
COPY scripts/clone_tpls_from_lock.sh "${SCRIPTS_DIR}"/clone_tpls_from_lock.sh
COPY package-source-diff/apt_packages_cudaq.txt package-source-diff/apt_packages_cudaqx.txt "${SCRIPTS_DIR}"/
COPY package-source-diff/pip_packages_cudaq.txt package-source-diff/pip_packages_cudaqx.txt "${SCRIPTS_DIR}"/

# Copy attribution
COPY NOTICE LICENSE "${SOURCES_ROOT}/"

# Fetch apt source, pip sdists, and clone tpls in parallel (prefix lines so logs stay readable)
RUN apt-get update && set -o pipefail && \
    ( set -o pipefail; cd "${SOURCES_ROOT}/apt" && \
      chmod 777 . && \
      : > "${SOURCES_ROOT}/apt/apt_omitted_packages.txt" && \
      for list in "${SCRIPTS_DIR}"/apt_packages_cudaq.txt "${SCRIPTS_DIR}"/apt_packages_cudaqx.txt; do \
        [ -f "$list" ] && while IFS= read -r pkg || [ -n "$pkg" ]; do \
          [ -z "$pkg" ] && continue; \
          apt-get source -y "$pkg" || echo "$pkg" >> "${SOURCES_ROOT}/apt/apt_omitted_packages.txt"; \
        done < "$list"; \
      done; \
      ) 2>&1 | sed 's/^/[apt] /' & \
    ( set -o pipefail; : > "${SOURCES_ROOT}/pip/pip_omitted_packages.txt" && \
      cd "${SOURCES_ROOT}/pip" && \
      for list in "${SCRIPTS_DIR}"/pip_packages_cudaq.txt "${SCRIPTS_DIR}"/pip_packages_cudaqx.txt; do \
        [ -f "$list" ] && while IFS= read -r package || [ -n "$package" ]; do \
          [ -z "$package" ] && continue; \
          url=$(unearth --no-binary "$package" 2>/dev/null | jq -r '.link.url'); \
          if [ -n "$url" ] && [ "$url" != "null" ]; then \
            curl -fsSL -O "$url" || echo "$package" >> pip_omitted_packages.txt; \
          else \
            echo "$package" >> pip_omitted_packages.txt; \
          fi; \
        done < "$list"; \
      done; \
      ) 2>&1 | sed 's/^/[pip] /' & \
    ( set -o pipefail; SOURCES_ROOT="${SOURCES_ROOT}" GITMODULES="${SCRIPTS_DIR}"/.gitmodules lock_file="${SCRIPTS_DIR}"/tpls_commits.lock \
      bash "${SCRIPTS_DIR}"/clone_tpls_from_lock.sh ) 2>&1 | sed 's/^/[tpls] /' & \
    wait

RUN echo -e "apt_omitted_packages.txt:\n$(cat ${SOURCES_ROOT}/apt/apt_omitted_packages.txt)"
RUN echo -e "pip_omitted_packages.txt:\n$(cat ${SOURCES_ROOT}/pip/pip_omitted_packages.txt)"

# For omitted apt packages (no source available), extract license/copyright/EULA from the .deb
RUN echo "Retrieving EULA/copyright for omitted apt packages..." && \
    mkdir -p "${SOURCES_ROOT}/apt/licenses" /tmp/deb_extract && \
    while IFS= read -r pkg || [ -n "$pkg" ]; do \
      [ -z "$pkg" ] && continue; \
      ( cd /tmp/deb_extract && apt-get download "$pkg" 2>/dev/null ) || true; \
      deb=$(ls /tmp/deb_extract/*.deb 2>/dev/null | head -1); \
      if [ -n "$deb" ]; then \
        dpkg-deb -R "$deb" "/tmp/deb_extract/${pkg}_pkg" 2>/dev/null || true; \
        dest="${SOURCES_ROOT}/apt/licenses/${pkg}"; \
        mkdir -p "$dest"; \
        find "/tmp/deb_extract/${pkg}_pkg" \( -iname "*license*" -o -iname "*eula*" -o -iname "*copyright*" \) -exec cp -a {} "$dest/" \; 2>/dev/null || true; \
        rm -rf "/tmp/deb_extract/${pkg}_pkg" /tmp/deb_extract/*.deb; \
      fi; \
    done < "${SOURCES_ROOT}/apt/apt_omitted_packages.txt"; \
    rm -rf /tmp/deb_extract

# For omitted pip packages (no sdist), get EULA/license from the wheel: fetch wheel from PyPI, extract, copy license/EULA/copyright files
RUN echo "Retrieving EULA/license for omitted pip packages..." && \
    mkdir -p "${SOURCES_ROOT}/pip/licenses" /tmp/wheel_extract && \
    while IFS= read -r package || [ -n "$package" ]; do \
      [ -z "$package" ] && continue; \
      name="${package%%==*}"; \
      version="${package#*==}"; \
      [ -z "$name" ] || [ -z "$version" ] || [ "$version" = "$package" ] && continue; \
      url=$(curl -sS "https://pypi.org/pypi/${name}/${version}/json" 2>/dev/null | jq -r '.urls[] | select(.packagetype=="bdist_wheel") | select(.filename | test("manylinux.*x86_64|manylinux_2.*x86_64")) | .url' 2>/dev/null | head -1); \
      if [ -n "$url" ] && [ "$url" != "null" ]; then \
        if curl -fsSL -o /tmp/pip_wheel.whl "$url" 2>/dev/null; then \
          (cd /tmp/wheel_extract && unzip -o -q /tmp/pip_wheel.whl 2>/dev/null) || true; \
          dest="${SOURCES_ROOT}/pip/licenses/${name}"; \
          mkdir -p "$dest"; \
          find /tmp/wheel_extract -type f \( -iname "*license*" -o -iname "*eula*" -o -iname "*copyright*" \) -exec cp -an {} "$dest/" \; 2>/dev/null || true; \
          if [ -z "$(ls -A "$dest" 2>/dev/null)" ]; then \
            license_text=$(curl -sS "https://pypi.org/pypi/${name}/${version}/json" 2>/dev/null | jq -r '.info.license // .info.license_expression // empty'); \
            [ -n "$license_text" ] && [ "$license_text" != "null" ] && echo "$license_text" > "$dest/LICENSE_from_PyPI.txt"; \
          fi; \
          find /tmp/wheel_extract -mindepth 1 -delete 2>/dev/null || rm -rf /tmp/wheel_extract/*; \
        fi; \
        rm -f /tmp/pip_wheel.whl; \
      fi; \
    done < "${SOURCES_ROOT}/pip/pip_omitted_packages.txt"; \
    rm -rf /tmp/wheel_extract

# Summary
RUN echo "apt: $(find ${SOURCES_ROOT}/apt -maxdepth 1 -type d 2>/dev/null | wc -l) dirs" && \
    echo "pip: $(find ${SOURCES_ROOT}/pip -maxdepth 1 -type f \( -name '*.tar.gz' -o -name '*.zip' \) 2>/dev/null | wc -l) sdists" && \
    echo "tpls: $(find ${SOURCES_ROOT}/tpls -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l) libraries"

WORKDIR ${SOURCES_ROOT}
