# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
#
# Builds an Ubuntu 24.04 image containing the source code of everything the
# CUDA-Q images ship on top of their base images: apt source packages at the
# exact installed versions, pip sdists, the repo's tpls/ (third-party library)
# sources, the sources of the software built from source in the images (the
# MPI stack, uv, ...) and the DOCA/OFED source bundle, together with the
# license, copyright and EULA files of all of it. The build fails if a package
# has neither source nor a reason to have none (see check_package_sources.sh).
#
# Build from repo root with package-source-diff/ and tpls_commits.lock (or generate with scripts/generate_tpls_lock.sh).
# Trimmed lists are produced inline in the build_package_sources workflow.
#   docker build -t package-sources:latest -f docker/build/package_sources.Dockerfile .
#
# base_image is the base image to use for the build.
# cuda_major is the CUDA major version, used in the names of the notice files.
#
# Expects in build context:
#   package-source-diff/ with, for each variant in cudaq, cudaqx, cudaq-decoding (and macos, pip only):
#     apt_sources_<variant>_trimmed.txt         - source=source_version per line
#     apt_package_details_<variant>_trimmed.tsv - binary, version, source, source version
#     apt_licenses_<variant>/<binary>/          - license/copyright/EULA files from the image
#     pip_packages_<variant>_trimmed.txt        - package==version per line
#     pip_licenses_<variant>/<name>/            - license files and METADATA from the installed distribution
#     images_<variant>.txt                      - the base and target image digests the lists are from
#   prereqs/                               - tarballs and clones from install_prerequisites.sh -l and generate_image_sources_lock.py
#   tpls_commits.lock, .gitmodules         - tpls submodule commits and URLs
#   NOTICE, LICENSE, LICENSES/             - attribution and third-party license texts
#   tpls/customizations, tpls/json, runtime/cudaq/algorithms/optimizers/nlopt - third-party code kept in this repo
# and the scripts and lists copied below.

ARG base_image=ubuntu:24.04
FROM ${base_image}

SHELL ["/bin/bash", "-c"]
ARG DEBIAN_FRONTEND=noninteractive
ARG cuda_major=

# Tolerate transient apt mirror failures.
RUN echo 'Acquire::Retries "5";' > /etc/apt/apt.conf.d/80-retries \
    && echo 'Acquire::Retries::Delay::Maximum "30";' >> /etc/apt/apt.conf.d/80-retries

# Install deps for fetching apt source (ubuntu-dev-tools fetches versions that
# the archive no longer carries from Launchpad), pip sdists, and cloning tpls
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    curl \
    dpkg-dev \
    git \
    jq \
    python3 \
    python3-venv \
    ubuntu-dev-tools \
    unzip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
# In its own environment: ubuntu-dev-tools brings Debian-managed Python packages that pip cannot upgrade.
RUN python3 -m venv /opt/unearth && /opt/unearth/bin/pip install --no-cache-dir unearth \
    && ln -s /opt/unearth/bin/unearth /usr/local/bin/unearth

# Enable source repositories (Ubuntu 24.04 DEB822 format)
RUN if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then \
      sed -i 's/^Types: deb$/Types: deb deb-src/' /etc/apt/sources.list.d/ubuntu.sources; \
    else \
      sed -i '/^# deb-src/s/^# //' /etc/apt/sources.list 2>/dev/null || true; \
    fi
RUN apt-get update

ENV SOURCES_ROOT=/sources
RUN mkdir -p "${SOURCES_ROOT}/NOTICES" \
              "${SOURCES_ROOT}/cudaq/apt" \
              "${SOURCES_ROOT}/cudaq/pip" \
              "${SOURCES_ROOT}/cudaqx/apt" \
              "${SOURCES_ROOT}/cudaqx/pip" \
              "${SOURCES_ROOT}/cudaq-decoding/apt" \
              "${SOURCES_ROOT}/cudaq-decoding/pip" \
              "${SOURCES_ROOT}/macos/pip" \
              "${SOURCES_ROOT}/licenses" \
              "${SOURCES_ROOT}/tpls" \
              "${SOURCES_ROOT}/.scripts"

ENV SCRIPTS_DIR=${SOURCES_ROOT}/.scripts

# Scripts, lists, and the package lists of each variant
COPY .gitmodules "${SCRIPTS_DIR}"/.gitmodules
COPY tpls_commits.lock "${SCRIPTS_DIR}"/tpls_commits.lock
COPY scripts/clone_tpls_from_lock.sh scripts/fetch_package_sources.sh scripts/check_package_sources.sh \
     scripts/generate_apt_attribution.py scripts/generate_pip_attribution.py "${SCRIPTS_DIR}"/
COPY docker/build/package_sources_proprietary.txt docker/build/pip_source_aliases.txt "${SCRIPTS_DIR}"/
COPY package-source-diff/ "${SCRIPTS_DIR}"/
COPY prereqs/ "${SOURCES_ROOT}/prereqs/"

# Attribution, and the third-party code that lives in this repo rather than in a submodule
COPY NOTICE LICENSE "${SOURCES_ROOT}/NOTICES/"
COPY LICENSES "${SOURCES_ROOT}/NOTICES/LICENSES/"
COPY tpls/customizations "${SOURCES_ROOT}/tpls-customizations/"
COPY tpls/json "${SOURCES_ROOT}/repo-third-party/json/"
COPY runtime/cudaq/algorithms/optimizers/nlopt "${SOURCES_ROOT}/repo-third-party/nlopt/"

# Fetch apt source, pip sdists, and clone tpls in parallel; fails if tpls cannot be cloned
RUN apt-get update && bash "${SCRIPTS_DIR}"/fetch_package_sources.sh

# Extract pip sdists and remove tarballs
RUN for dir in "${SOURCES_ROOT}/cudaq/pip" "${SOURCES_ROOT}/cudaqx/pip" "${SOURCES_ROOT}/cudaq-decoding/pip" "${SOURCES_ROOT}/macos/pip"; do \
      cd "$dir" && \
      for f in *.tar.gz *.tgz *.tar.bz2 *.tar.xz; do \
        [ -f "$f" ] || continue; \
        tar -xf "$f" && rm -f "$f"; \
      done; \
      for f in *.zip; do \
        [ -f "$f" ] || continue; \
        unzip -q -o "$f" && rm -f "$f"; \
      done; \
    done

# Unpack the DOCA dkms packages that are the source of the kernel driver (kept next to the .deb)
RUN for deb in "${SOURCES_ROOT}"/prereqs/doca-ofed-src*/*.deb; do \
      [ -f "$deb" ] && dpkg-deb -x "$deb" "${deb%.deb}"; \
    done; true

# Report what has no source, then fail on anything that should have had some
RUN for variant in cudaq cudaqx cudaq-decoding; do \
      echo "${variant} apt omitted packages:"; cat "${SOURCES_ROOT}/${variant}/apt/apt_omitted_packages.txt"; \
      echo "${variant} pip omitted packages:"; cat "${SOURCES_ROOT}/${variant}/pip/pip_omitted_packages.txt"; \
    done; \
    echo "macos pip omitted packages:"; cat "${SOURCES_ROOT}/macos/pip/macos_pip_omitted_packages.txt"
RUN bash "${SCRIPTS_DIR}"/check_package_sources.sh

# License, copyright and EULA files of every package, and the notice files for
# the apt and pip packages generated from them and from the sources above.
RUN set -e; \
    for variant in cudaq cudaqx cudaq-decoding macos; do \
      mkdir -p "${SOURCES_ROOT}/licenses/${variant}"; \
      [ -d "${SCRIPTS_DIR}/apt_licenses_${variant}" ] && cp -r "${SCRIPTS_DIR}/apt_licenses_${variant}" "${SOURCES_ROOT}/licenses/${variant}/apt"; \
      [ -d "${SCRIPTS_DIR}/pip_licenses_${variant}" ] && cp -r "${SCRIPTS_DIR}/pip_licenses_${variant}" "${SOURCES_ROOT}/licenses/${variant}/pip"; \
      mkdir -p "${SOURCES_ROOT}/licenses/${variant}/apt" "${SOURCES_ROOT}/licenses/${variant}/pip"; \
      suffix="${cuda_major:+_cu${cuda_major}}"; \
      python3 "${SCRIPTS_DIR}"/generate_pip_attribution.py "${SCRIPTS_DIR}/pip_packages_${variant}_trimmed.txt" \
        --pip-dir "${SOURCES_ROOT}/${variant}/pip" --pip-rel "${variant}/pip" \
        --licenses-dir "${SOURCES_ROOT}/licenses/${variant}/pip" --licenses-rel "licenses/${variant}/pip" \
        --aliases "${SCRIPTS_DIR}/pip_source_aliases.txt" \
        -o "${SOURCES_ROOT}/NOTICES/NOTICE_PIP_${variant}${suffix}"; \
      [ "$variant" = macos ] || python3 "${SCRIPTS_DIR}"/generate_apt_attribution.py "${SCRIPTS_DIR}/apt_package_details_${variant}_trimmed.tsv" \
        --licenses-dir "${SOURCES_ROOT}/licenses/${variant}/apt" --licenses-rel "licenses/${variant}/apt" \
        -o "${SOURCES_ROOT}/NOTICES/NOTICE_APT_${variant}${suffix}"; \
    done; \
    for f in "${SCRIPTS_DIR}"/images_*.txt; do \
      v="${f##*/images_}"; echo "${v%.txt}:"; sed 's/^/  /' "$f"; \
    done > "${SOURCES_ROOT}/IMAGES.txt"

# Summary
RUN echo "apt: $(find ${SOURCES_ROOT}/*/apt -maxdepth 1 -name '*.dsc' | wc -l) source packages" && \
    echo "pip: $(find ${SOURCES_ROOT}/*/pip -maxdepth 1 -mindepth 1 -type d | wc -l) sdists" && \
    echo "tpls: $(find ${SOURCES_ROOT}/tpls -maxdepth 1 -mindepth 1 -type d | wc -l) libraries" && \
    echo "prereqs: $(find ${SOURCES_ROOT}/prereqs -maxdepth 1 -mindepth 1 -type d | wc -l) components"

WORKDIR ${SOURCES_ROOT}
