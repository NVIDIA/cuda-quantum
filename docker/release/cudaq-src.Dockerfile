# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
# ============================================================================ #
#
# This image extends the CUDA-Q base runtime image by adding the source code
# of the third-party libraries (tpls) used by CUDA-Q (see NOTICE). Source is
# placed under /opt/nvidia/cudaq-tpls-src.
#
# Build from repo root with submodules initialized:
#   git submodule update --init --recursive
#   docker build -t nvcr.io/nvidia/nightly/cuda-quantum-src:cu12-latest-base \
#     -f docker/release/cudaq-src.Dockerfile .
#
# Optional build-args:
#   base_image  – base CUDA-Q image (default: nvcr.io/nvidia/nightly/cuda-quantum:cu12-latest-base)

ARG base_image=nvcr.io/nvidia/nightly/cuda-quantum:cu12-latest-base
FROM $base_image

USER root

# Copy only the third-party library source (tpls). Build context must have
# submodules checked out: git submodule update --init --recursive
ENV CUDAQ_TPL_SRC_ROOT=/opt/nvidia/cudaq-tpls-src
COPY tpls/ "${CUDAQ_TPL_SRC_ROOT}/tpls/"
COPY NOTICE LICENSE "${CUDAQ_TPL_SRC_ROOT}/"

RUN chown -R cudaq:cudaq "${CUDAQ_TPL_SRC_ROOT}"

USER cudaq
WORKDIR /home/cudaq
