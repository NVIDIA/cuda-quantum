# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Build from the CUDA-Q repository root:
#   docker build -f pulse/docker/Dockerfile -t cudaq-pulse-preview .
#
# Pulse compiles against the CUDA-Q toolchain shipped in the `cudaq-devel`
# wheel plus `libcudaqMLIR` from the `cudaq` runtime wheel, so this image needs
# neither the CUDA-Q development container nor a source build of LLVM. To build
# against locally produced wheels instead of the package index, place them in a
# directory inside the build context and pass, for example,
#   --build-arg pip_index_args="--no-index --find-links=dist"
ARG base_image=ubuntu:24.04
FROM ${base_image} AS pulse-build

ARG cudaq_version=
ARG pip_index_args=

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      g++-12 \
      python3 \
      python3-pip \
      python3-venv \
 && rm -rf /var/lib/apt/lists/*

ENV CC=gcc-12 CXX=g++-12

WORKDIR /opt/cuda-quantum
COPY . .

# One virtual environment holds the CUDA-Q toolchain, the build tools, and the
# test/documentation dependencies, so CMake's site-packages probe finds them.
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv "${VIRTUAL_ENV}"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN pip install --no-cache-dir ${pip_index_args} \
      "cudaq${cudaq_version:+==${cudaq_version}}" \
      "cudaq-devel${cudaq_version:+==${cudaq_version}}" \
 && pip install --no-cache-dir \
      "cmake>=4.0" \
      hypothesis \
      lit \
      "myst-parser>=3.0" \
      "nanobind>=2.12" \
      ninja \
      numpy \
      nvidia-sphinx-theme==0.0.8 \
      pytest==9.0.3 \
      "sphinx>=8.1,<8.3"

RUN cmake -S pulse -B build-pulse -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCUDAQ_BUILD_TESTS=ON \
      -DCUDAQ_PULSE_BUILD_DOCS=ON \
  && cmake --build build-pulse --parallel 2 \
      --target pulse

ENV PATH="/opt/cuda-quantum/build-pulse/bin:${PATH}"
ENV PYTHONPATH="/opt/cuda-quantum/build-pulse/python"

FROM pulse-build AS pulse-ci
RUN cmake --build build-pulse --parallel 2 --target check-pulse pulse-docs

FROM pulse-build AS pulse
CMD ["bash"]
