# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Build from the CUDA-Q repository root:
#   docker build -f pulse/docker/Dockerfile -t cudaq-pulse-preview .
# The LLVM-tagged CUDA-Q development image already contains the repository's
# pinned LLVM/MLIR toolchain; this Dockerfile only builds pulse.
ARG base_image=ghcr.io/nvidia/cuda-quantum-devcontainer:amd64-cu12.6-llvm-main
FROM ${base_image} AS pulse-build

USER root
WORKDIR /opt/cuda-quantum

COPY . .

RUN python3 -m pip install --no-cache-dir --break-system-packages \
      hypothesis \
      lit \
      "myst-parser>=3.0" \
      nvidia-sphinx-theme==0.0.8 \
      numpy==1.26.4 \
      pytest==9.0.3 \
      "sphinx>=8.1,<8.3"

RUN cmake -S . -B build-pulse -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCUDAQ_BUILD_TESTS=ON \
      -DCUDAQ_DISABLE_CPP_FRONTEND=ON \
      -DCUDAQ_ENABLE_ALL_BACKEND=OFF \
      -DCUDAQ_ENABLE_ENSMALLEN=OFF \
      -DCUDAQ_ENABLE_NLOPT=OFF \
      -DCUDAQ_ENABLE_PASQAL_QRMI_CONNECTOR=OFF \
      -DCUDAQ_ENABLE_PROJECTS=pulse \
      -DCUDAQ_ENABLE_REST=OFF \
      -DCUDAQ_PULSE_BUILD_DOCS=ON \
      -DLLVM_DIR="${LLVM_INSTALL_PREFIX}/lib/cmake/llvm" \
      -DPython_EXECUTABLE=/usr/bin/python3 \
  && cmake --build build-pulse --parallel 2 \
      --target pulse

ENV PATH="/opt/cuda-quantum/build-pulse/bin:${PATH}"
ENV PYTHONPATH="/opt/cuda-quantum/build-pulse/python"

FROM pulse-build AS pulse-ci
RUN cmake --build build-pulse --parallel 2 --target check-pulse pulse-docs

FROM pulse-build AS pulse
CMD ["bash"]
