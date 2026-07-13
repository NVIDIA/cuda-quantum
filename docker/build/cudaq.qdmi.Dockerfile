# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                        #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This CI-only image keeps the optional QDMI dependency and its tests isolated
# from the standard CUDA-Q development image.
ARG base_image
FROM ${base_image} AS qdmi-build
SHELL ["/bin/bash", "-c"]

# Use the released MQT Core package by default. A source revision can be pinned
# while CUDA-Q depends on changes that have not been released yet. TODO: remove pin
ARG mqt_core_ref=
RUN if [ -n "$mqt_core_ref" ]; then \
        python3 -m pip install --break-system-packages \
          "git+https://github.com/munich-quantum-toolkit/core.git@$mqt_core_ref"; \
    else \
        python3 -m pip install --break-system-packages mqt-core; \
    fi

ARG workspace=.
ARG destination="$CUDAQ_REPO_ROOT"
ADD "$workspace" "$destination"
WORKDIR "$destination"

RUN CMAKE_BUILD_TYPE=Debug bash scripts/build_cudaq.sh -v -- \
      -DCUDAQ_ENABLE_QDMI_BACKEND=ON \
      -DCUDAQ_TEST_OMP_SLOTS=2 \
      "-DCMAKE_PREFIX_PATH=$(mqt-core-cli --cmake_dir)" && \
    grep -q '^CUDAQ_QDMI_DDSIM_AVAILABLE:INTERNAL=1$' \
      "$CUDAQ_REPO_ROOT/build/CMakeCache.txt"

FROM qdmi-build AS test
RUN cmake --build build --target check-targets
