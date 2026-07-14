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

# TODO: Use the released package once
# https://github.com/munich-quantum-toolkit/core/pull/1887 is released.
RUN python3 -m pip install --break-system-packages \
      "git+https://github.com/munich-quantum-toolkit/core.git@def69f1517602ba802d1401ab97e6fa331e99d2e"

ARG workspace=.
ARG destination="$CUDAQ_REPO_ROOT"
ADD "$workspace" "$destination"
WORKDIR "$destination"

RUN CMAKE_BUILD_TYPE=Debug bash scripts/build_cudaq.sh -v -- \
      -DCUDAQ_ENABLE_QDMI_BACKEND=ON \
      -DCUDAQ_TEST_OMP_SLOTS=2 \
      "-DCMAKE_PREFIX_PATH=$(mqt-core-cli --cmake_dir)" && \
    grep -q 'config.cudaq_backends_qdmi = "1"' \
      "$CUDAQ_REPO_ROOT/build/targettests/lit.site.cfg.py"

FROM qdmi-build AS test
RUN cmake --build build --target check-targets
