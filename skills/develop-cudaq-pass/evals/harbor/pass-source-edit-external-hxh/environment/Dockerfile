# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

FROM ghcr.io/nvidia/cuda-quantum-dev:amd64-gcc12-main

ENV PATH="/workspaces/cuda-quantum/build/bin:/usr/local/cudaq/bin:${PATH}"
ENV LD_LIBRARY_PATH="/workspaces/cuda-quantum/build/lib:/usr/local/cudaq/lib:${LD_LIBRARY_PATH}"
ENV CUDAQ_SOURCE_DIR="/workspaces/cuda-quantum"
ENV CUDAQ_BUILD_DIR="/workspaces/cuda-quantum/build"

RUN git -C /workspaces/cuda-quantum rev-parse HEAD \
      > /opt/cudaq-eval-base-revision && \
    install -D -m 0555 /usr/local/llvm/bin/FileCheck \
      /opt/cudaq-eval-tools/FileCheck && \
    sha256sum \
      /workspaces/cuda-quantum/build/bin/cudaq-opt \
      /workspaces/cuda-quantum/build/bin/CircuitCheck \
      /opt/cudaq-eval-tools/FileCheck \
      > /opt/cudaq-eval-tool-sha256 && \
    rm -rf /workspaces/cuda-quantum/skills/develop-cudaq-pass && \
    mkdir -p /workspace/skills /logs/agent /logs/verifier

COPY skills/ /workspace/skills/
COPY skills/ /root/.claude/skills/
COPY skills/ /root/.agents/skills/
COPY skills/ /root/.config/opencode/skills/

RUN rm -rf \
    /workspace/skills/develop-cudaq-pass/evals \
    /root/.claude/skills/develop-cudaq-pass/evals \
    /root/.agents/skills/develop-cudaq-pass/evals \
    /root/.config/opencode/skills/develop-cudaq-pass/evals

WORKDIR /workspace
