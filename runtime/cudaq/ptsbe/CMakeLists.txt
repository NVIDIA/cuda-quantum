# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# PTSBE (Pre-Trajectory Sampling with Batched Execution) subdirectory
# This module provides efficient noisy quantum circuit simulation through
# pre-sampling of noise trajectories and batched execution.

# Source files
set(PTSBE_SOURCES
    PTSBESampleIntegration.cpp
    PTSBESampler.cpp
    ShotAllocationStrategy.cpp
)

# Install headers
install(FILES
    KrausSelection.h
    KrausTrajectory.h
    PTSBESampler.h
    PTSBEResult.h
    PTSBESampleIntegration.h
    ShotAllocationStrategy.h
    DESTINATION include/cudaq/ptsbe
)
