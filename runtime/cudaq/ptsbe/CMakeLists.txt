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
    PTSBEExecutionData.cpp
    PTSBESample.cpp
    PTSBESampleResult.cpp
    PTSBESampler.cpp
    ShotAllocationStrategy.cpp
    TrajectoryDeduplication.cpp
    strategies/ProbabilisticSamplingStrategy.cpp
    strategies/ExhaustiveSamplingStrategy.cpp
    strategies/OrderedSamplingStrategy.cpp
    strategies/ConditionalSamplingStrategy.cpp
    NoiseExtractor.cpp
)

# Install headers
install(FILES
    KrausSelection.h
    KrausTrajectory.h
    TrajectoryDeduplication.h
    PTSBEOptions.h
    PTSBESampler.h
    PTSBESampleResult.h
    PTSBEExecutionData.h
    PTSBESample.h
    ShotAllocationStrategy.h
    PTSSamplingStrategy.h
    NoiseExtractor.h
    DESTINATION include/cudaq/ptsbe
)

# Install strategy headers to strategies subdirectory
install(FILES
    strategies/ProbabilisticSamplingStrategy.h
    strategies/ExhaustiveSamplingStrategy.h
    strategies/OrderedSamplingStrategy.h
    strategies/ConditionalSamplingStrategy.h
    DESTINATION include/cudaq/ptsbe/strategies
)
