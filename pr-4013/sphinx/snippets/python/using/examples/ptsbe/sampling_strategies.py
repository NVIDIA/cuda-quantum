# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

#[Begin PTSBE_Sampling]
import cudaq
from utils import bell, noise

# Reproducible probabilistic sampling
result = cudaq.ptsbe.sample(
    bell,
    shots_count=10_000,
    noise_model=noise,
    sampling_strategy=cudaq.ptsbe.ProbabilisticSamplingStrategy(seed=42),
)
print(result)

# Top-100 trajectories by probability
result = cudaq.ptsbe.sample(
    bell,
    shots_count=10_000,
    noise_model=noise,
    max_trajectories=100,
    sampling_strategy=cudaq.ptsbe.OrderedSamplingStrategy(),
)
print(result)
#[End PTSBE_Sampling]