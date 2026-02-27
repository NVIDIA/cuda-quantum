# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# [Begin PTSBE_Shot_Allocation]
import cudaq
from utils import bell, noise

alloc = cudaq.ptsbe.ShotAllocationStrategy(
        cudaq.ptsbe.ShotAllocationType.LOW_WEIGHT_BIAS,
        bias_strength=2.0)

result = cudaq.ptsbe.sample(
    bell,
    shots_count=10_000,
    noise_model=noise,
    shot_allocation=alloc,
)
print(result)
# [End PTSBE_Shot_Allocation]