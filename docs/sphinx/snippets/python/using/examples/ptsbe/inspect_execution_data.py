# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# [Begin PTSBE_Execution_Data]
from cudaq
from utils import bell, noise

result = cudaq.ptsbe.sample(
          bell,
          shots_count=1_000,
          noise_model=noise,
          return_execution_data=True,
        )

data = result.ptsbe_execution_data

# Circuit structure
for inst in data.instructions:
    print(inst.type, inst.name, inst.targets)

# Trajectory details
for trajectory in data.trajectories:
    print(f"id={trajectory.trajectory_id}  p={trajectory.probability:.4f}"
        f"  shots={trajectory.num_shots}")
# [End PTSBE_Execution_Data]