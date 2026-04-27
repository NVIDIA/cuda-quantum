# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# [Begin PTSBE_Execution_Data]
import cudaq
from utils import bell, noise

result = cudaq.ptsbe.sample(
    bell,
    shots_count=1_000,
    noise_model=noise,
    return_execution_data=True,
)

data = result.ptsbe_execution_data

# Circuit structure. For Noise instructions, ``inst.params`` carries the
# channel's numeric parameters and ``inst.channel`` is a ``cudaq.KrausChannel``
# exposing ``.noise_type``, ``.parameters``, and ``.get_ops()``. For Gate and
# Measurement instructions ``inst.channel`` is ``None``.
Noise = cudaq.ptsbe.TraceInstructionType.Noise
for inst in data.instructions:
    print(inst.type, inst.name, inst.targets)
    if inst.type == Noise:
        print(f"  params={list(inst.params)}  "
              f"noise_type={inst.channel.noise_type}  "
              f"num_kraus_ops={len(inst.channel.get_ops())}")

# Trajectory details
for trajectory in data.trajectories:
    print(f"id={trajectory.trajectory_id}  p={trajectory.probability:.4f}"
          f"  shots={trajectory.num_shots}")
# [End PTSBE_Execution_Data]
