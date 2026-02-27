# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq

@cudaq.kernel
def bell():
    q = cudaq.qvector(2)
    h(q[0])
    cx(q[0], q[1])
    mz(q)

noise = cudaq.NoiseModel()
noise.add_channel("h", [0],    cudaq.DepolarizationChannel(0.01))
noise.add_channel("x", [0, 1], cudaq.Depolarization2(0.005))