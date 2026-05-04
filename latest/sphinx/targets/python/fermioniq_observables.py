# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os
import cudaq
from cudaq import spin

remote_config = os.environ.get("FERMIONIQ_REMOTE_CONFIG_ID", "")
project_id = os.environ.get("FERMIONIQ_PROJECT_ID", "")

# You only have to set the target once! No need to redefine it
# for every execution call on your kernel.
cudaq.set_target("fermioniq", **{
    "remote_config": remote_config,
    "project_id": project_id
})


@cudaq.kernel
def kernel(theta: float):
    qvector = cudaq.qvector(2)
    x(qvector[0])
    ry(theta, qvector[1])
    x.ctrl(qvector[1], qvector[0])


spin_operator = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
    0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

# Pre-computed angle that minimizes the energy expectation of the `spin_operator`.
angle = 0.59

energy = cudaq.observe(kernel, spin_operator, angle).expectation()
print(f"Energy is {energy}")
