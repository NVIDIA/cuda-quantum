# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../.. python3 %s
# RUN: PYTHONPATH=../../.. python3 %s --target quantinuum --emulate

import cudaq
from cudaq import spin


@cudaq.kernel
def ansatz(angle: float):
    q = cudaq.qvector(2)
    x(q[0])
    ry(angle, q[1])
    x.ctrl(q[1], q[0])


hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
    0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

cudaq.set_random_seed(13)

energy = cudaq.observe(ansatz, hamiltonian, .59).expectation()
print('Energy is {}'.format(energy))

# CHECK: Energy is -1.
