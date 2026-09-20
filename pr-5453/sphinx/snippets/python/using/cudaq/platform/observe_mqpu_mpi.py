# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
# [Begin Documentation]
import cudaq
from cudaq import spin

cudaq.mpi.initialize()
cudaq.set_target("nvidia", option="mqpu")


# Define spin ansatz.
@cudaq.kernel
def kernel(angle: float):
    qvector = cudaq.qvector(2)
    x(qvector[0])
    ry(angle, qvector[1])
    x.ctrl(qvector[1], qvector[0])


# Define spin Hamiltonian.
hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
    0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

exp_val = cudaq.observe(kernel, hamiltonian, 0.59,
                        execution=cudaq.parallel.mpi).expectation()
if cudaq.mpi.rank() == 0:
    print("Expectation value: ", exp_val)

cudaq.mpi.finalize()
