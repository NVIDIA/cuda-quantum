# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
# [Begin Documentation]
import cudaq
from cudaq import spin

cudaq.set_target("nvidia-mqpu")
target = cudaq.get_target()
num_qpus = target.num_qpus()
print("Number of QPUs:", num_qpus)

# Define spin ansatz.
kernel, theta = cudaq.make_kernel(float)
qreg = kernel.qalloc(2)
kernel.x(qreg[0])
kernel.ry(theta, qreg[1])
kernel.cx(qreg[1], qreg[0])
# Define spin Hamiltonian.
hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
    0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

exp_val = cudaq.observe(kernel,
                        hamiltonian,
                        0.59,
                        execution=cudaq.parallel.thread).expectation_z()
print("Expectation value: ", exp_val)
