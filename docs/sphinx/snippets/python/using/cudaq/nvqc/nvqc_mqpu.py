# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
# [Begin Documentation]
import cudaq
from cudaq import spin
import math

# Use NVQC with 3 virtual QPUs
cudaq.set_target("nvqc", nqpus=3)

print("Number of QPUs:", cudaq.get_target().num_qpus())
# Create the parameterized ansatz
kernel, theta = cudaq.make_kernel(float)
qreg = kernel.qalloc(2)
kernel.x(qreg[0])
kernel.ry(theta, qreg[1])
kernel.cx(qreg[1], qreg[0])

# Define its spin Hamiltonian.
hamiltonian = (5.907 - 2.1433 * spin.x(0) * spin.x(1) -
               2.1433 * spin.y(0) * spin.y(1) + 0.21829 * spin.z(0) -
               6.125 * spin.z(1))


def opt_gradient(parameter_vector):
    # Evaluate energy and gradient on different remote QPUs
    # (i.e., concurrent job submissions to NVQC)
    energy_future = cudaq.observe_async(kernel,
                                        hamiltonian,
                                        parameter_vector[0],
                                        qpu_id=0)
    plus_future = cudaq.observe_async(kernel,
                                      hamiltonian,
                                      parameter_vector[0] + 0.5 * math.pi,
                                      qpu_id=1)
    minus_future = cudaq.observe_async(kernel,
                                       hamiltonian,
                                       parameter_vector[0] - 0.5 * math.pi,
                                       qpu_id=2)
    return (energy_future.get().expectation(), [
        (plus_future.get().expectation() - minus_future.get().expectation()) /
        2.0
    ])


optimizer = cudaq.optimizers.LBFGS()
optimal_value, optimal_parameters = optimizer.optimize(1, opt_gradient)
print("Ground state energy =", optimal_value)
print("Optimal parameters =", optimal_parameters)
