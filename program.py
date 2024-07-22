# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
from cudaq import spin
import numpy as np

cudaq.set_target("remote-mqpu", url="localhost:3030")

def test_complex_vqe_named_lambda(optimizer, gradient):
    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

    @cudaq.kernel
    def kernel(angles: list[float], num_qubits: int):
        qvector = cudaq.qvector(num_qubits)
        x(qvector[0])
        ry(angles[0], qvector[1])
        x.ctrl(qvector[1], qvector[0])

    num_qubits = 2
    arg_mapper = lambda x: (x, num_qubits)
    energy, parameter = cudaq.vqe(kernel=kernel,
                                  gradient_strategy=gradient,
                                  spin_operator=hamiltonian,
                                  optimizer=optimizer,
                                  argument_mapper=arg_mapper,
                                  parameter_count=1)

    print(f"\nminimized <H> = {round(energy,16)}")
    print(f"optimal theta = {round(parameter[0],16)}")
    want_expectation_value = -1.7487948611472093
    want_optimal_parameters = [0.59]


# @pytest.mark.parametrize("optimizer", [
#     cudaq.optimizers.LBFGS(),
#     cudaq.optimizers.Adam(),
#     cudaq.optimizers.GradientDescent(),
#     cudaq.optimizers.SGD(),
# ])
def test_complex_vqe_named_lambda_sweep_opt(optimizer):
    test_complex_vqe_named_lambda(optimizer,
                                  cudaq.gradients.CentralDifference())

test_complex_vqe_named_lambda_sweep_opt(cudaq.optimizers.LBFGS())

# cudaq.reset_target()
# #cudaq.set_target("remote-mqpu", url="localhost:3030")#
# cudaq.set_target("quantinuum")

# c = np.ndarray([1. / np.sqrt(2.), 1. / np.sqrt(2.), 0., 0.], dtype=np.complex128)

# @cudaq.kernel
# def kernel():
#     q = cudaq.qvector(c)

# counts = cudaq.sample(kernel)

# print(counts)


#################################################
# @cudaq.kernel
# def kernel(angles: list[float], num_qubits: int):
#     qvector = cudaq.qvector(num_qubits)
#     x(qvector[0])
#     ry(angles[0], qvector[1])
#     x.ctrl(qvector[1], qvector[0])

# counts = cudaq.sample(kernel, [0.0, 1.0], 2)

# print(counts)