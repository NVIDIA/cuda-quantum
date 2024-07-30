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


# def test_vector():
#     cudaq.reset_target()
#     cudaq.set_target("remote-mqpu", url="localhost:3030")
#     #cudaq.set_target("quantinuum", emulate=True)

#     c = [1. / np.sqrt(2.), 1. / np.sqrt(2.), 0., 0.]

#     @cudaq.kernel
#     def kernel(r: int):
#         q = cudaq.qvector(c)

#     synthesized = cudaq.synthesize(kernel, 0)
#     counts = cudaq.sample(synthesized)
#     print(counts)

# #test_vector()

# def test_state_from_data():
#     cudaq.reset_target()
#     cudaq.set_target("remote-mqpu", url="localhost:3030")
#     #cudaq.set_target("quantinuum", emulate=True)


#     c = np.array([1. / np.sqrt(2.),  1. / np.sqrt(2.), 0., 0.],
#                     dtype=complex)
#     print(cudaq.complex())
#     state = cudaq.State.from_data(c)

#     @cudaq.kernel
#     def kernel(s: cudaq.State):
#         q = cudaq.qvector(s)

#     counts = cudaq.sample(kernel, state)
#     print(counts)

# #test_state_from_data()

# def test_state_from_another_kernel():
#     cudaq.reset_target()
#     cudaq.set_target("remote-mqpu", url="localhost:3030")
#     #cudaq.set_target("quantinuum", emulate=True)

#     @cudaq.kernel
#     def initState(n: int):
#         q = cudaq.qvector(n)
#         ry(np.pi/2, q[0])

#     state = cudaq.get_state(initState, 2)

#     @cudaq.kernel
#     def kernel(s: cudaq.State):
#         q = cudaq.qvector(s)

#     counts = cudaq.sample(kernel, state)
#     print(counts)

# #test_state_from_another_kernel()

# #################################################
# # @cudaq.kernel
# # def kernel(angles: list[float], num_qubits: int):
# #     qvector = cudaq.qvector(num_qubits)
# #     x(qvector[0])
# #     ry(angles[0], qvector[1])
# #     x.ctrl(qvector[1], qvector[0])

# # counts = cudaq.sample(kernel, [0.0, 1.0], 2)

# # print(counts)

# def assert_close(want, got, tolerance=1.e-5) -> bool:
#     return abs(want - got) < tolerance

# def test_optimizer():
#     cudaq.reset_target()
#     cudaq.set_target("remote-mqpu", url="localhost:3030")
    
#     hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
#         0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

#     # Verify that variables can be captured by kernels
#     testVar = 0

#     @cudaq.kernel
#     def kernel(angles: list[float]):
#         qvector = cudaq.qvector(2)
#         x(qvector[0])
#         ry(angles[0] + testVar, qvector[1])
#         x.ctrl(qvector[1], qvector[0])

#     optimizer = cudaq.optimizers.Adam()
#     gradient = cudaq.gradients.CentralDifference()

#     def objective_function(parameter_vector: list[float],
#                            hamiltonian=hamiltonian,
#                            gradient_strategy=gradient,
#                            kernel=kernel) -> tuple[float, list[float]]:
#         get_result = lambda parameter_vector: cudaq.observe(
#             kernel, hamiltonian, parameter_vector).expectation()
#         cost = get_result(parameter_vector)
#         gradient_vector = gradient_strategy.compute(parameter_vector,
#                                                     get_result, cost)
#         return cost, gradient_vector

#     energy, parameter = optimizer.optimize(dimensions=1,
#                                            function=objective_function)
#     print(f"\nminimized <H> = {round(energy,16)}")
#     print(f"optimal theta = {round(parameter[0],16)}")
#     assert assert_close(energy, -1.7483830311526454, 1e-3)
#     assert assert_close(parameter[0], 0.5840908448487905, 1e-3)

# test_optimizer()

import cudaq

kernel = cudaq.make_kernel()

qubits = kernel.qalloc(2)
kernel.h(qubits[0])
kernel.cx(qubits[0], qubits[1])

counts = cudaq.sample(kernel)
counts.dump()