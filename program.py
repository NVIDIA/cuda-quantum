# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
from cudaq import spin
from typing import Callable
import numpy as np

import cupy as cp

# #cudaq.set_target('quantinuum', emulate=True)
# #cudaq.set_target("remote-mqpu", auto_launch="1")

# def trotter():

#     # Alternating up/down spins
#     @cudaq.kernel
#     def getInitState(numSpins: int):
#         q = cudaq.qvector(numSpins)
#         for qId in range(0, numSpins, 2):
#             x(q[qId])

#     # This performs a single-step Trotter on top of an initial state, e.g.,
#     # result state of the previous Trotter step.
#     @cudaq.kernel
#     def trotter(state: cudaq.State, coefficients: list[complex],
#                 words: list[cudaq.pauli_word], dt: float):
#         q = cudaq.qvector(state)
#         for i in range(len(coefficients)):
#             exp_pauli(coefficients[i].real * dt, q, words[i])

#     def run_steps(steps: int, spins: int):
#         g = 1.0
#         Jx = 1.0
#         Jy = 1.0
#         Jz = g
#         dt = 0.05
#         n_steps = steps
#         n_spins = spins
#         omega = 2 * np.pi

#         def heisenbergModelHam(t: float) -> cudaq.SpinOperator:
#             tdOp = cudaq.SpinOperator(num_qubits=n_spins)
#             for i in range(0, n_spins - 1):
#                 tdOp += (Jx * cudaq.spin.x(i) * cudaq.spin.x(i + 1))
#                 tdOp += (Jy * cudaq.spin.y(i) * cudaq.spin.y(i + 1))
#                 tdOp += (Jz * cudaq.spin.z(i) * cudaq.spin.z(i + 1))
#             for i in range(0, n_spins):
#                 tdOp += (np.cos(omega * t) * cudaq.spin.x(i))
#             return tdOp

#         def termCoefficients(op: cudaq.SpinOperator) -> list[complex]:
#             result = []
#             ham.for_each_term(
#                 lambda term: result.append(term.get_coefficient()))
#             return result

#         def termWords(op: cudaq.SpinOperator) -> list[str]:
#             result = []
#             ham.for_each_term(lambda term: result.append(term.to_string(False)))
#             return result

#         # Observe the average magnetization of all spins (<Z>)
#         average_magnetization = cudaq.SpinOperator(num_qubits=n_spins)
#         for i in range(0, n_spins):
#             average_magnetization += ((1.0 / n_spins) * cudaq.spin.z(i))
#         average_magnetization -= 1.0

#         # Run loop
#         state = cudaq.get_state(getInitState, n_spins)

#         exp_results = []
#         for i in range(0, n_steps):
#             ham = heisenbergModelHam(i * dt)
#             coefficients = termCoefficients(ham)
#             words = termWords(ham)
#             magnetization_exp_val = cudaq.observe(trotter,
#                                                   average_magnetization, state,
#                                                   coefficients, words, dt)
#             exp_results.append(magnetization_exp_val.expectation())
#             state = cudaq.get_state(trotter, state, coefficients, words, dt)

#         for result in exp_results:
#             print(result)
#             #assert -1.0 <= result and result < 0.

#     #run_steps(10, 11)
#     run_steps(10, 11)

# trotter()


def test_quantinuum_state_synthesis():
    kernel, state = cudaq.make_kernel(cudaq.State)
    qubits = kernel.qalloc(state)

    state = cudaq.State.from_data(
        np.array([1. / np.sqrt(2.), 1. / np.sqrt(2.), 0., 0.], dtype=complex))

    counts = cudaq.sample(kernel, state)
    print(counts)
    assert "00" in counts
    assert "10" in counts
    assert "01" not in counts
    assert "11" not in counts

    synthesized = cudaq.synthesize(kernel, state)
    counts = cudaq.sample(synthesized)
    print(counts)
    assert '00' in counts
    assert '10' in counts
    assert len(counts) == 2

cudaq.set_target('quantinuum', emulate=True)
test_quantinuum_state_synthesis()