# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import time
import numpy as np
from typing import List

# Compute magnetization using Suzuki-Trotter approximation.
# This example demonstrates usage of quantum states in kernel mode.
#
# Details
# https://pubs.aip.org/aip/jmp/article-abstract/32/2/400/229229/General-theory-of-fractal-path-integrals-with
#
# Hamiltonian used
# https://en.m.wikipedia.org/wiki/Quantum_Heisenberg_model

# If you have a NVIDIA GPU you can use this example to see
# that the GPU-accelerated backends can easily handle a
# larger number of qubits compared the CPU-only backend.

# Depending on the available memory on your GPU, you can
# set the number of qubits to around 30 qubits, and un-comment
# the `cudaq.set_target(nvidia)` line.

# Note: Without setting the target to the `nvidia` backend,
# there will be a noticeable decrease in simulation performance.
# This is because the CPU-only backend has difficulty handling
# 30+ qubit simulations.

spins = 11  # set to around 25 qubits for `nvidia` target
steps = 10  # set to around 100 for `nvidia` target
# ```
# cudaq.set_target("nvidia")
# ```


# Alternating up/down spins
@cudaq.kernel
def getInitState(numSpins: int):
    q = cudaq.qvector(numSpins)
    for qId in range(0, numSpins, 2):
        x(q[qId])


# This performs a single-step Trotter on top of an initial state, e.g.,
# result state of the previous Trotter step.
@cudaq.kernel
def trotter(state: cudaq.State, coefficients: List[complex],
            words: List[cudaq.pauli_word], dt: float):
    q = cudaq.qvector(state)
    for i in range(len(coefficients)):
        exp_pauli(coefficients[i].real * dt, q, words[i])


def run_steps(steps: int, spins: int):
    g = 1.0
    Jx = 1.0
    Jy = 1.0
    Jz = g
    dt = 0.05
    n_steps = steps
    n_spins = spins
    omega = 2 * np.pi

    def heisenbergModelHam(t: float) -> cudaq.SpinOperator:
        tdOp = cudaq.SpinOperator(num_qubits=n_spins)
        for i in range(0, n_spins - 1):
            tdOp += (Jx * cudaq.spin.x(i) * cudaq.spin.x(i + 1))
            tdOp += (Jy * cudaq.spin.y(i) * cudaq.spin.y(i + 1))
            tdOp += (Jz * cudaq.spin.z(i) * cudaq.spin.z(i + 1))
        for i in range(0, n_spins):
            tdOp += (np.cos(omega * t) * cudaq.spin.x(i))
        return tdOp

    # Collect coefficients from a spin operator so we can pass them to a kernel
    def termCoefficients(op: cudaq.SpinOperator) -> List[complex]:
        result = []
        ham.for_each_term(lambda term: result.append(term.get_coefficient()))
        return result

    # Collect Pauli words from a spin operator so we can pass them to a kernel
    def termWords(op: cudaq.SpinOperator) -> List[str]:
        result = []
        ham.for_each_term(lambda term: result.append(term.to_string(False)))
        return result

    # Observe the average magnetization of all spins (<Z>)
    average_magnetization = cudaq.SpinOperator(num_qubits=n_spins)
    for i in range(0, n_spins):
        average_magnetization += ((1.0 / n_spins) * cudaq.spin.z(i))
    average_magnetization -= 1.0

    # Run loop
    state = cudaq.get_state(getInitState, n_spins)

    results = []
    times = []
    for i in range(0, n_steps):
        start_time = time.time()
        ham = heisenbergModelHam(i * dt)
        coefficients = termCoefficients(ham)
        words = termWords(ham)
        magnetization_exp_val = cudaq.observe(trotter, average_magnetization,
                                              state, coefficients, words, dt)
        result = magnetization_exp_val.expectation()
        results.append(result)
        state = cudaq.get_state(trotter, state, coefficients, words, dt)
        stepTime = time.time() - start_time
        times.append(stepTime)
        print(f"Step {i}: time [s]: {stepTime}, result: {result}")

    print("")
    # Print all the times and results (useful for plotting).
    print(f"Step times: {times}")
    print(f"Results: {results}")

    print("")


start_time = time.time()
run_steps(steps, spins)
print(f"Total time: {time.time() - start_time}s")
