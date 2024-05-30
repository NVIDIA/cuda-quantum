# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import pytest

import cudaq
import numpy as np

def test_trotter():

    # Alternating up/down spins
    @cudaq.kernel
    def getInitState(numSpins:int):
        q = cudaq.qvector(numSpins)
        for qId in range(0, numSpins, 2):
            x(q[qId])
        

    # This performs a single-step Trotter on top of an initial state, e.g.,
    # result state of the previous Trotter step.
    @cudaq.kernel
    def trotter(initialState:cudaq.State, data:list[float], n_spins:int, dt:float):
        ham = cudaq.SpinOperator(data, n_spins)
        q = cudaq.qvector(initialState)
        ham.for_each_term(
            lambda term:
                cudaq.exp_pauli(term.get_coefficient().real() * dt, q, term.get_coefficient().to_string(False))
            )


    g = 1.0
    Jx = 1.0
    Jy = 1.0
    Jz = g
    dt = 0.05
    n_steps = 100
    n_spins = 25
    omega = 2 * np.pi

    def heisenbergModelHam(t:float) -> cudaq.SpinOperator:
        tdOp = cudaq.SpinOperator(num_qubits=n_spins)
        for i in range(0, n_spins-1):
            tdOp += (Jx * cudaq.spin.x(i) * cudaq.spin.x(i + 1))
            tdOp += (Jy * cudaq.spin.x(i) * cudaq.spin.y(i + 1))
            tdOp += (Jz * cudaq.spin.x(i) * cudaq.spin.z(i + 1))
        for i in range(0, n_spins):
            tdOp += (np.cos(omega * t) * cudaq.spin.x(i))
        return tdOp

    # Observe the average magnetization of all spins (<Z>)
    average_magnetization = 0.0
    for i in range(0, n_spins):
        average_magnetization += ((1.0 / n_spins) * cudaq.spin.z(i))
    average_magnetization -= 1.0

    # Run loop
    state = cudaq.get_state(getInitState, n_spins)

    exp_results = []
    for i in range(0, n_steps):
        ham = heisenbergModelHam(i * dt)
        magnetization_exp_val = cudaq.observe(trotter, average_magnetization, state, ham.serialize(), n_spins, dt)
        exp_results.append(magnetization_exp_val.expectation())
        state = cudaq.get_state(trotter, state, ham, dt)

    for result in exp_results:
        print(result)
        assert result == 0

