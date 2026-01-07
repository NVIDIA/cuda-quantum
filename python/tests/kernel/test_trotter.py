# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import pytest

import cudaq
import numpy as np

skipIfNvidiaFP64NotInstalled = pytest.mark.skipif(
    not (cudaq.num_available_gpus() > 0 and cudaq.has_target('nvidia-fp64')),
    reason='Could not find nvidia-fp64 in installation')

skipIfNvidiaNotInstalled = pytest.mark.skipif(
    not (cudaq.num_available_gpus() > 0 and cudaq.has_target('nvidia')),
    reason='Could not find nvidia in installation')


def trotter():

    # Alternating up/down spins
    @cudaq.kernel
    def getInitState(numSpins: int):
        q = cudaq.qvector(numSpins)
        for qId in range(0, numSpins, 2):
            x(q[qId])

    # This performs a single-step Trotter on top of an initial state, e.g.,
    # result state of the previous Trotter step.
    @cudaq.kernel
    def trotter(state: cudaq.State, coefficients: list[complex],
                words: list[cudaq.pauli_word], dt: float):
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
            tdOp = cudaq.SpinOperator.empty()
            for i in range(0, n_spins - 1):
                tdOp += (Jx * cudaq.spin.x(i) * cudaq.spin.x(i + 1))
                tdOp += (Jy * cudaq.spin.y(i) * cudaq.spin.y(i + 1))
                tdOp += (Jz * cudaq.spin.z(i) * cudaq.spin.z(i + 1))
            for i in range(0, n_spins):
                tdOp += (np.cos(omega * t) * cudaq.spin.x(i))
            print(tdOp)
            return tdOp

        def termCoefficients(op: cudaq.SpinOperator) -> list[complex]:
            result = []
            for term in op:
                result.append(term.get_coefficient())
            return result

        def termWords(op: cudaq.SpinOperator) -> list[str]:
            result = []
            for term in op:
                # The way the trotter kernel is written, it
                # wants exp pauli to act on the entire state.
                # That means we need to make it explicit that each term
                # in this Hamiltonian indeed is supposed to act on each qubit.
                for i in range(0, n_spins):
                    term *= cudaq.spin.i(i)
                result.append(term.get_pauli_word())
            return result

        # Observe the average magnetization of all spins (<Z>)
        average_magnetization = cudaq.SpinOperator.empty()
        for i in range(0, n_spins):
            average_magnetization += ((1.0 / n_spins) * cudaq.spin.z(i))

        # Run loop
        state = cudaq.get_state(getInitState, n_spins)

        exp_results = []
        for i in range(0, n_steps):
            ham = heisenbergModelHam(i * dt)
            coefficients = termCoefficients(ham)
            words = termWords(ham)
            magnetization_exp_val = cudaq.observe(trotter,
                                                  average_magnetization, state,
                                                  coefficients, words, dt)
            exp_results.append(magnetization_exp_val.expectation())
            state = cudaq.get_state(trotter, state, coefficients, words, dt)

        for result in exp_results:
            assert -1.0 <= result and result < 0.

    run_steps(10, 11)


@skipIfNvidiaFP64NotInstalled
def test_trotter_f64():
    trotter()


@skipIfNvidiaNotInstalled
def test_trotter_f32():
    trotter()
