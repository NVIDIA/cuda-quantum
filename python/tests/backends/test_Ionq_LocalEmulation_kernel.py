# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import cudaq.kernels
from cudaq import spin
import pytest
import os
from typing import List
import numpy as np


def assert_close(want, got, tolerance=1.0e-1) -> bool:
    return abs(want - got) < tolerance


@pytest.fixture(scope="function", autouse=True)
def configureTarget():
    # Set the targeted QPU
    cudaq.set_target('ionq', emulate='true')

    yield "Running the tests."

    cudaq.reset_target()


def test_Ionq_observe():
    cudaq.set_random_seed(13)

    @cudaq.kernel
    def ansatz_x():
        q = cudaq.qvector(1)

    s = cudaq.spin.x(0)
    res = cudaq.observe(ansatz_x, s, shots_count=10000)
    assert assert_close(0.0, res.expectation())

    @cudaq.kernel
    def ansatz_y():
        q = cudaq.qvector(4)
        x(q[0])

    s = cudaq.spin.y(3)
    res = cudaq.observe(ansatz_y, s, shots_count=10000)
    assert assert_close(0.0, res.expectation())

    @cudaq.kernel
    def ansatz_z():
        q = cudaq.qvector(2)
        x(q[0])

    s = cudaq.spin.z(0) * cudaq.spin.z(1)
    res = cudaq.observe(ansatz_z, s, shots_count=10000)
    counts = cudaq.sample(ansatz_z, shots_count=10000)

    assert assert_close(res.expectation(), counts.expectation())


def test_Ionq_cudaq_uccsd():

    num_electrons = 2
    num_qubits = 8

    thetas = [
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558
    ]

    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(num_qubits)
        for i in range(num_electrons):
            x(qubits[i])
        cudaq.kernels.uccsd(qubits, thetas, num_electrons, num_qubits)

    counts = cudaq.sample(kernel, shots_count=1000)
    assert len(counts) == 6
    assert '00000011' in counts
    assert '00000110' in counts
    assert '00010010' in counts
    assert '01000010' in counts
    assert '10000001' in counts
    assert '11000000' in counts


def test_Ionq_state_synthesis_from_simulator():

    @cudaq.kernel
    def kernel(state: cudaq.State):
        qubits = cudaq.qvector(state)

    state = cudaq.State.from_data(
        np.array([1. / np.sqrt(2.), 1. / np.sqrt(2.), 0., 0.], dtype=complex))

    counts = cudaq.sample(kernel, state)
    assert "00" in counts
    assert "10" in counts
    assert len(counts) == 2

    synthesized = cudaq.synthesize(kernel, state)
    counts = cudaq.sample(synthesized)
    assert '00' in counts
    assert '10' in counts
    assert len(counts) == 2


def test_Ionq_state_synthesis():

    @cudaq.kernel
    def init(n: int):
        q = cudaq.qvector(n)
        x(q[0])
        mz(q)

    @cudaq.kernel
    def kernel(s: cudaq.State):
        q = cudaq.qvector(s)
        x(q[1])
        mz(q)

    s = cudaq.get_state(init, 2)
    s = cudaq.get_state(kernel, s)
    counts = cudaq.sample(kernel, s)
    assert '10' in counts
    assert len(counts) == 1


def test_Ionq_trotter():

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
            tdOp = cudaq.SpinOperator(num_qubits=n_spins)
            for i in range(0, n_spins - 1):
                tdOp += (Jx * cudaq.spin.x(i) * cudaq.spin.x(i + 1))
                tdOp += (Jy * cudaq.spin.y(i) * cudaq.spin.y(i + 1))
                tdOp += (Jz * cudaq.spin.z(i) * cudaq.spin.z(i + 1))
            for i in range(0, n_spins):
                tdOp += (np.cos(omega * t) * cudaq.spin.x(i))
            return tdOp

        def termCoefficients(op: cudaq.SpinOperator) -> list[complex]:
            result = []
            ham.for_each_term(
                lambda term: result.append(term.get_coefficient()))
            return result

        def termWords(op: cudaq.SpinOperator) -> list[str]:
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

    run_steps(4, 5)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
