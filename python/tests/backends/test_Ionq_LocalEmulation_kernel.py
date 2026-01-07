# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
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
        np.array([1. / np.sqrt(2.), 1. / np.sqrt(2.), 0., 0.],
                 dtype=cudaq.complex()))

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

    @cudaq.kernel
    def kernel(s: cudaq.State):
        q = cudaq.qvector(s)
        x(q[1])

    s = cudaq.get_state(init, 2)
    s = cudaq.get_state(kernel, s)
    counts = cudaq.sample(kernel, s)
    assert '10' in counts
    assert len(counts) == 1


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
