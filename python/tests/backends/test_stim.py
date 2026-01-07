# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os
from typing import List
import pytest

import cudaq
import numpy as np


@pytest.fixture(scope="session", autouse=True)
def setTarget():
    cudaq.set_target('stim')
    yield
    cudaq.reset_target()


def test_stim_non_clifford():

    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(10)
        rx(0.1, qubits[0])

    with pytest.raises(RuntimeError) as e:
        # Cannot perform non-Clifford gates in Stim simulator
        cudaq.sample(kernel)
    assert 'Gate not supported by Stim simulator' in repr(e)


def test_stim_toffoli_gates():

    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(10)
        cx(qubits[0:9], qubits[9])

    with pytest.raises(RuntimeError) as e:
        # Cannot perform Toffoli gates in Stim simulator
        cudaq.sample(kernel)
    assert 'Gates with >1 controls not supported by Stim simulator' in repr(e)


def test_stim_sample():
    # Create the kernel we'd like to execute on Stim
    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(250)
        h(qubits[0])
        # Stim is a Clifford-only simulator, so it can do many qubits.
        for i in range(1, 250):
            cx(qubits[i - 1], qubits[i])
        mz(qubits)

    counts = cudaq.sample(kernel)
    assert (len(counts) == 2)
    assert ('0' * 250 in counts)
    assert ('1' * 250 in counts)


def test_stim_all_mz_types():
    # Create the kernel we'd like to execute on Stim
    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(10)
        mx(qubits)
        my(qubits)
        mz(qubits)

    counts = cudaq.sample(kernel)
    assert (len(counts) > 1)


def test_stim_state_preparation():

    @cudaq.kernel
    def kernel(vec: List[complex]):
        qubits = cudaq.qvector(vec)

    with pytest.raises(RuntimeError) as e:
        # Cannot initialize qubits from state data in this simulator
        state = [1. / np.sqrt(2.), 1. / np.sqrt(2.), 0., 0.]
        cudaq.sample(kernel, state)


def test_stim_state_preparation_builder():
    kernel, state = cudaq.make_kernel(List[complex])
    qubits = kernel.qalloc(state)

    with pytest.raises(RuntimeError) as e:
        # Cannot initialize qubits from state data in this simulator
        state = [1. / np.sqrt(2.), 1. / np.sqrt(2.), 0., 0.]
        cudaq.sample(kernel, state)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
