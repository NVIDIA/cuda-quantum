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


@pytest.mark.parametrize("type_", [float, int])
def test_quake_value_operators(type_):
    """
    Test `cudaq.QuakeValue` and each of its binary operators
    for every applicable `QuakeValue` type.
    """
    kernel, value_0, value_1 = cudaq.make_kernel(type_, type_)
    # Checking the binary operators.
    # Multiplication.
    test = value_0 * 1.
    # Ensure we're returning a `QuakeValue`.
    assert type(test) == cudaq.QuakeValue
    # Ensure this is a new `QuakeValue`.
    assert test != value_0
    test = value_0 * value_1
    assert type(test) == cudaq.QuakeValue
    assert test != value_0 and test != value_1
    test = 1. * value_0
    assert type(test) == cudaq.QuakeValue
    assert test != value_0

    # Addition.
    test = value_0 + 1.
    assert type(test) == cudaq.QuakeValue
    assert test != value_0
    test = value_0 + value_1
    assert type(test) == cudaq.QuakeValue
    assert test != value_0 and test != value_1
    test = 1 + value_0
    assert type(test) == cudaq.QuakeValue
    assert test != value_0

    # Subtraction.
    test = value_0 - 1.
    assert type(test) == cudaq.QuakeValue
    assert test != value_0
    test = value_0 - value_1
    assert type(test) == cudaq.QuakeValue
    assert test != value_0 and test != value_1
    test = 1 - value_0
    assert type(test) == cudaq.QuakeValue
    assert test != value_0

    # Negation.
    test = -value_0
    assert type(test) == cudaq.QuakeValue
    assert test != value_0
    test = -value_1
    assert type(test) == cudaq.QuakeValue
    assert test != value_1


def test_QuakeValueLifetimeAndPrint():
    """Tests Bug #64 for the lifetime of a QuakeValue"""
    circuit = cudaq.make_kernel()
    qubitRegister = circuit.qalloc(2)
    circuit.x(qubitRegister[0])
    s = str(circuit)
    print(s)

    assert s.count('quake.x') == 1

    circuit.x(qubitRegister[0])
    s = str(circuit)
    print(s)
    assert s.count('quake.x') == 2


def test_QuakeValueDivOp():
    """Tests division operators"""
    kernel1, theta = cudaq.make_kernel(list)
    qubit1 = kernel1.qalloc(1)
    # Division of a QuakeValue
    kernel1.rx(theta[0] / 8.0, qubit1[0])
    state1 = cudaq.get_state(kernel1, [np.pi])
    # Verification
    kernel2 = cudaq.make_kernel()
    qubit2 = kernel2.qalloc(1)
    kernel2.rx(np.pi / 8.0, qubit2[0])
    state2 = cudaq.get_state(kernel2)
    assert np.allclose(state1, state2)

    # RHS division: float/QuakeValue
    kernel3, factor = cudaq.make_kernel(list)
    qubit3 = kernel3.qalloc(1)
    kernel3.rx(np.pi / factor[0], qubit3[0])
    state3 = cudaq.get_state(kernel3, [8.0])
    assert np.allclose(state3, state2)

    # QuakeValue/QuakeValue division
    kernel4, kernel4_args = cudaq.make_kernel(list)
    qubit4 = kernel4.qalloc(1)
    kernel4.rx(kernel4_args[0] / kernel4_args[1], qubit4[0])
    state4 = cudaq.get_state(kernel4, [np.pi, 8.0])
    assert np.allclose(state4, state2)


def test_QuakeValueInForLoop():
    """Tests QuakeValue used in as loop indices"""
    kernel, start, stop = cudaq.make_kernel(int, int)
    qubits = kernel.qalloc(stop)
    kernel.h(qubits[0])

    def foo(index: int):
        """A function that will be applied to `kernel` in a for loop."""
        kernel.x(qubits[index])

    # `QuakeValue` as start index
    kernel.for_loop(start, 1, foo)
    kernel.for_loop(start, stop - 1, foo)
    print(kernel)

    # Execute the kernel, passing along concrete values for the
    # `start` and `stop` arguments.
    counts = cudaq.sample(kernel, 0, 8)
    print(counts)
