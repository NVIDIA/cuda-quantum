# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest

import cudaq


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