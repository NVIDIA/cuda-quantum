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
    print(kernel.__str__(canonicalize=False))

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

    # Division 
    test = value_0 / 8
    print(kernel.__str__(canonicalize=False))

def test_QuakeValueSize():
    kernel, thetas, runtimeSize = cudaq.make_kernel(list, int)
    q = kernel.qalloc(5)
    print(kernel)
    assert q.size() == 5 

    otherq = kernel.qalloc(runtimeSize)
    s = otherq.size()
    quake = kernel.__str__(canonicalize=False)
    assert 'quake.veq_size' in quake

    ts = thetas.size()
    quake = kernel.__str__(canonicalize=False)
    assert 'cc.stdvec_size' in quake 

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