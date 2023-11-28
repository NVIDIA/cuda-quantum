# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os

import pytest
import numpy as np

import cudaq
from cudaq import spin 

def test_parameterized_op1():
    custom_ry = cudaq.register_operation(lambda param: np.array([[
        np.cos(param / 2), -np.sin(param / 2)
    ], [np.sin(param / 2), np.cos(param / 2)]]))

    @cudaq.kernel(jit=True)
    def ansatz(theta: float):
        q = cudaq.qvector(2)
        x(q[0])
        custom_ry(theta, q[1])
        x.ctrl(q[1], q[0])

    print(ansatz)
    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

    result = cudaq.observe(ansatz, hamiltonian, .59)
    assert np.isclose(result.expectation(), -1.74, atol=1e-2)

def test_parameterized_op2():
    custom_rx = cudaq.register_operation(lambda param: np.array([[
        np.cos(param / 2), -1j*np.sin(param / 2)
    ], [-1j*np.sin(param / 2), np.cos(param / 2)]]))

    @cudaq.kernel(jit=True, verbose=True)
    def kernel(theta: float):
        q = cudaq.qubit()
        custom_rx(theta, q)

    print(kernel)

    counts = cudaq.sample(kernel, np.pi)
    assert '1' in counts and len(counts) == 1

def test_parameterized_op3_givens():
    angle = 0.2
    custom_givens = cudaq.register_operation(lambda param: np.array([
        [1, 0, 0, 0],
        [0, np.cos(param), -np.sin(param), 0], 
        [0, np.sin(param), np.cos(param), 0],
        [0, 0, 0, 1]
        ]))
    c = np.cos(angle)
    s = np.sin(angle)
    @cudaq.kernel(jit=True)
    def kernel(angle:float, flag:bool):
        q = cudaq.qvector(2)
        if not flag: 
            x(q[0])
        else:
            x(q[1])
        custom_givens(angle, q[0], q[1])

    print(kernel)
    ss_01 = cudaq.get_state(kernel, angle, False)
    assert np.isclose(ss_01[1], -s, 1e-3)
    assert np.isclose(ss_01[2], c, 1e-3)

    ss_10 = cudaq.get_state(kernel, angle, True)
    assert np.isclose(ss_10[1], c, 1e-3)
    assert np.isclose(ss_10[2], s, 1e-3)


def test_parameterized_op4():
    # Can define functions in scope
    def generator(param:float):
        return np.array([[
        np.cos(param / 2), -np.sin(param / 2)
    ], [np.sin(param / 2), np.cos(param / 2)]])

    custom_ry = cudaq.register_operation(generator)

    @cudaq.kernel(jit=True)
    def ansatz(theta: float):
        q = cudaq.qvector(2)
        x(q[0])
        custom_ry(theta, q[1])
        x.ctrl(q[1], q[0])

    print(ansatz)
    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

    result = cudaq.observe(ansatz, hamiltonian, .59)
    assert np.isclose(result.expectation(), -1.74, atol=1e-2)


 # Can define functions in scope
def generator(param:float):
    return np.array([[
    np.cos(param / 2), -np.sin(param / 2)
], [np.sin(param / 2), np.cos(param / 2)]])

def test_parameterized_op5():
    # Can define them outside of the function
    custom_ry = cudaq.register_operation(generator)

    @cudaq.kernel(jit=True)
    def ansatz(theta: float):
        q = cudaq.qvector(2)
        x(q[0])
        custom_ry(theta, q[1])
        x.ctrl(q[1], q[0])

    print(ansatz)
    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

    result = cudaq.observe(ansatz, hamiltonian, .59)
    assert np.isclose(result.expectation(), -1.74, atol=1e-2)
