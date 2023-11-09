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
from typing import Callable 

# This file is a hodgepodge of tests that I've used for 
# understanding the AST and mapping to MLIR

@pytest.fixture(autouse=True)
def do_something():
    cudaq.__clearKernelRegistries()
    yield 
    return 


def test_adjoint():
    """Test that adjoint can be called on kernels and operations."""
    @cudaq.kernel(jit=True)
    def single_adjoint_test():
        q = cudaq.qubit()
        t(q)
        t.adj(q)

    counts = cudaq.sample(single_adjoint_test)
    assert '0' in counts
    assert len(counts) == 1

    @cudaq.kernel(jit=True)
    def qvector_adjoint_test():
        q = cudaq.qvector(2)
        t(q)
        t.adj(q)

    counts = cudaq.sample(qvector_adjoint_test)
    assert '00' in counts
    assert len(counts) == 1

    @cudaq.kernel(jit=True)
    def rotation_adjoint_test():
        q = cudaq.qubit()
        rx(1.1, q)
        rx.adj(1.1, q)

        ry(1.1, q)
        ry.adj(1.1, q)

    counts = cudaq.sample(rotation_adjoint_test)
    assert '0' in counts
    assert len(counts) == 1

    @cudaq.kernel(jit=True)
    def test_kernel_adjoint(q:cudaq.qview):
        h(q[0])
        t(q[1])
        s(q[2])

    @cudaq.kernel(jit=True)
    def test_caller():
        q = cudaq.qvector(3)
        x(q[0])
        x(q[2])
        test_kernel_adjoint(q)
        cudaq.adjoint(test_kernel_adjoint, q)

    counts = cudaq.sample(test_caller)
    assert len(counts) == 1
    assert '101' in counts

def test_exp_pauli():
    h2_data = [
      3, 1, 1, 3, 0.0454063,  0,  2,  0, 0, 0, 0.17028,    0,
      0, 0, 2, 0, -0.220041,  -0, 1,  3, 3, 1, 0.0454063,  0,
      0, 0, 0, 0, -0.106477,  0,  0,  2, 0, 0, 0.17028,    0,
      0, 0, 0, 2, -0.220041,  -0, 3,  3, 1, 1, -0.0454063, -0,
      2, 2, 0, 0, 0.168336,   0,  2,  0, 2, 0, 0.1202,     0,
      0, 2, 0, 2, 0.1202,     0,  2,  0, 0, 2, 0.165607,   0,
      0, 2, 2, 0, 0.165607,   0,  0,  0, 2, 2, 0.174073,   0,
      1, 1, 3, 3, -0.0454063, -0, 15
    ]
    h = cudaq.SpinOperator(h2_data, 4)

    @cudaq.kernel(jit=True, verbose=True)
    def kernel(theta:float):
        q = cudaq.qvector(4)
        x(q[0])
        x(q[1])
        exp_pauli(theta, q, 'XXXY')

    print(kernel)
    want_exp = cudaq.observe(kernel, h, .11).expectation()
    assert np.isclose(want_exp, -1.13, atol=1e-2)


# Fail due to Issue 806 (swap lowering) and 805
# @pytest.mark.xfail
# def test_control():
#     """Test that we can control on kernel functions."""
#     @cudaq.kernel(jit=True, verbose=True)
#     def fancyCnot(a:cudaq.qubit, b:cudaq.qubit):
#         x.ctrl(a, b)

#     @cudaq.kernel(jit=True, verbose=True)
#     def toffoli():
#         q = cudaq.qvector(3)
#         ctrl = q.front()
#         # without a control, apply x to all 
#         x(ctrl, q[2])
#         cudaq.control(fancyCnot, [ctrl], q[1], q[2])

#     counts = cudaq.sample(toffoli)
#     assert len(counts) == 1
#     assert '101' in counts

#     @cudaq.kernel(jit=True, verbose=True)
#     def test():
#         q, r, s = cudaq.qubit(), cudaq.qubit(), cudaq.qubit()
#         x(q, s)
#         swap.ctrl(q, r, s)
    
#     print(test)
#     counts = cudaq.sample(test)
#     assert len(counts) == 1
#     assert '110' in counts


def test_grover():
    """Test that compute_action works in tandem with kernel composability."""
    @cudaq.kernel(jit=True)
    def reflect(qubits:cudaq.qview):
        ctrls = qubits.front(qubits.size()-1)
        last = qubits.back()

        # def compute():
        #     h(qubits)
        #     x(qubits)
        h(qubits)
        x(qubits)
        z.ctrl(ctrls, last)
        x(qubits)
        h(qubits)
        # cudaq.compute_action(compute, lambda: z.ctrl(ctrls, last))

    print(reflect)

    # Order matters, kernels must be "in-scope"
    @cudaq.kernel(jit=True)
    def oracle(q:cudaq.qview):
        z.ctrl(q[0], q[2])
        z.ctrl(q[1], q[2])

    @cudaq.kernel(jit=True)
    def grover(N:int, M:int, oracle:Callable[[cudaq.qview],None]):
        q = cudaq.qvector(N)
        h(q)
        for i in range(M):
            oracle(q)
            reflect(q)
        mz(q)

    print(grover)

    print(oracle)

    counts = cudaq.sample(grover, 3, 1, oracle)
    assert len(counts) == 2
    assert '101' in counts
    assert '011' in counts


def test_grover_compute_action():
    """Test that compute_action works in tandem with kernel composability."""
    @cudaq.kernel(jit=True)
    def reflect(qubits:cudaq.qview):
        ctrls = qubits.front(qubits.size()-1)
        last = qubits.back()

        def compute():
            h(qubits)
            x(qubits)
        # can also use 
        # compute = lambda : (h(qubits), x(qubits))

        cudaq.compute_action(compute, lambda: z.ctrl(ctrls, last))

    print(reflect)

    # Order matters, kernels must be "in-scope"
    @cudaq.kernel(jit=True)
    def oracle(q:cudaq.qview):
        z.ctrl(q[0], q[2])
        z.ctrl(q[1], q[2])

    @cudaq.kernel(jit=True)
    def grover(N:int, M:int, oracle:Callable[[cudaq.qview],None]):
        q = cudaq.qvector(N)
        h(q)
        for i in range(M):
            oracle(q)
            reflect(q)
        mz(q)

    # print(grover)

    counts = cudaq.sample(grover, 3, 1, oracle)
    assert len(counts) == 2
    assert '101' in counts
    assert '011' in counts


def test_dynamic_circuit():
    """Test that we correctly sample circuits with 
       mid-circuit measurements and conditionals."""
    @cudaq.kernel(jit=True, verbose=True)
    def simple():
        q = cudaq.qvector(2)
        h(q[0])
        i = mz(q[0])
        if i:
            x(q[1])
        mz(q)

    print(simple)

    counts = cudaq.sample(simple)
    counts.dump()
    c0 = counts.get_register_counts('i')
    assert '0' in c0 and '1' in c0
    assert '00' in counts and '11' in counts
    