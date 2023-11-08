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
from cudaq import spin, x, z, h, ry, t, swap, r1, mz, rx, s


def test_adjoint():
    """Test that adjoint can be called on kernels and operations."""
    @cudaq.kernel
    def single_adjoint_test():
        q = cudaq.qubit()
        t(q)
        t.adj(q)

    counts = cudaq.sample(single_adjoint_test)
    assert '0' in counts
    assert len(counts) == 1

    @cudaq.kernel
    def qvector_adjoint_test():
        q = cudaq.qvector(2)
        t(q)
        t.adj(q)

    counts = cudaq.sample(qvector_adjoint_test)
    assert '00' in counts
    assert len(counts) == 1

    @cudaq.kernel
    def rotation_adjoint_test():
        q = cudaq.qubit()
        rx(1.1, q)
        rx.adj(1.1, q)

        ry(1.1, q)
        ry.adj(1.1, q)

    counts = cudaq.sample(rotation_adjoint_test)
    assert '0' in counts
    assert len(counts) == 1

    @cudaq.kernel
    def test_kernel_adjoint(q):
        h(q[0])
        t(q[1])
        s(q[2])

    @cudaq.kernel
    def test_caller():
        q = cudaq.qvector(3)
        x(q[0])
        x(q[2])
        test_kernel_adjoint(q)
        cudaq.adjoint(test_kernel_adjoint, q)

    counts = cudaq.sample(test_caller)
    assert len(counts) == 1
    assert '101' in counts


def test_control():
    """Test that we can control on kernel functions."""
    @cudaq.kernel
    def fancyCnot(a, b):
        x.ctrl(a, b)

    @cudaq.kernel
    def toffoli():
        q = cudaq.qvector(3)
        ctrl = q.front()
        x(ctrl, q[2])
        cudaq.control(fancyCnot, [ctrl], q[1], q[2])

    counts = cudaq.sample(toffoli)
    assert len(counts) == 1
    assert '101' in counts

    @cudaq.kernel
    def test():
        q, r, s = cudaq.qubit(), cudaq.qubit(), cudaq.qubit()
        x(q, s)
        swap.ctrl(q, r, s)
    counts = cudaq.sample(test)
    assert len(counts) == 1
    assert '110' in counts


def test_grover():
    """Test that compute_action works in tandem with kernel composability."""
    @cudaq.kernel
    def reflect(qubits):
        ctrls = qubits.front(qubits.size()-1)
        last = qubits.back()

        def compute():
            h(qubits)
            x(qubits)

        cudaq.compute_action(compute, lambda: z.ctrl(ctrls, last))

    @cudaq.kernel
    def grover(N, M, oracle):
        q = cudaq.qvector(N)
        h(q)
        for i in range(M):
            oracle(q)
            reflect(q)
        mz(q)

    @cudaq.kernel
    def oracle(q):
        z.ctrl(q[0], q[2])
        z.ctrl(q[1], q[2])

    counts = cudaq.sample(grover, 3, 1, oracle)
    assert len(counts) == 2
    assert '101' in counts
    assert '011' in counts


def test_dynamic_circuit():
    """Test that we correctly sample circuits with 
       mid-circuit measurements and conditionals."""
    @cudaq.kernel
    def simple():
        q = cudaq.qvector(2)
        h(q[0])
        i = mz(q[0], "c0")
        if i:
            x(q[1])
        mz(q)

    counts = cudaq.sample(simple)
    counts.dump()
