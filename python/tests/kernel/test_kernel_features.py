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
from typing import Callable 

import cudaq

@pytest.fixture(autouse=True)
def do_something():
    if os.getenv("CUDAQ_PYTEST_EAGER_MODE") == 'OFF':
        cudaq.enable_jit()
    yield

    if cudaq.is_jit_enabled(): cudaq.__clearKernelRegistries()
    cudaq.disable_jit()

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
    def test_kernel_adjoint(q:cudaq.qview):
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
    def fancyCnot(a:cudaq.qubit, b:cudaq.qubit):
        x.ctrl(a, b)

    @cudaq.kernel
    def toffoli():
        q = cudaq.qvector(3)
        ctrl = q.front()
        # without a control, apply x to all
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

    @cudaq.kernel#(verbose=True)
    def reflect(qubits:cudaq.qview):
        ctrls = qubits.front(qubits.size() - 1)
        last = qubits.back()
        cudaq.compute_action(lambda: (h(qubits), x(qubits)),
                             lambda: z.ctrl(ctrls, last))
        
    # FIXME This currently has to be defined before the 
    # kernel that uses it as input
    @cudaq.kernel
    def oracle(q:cudaq.qview):
        z.ctrl(q[0], q[2])
        z.ctrl(q[1], q[2])

    print(reflect)
    @cudaq.kernel
    def grover(N:int, M:int, oracle:Callable[[cudaq.qview], None]):
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



def test_2grover_compute_action():
    """Test that compute_action works in tandem with kernel composability."""

    @cudaq.kernel
    def reflect2(qubits: cudaq.qview):
        ctrls = qubits.front(qubits.size() - 1)
        last = qubits.back()

        def compute():
            h(qubits)
            x(qubits)

        # can also use
        # compute = lambda : (h(qubits), x(qubits))

        cudaq.compute_action(compute, lambda: z.ctrl(ctrls, last))

    print(reflect2)

    # Order matters, kernels must be "in-scope"
    @cudaq.kernel
    def oracle2(q: cudaq.qview):
        z.ctrl(q[0], q[2])
        z.ctrl(q[1], q[2])

    @cudaq.kernel
    def grover(N: int, M: int, oracle: Callable[[cudaq.qview], None]):
        q = cudaq.qvector(N)
        h(q)
        for i in range(M):
            oracle(q)
            reflect2(q)
        mz(q)

    # print(grover)

    counts = cudaq.sample(grover, 3, 1, oracle2)
    assert len(counts) == 2
    assert '101' in counts
    assert '011' in counts


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

    @cudaq.kernel
    def kernel(theta: float):
        q = cudaq.qvector(4)
        x(q[0])
        x(q[1])
        exp_pauli(theta, q, 'XXXY')

    print(kernel)
    want_exp = cudaq.observe(kernel, h, .11).expectation()
    assert np.isclose(want_exp, -1.13, atol=1e-2)


def test_dynamic_circuit():
    """Test that we correctly sample circuits with 
       mid-circuit measurements and conditionals."""

    @cudaq.kernel
    def simple():
        q = cudaq.qvector(2)
        h(q[0])
        i = mz(q[0], register_name="c0")
        if i:
            x(q[1])
        mz(q)

    counts = cudaq.sample(simple)
    counts.dump()
    c0 = counts.get_register_counts('c0')
    assert '0' in c0 and '1' in c0
    assert '00' in counts and '11' in counts

    @cudaq.kernel
    def simple():
        q = cudaq.qvector(2)
        h(q[0])
        i = mz(q[0])
        if i:
            x(q[1])
        mz(q)

    counts = cudaq.sample(simple)
    counts.dump()
    c0 = counts.get_register_counts('i')
    assert '0' in c0 and '1' in c0
    assert '00' in counts and '11' in counts


def test_teleport():

    @cudaq.kernel
    def teleport():
        q = cudaq.qvector(3)
        x(q[0])
        h(q[1])

        x.ctrl(q[1], q[2])

        x.ctrl(q[0], q[1])
        h(q[0])

        b0 = mz(q[0])
        b1 = mz(q[1])

        if b1:
            x(q[2])

        if b0:
            z(q[2])

        mz(q[2])

    counts = cudaq.sample(teleport, shots_count=100)
    counts.dump()
    # Note this is testing that we can provide
    # the register name automatically
    b0 = counts.get_register_counts('b0')
    assert '0' in b0 and '1' in b0
