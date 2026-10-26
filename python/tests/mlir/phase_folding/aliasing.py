# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../../ pytest -rP  %s

import cudaq
import pytest


def is_close(x, y, tolerance):
    return abs(x - y) < tolerance


def assert_states_match(state1, state2):
    assert state1.num_qubits() == state2.num_qubits()
    overlap = state1.overlap(state2)
    assert is_close(overlap.real, 1.0, 1e-6)
    assert is_close(overlap.imag, 0.0, 1e-6)


def test_aliasing_slice():

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(2)
        cx(q[0], q[1])
        rz(1.0, q[1])
        r = q[1:2][0]
        h(r)
        rz(2.0, q[1])

    # Without phase folding
    cudaq.set_target('qpp-cpu')
    state1 = cudaq.get_state(kernel)
    counts1 = cudaq.estimate_resources(kernel)
    assert counts1.count('rz') == 2

    # With phase folding
    cudaq.set_target('phase-folding-bench-mins')
    state2 = cudaq.get_state(kernel)
    counts2 = cudaq.estimate_resources(kernel)
    assert counts2.count('rz') == 2

    assert_states_match(state1, state2)


def test_aliasing_loop_ref():

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(5)
        cx(q[0], q[1])
        rz(1.0, q[1])
        for i in range(5):
            r = q[i]
            h(r)
        rz(1.0, q[1])

    # Without phase folding
    cudaq.set_target('qpp-cpu')
    state1 = cudaq.get_state(kernel)
    counts1 = cudaq.estimate_resources(kernel)
    assert counts1.count('rz') == 2

    # With phase folding
    cudaq.set_target('phase-folding-bench-mins')
    state2 = cudaq.get_state(kernel)
    counts2 = cudaq.estimate_resources(kernel)
    assert counts2.count('rz') == 2

    assert_states_match(state1, state2)


def test_aliasing_subkernel():

    @cudaq.kernel
    def subkernel(r: cudaq.qubit):
        h(r)

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(2)
        cx(q[0], q[1])
        rz(1.0, q[1])
        subkernel(q.back())
        rz(1.0, q[1])

    # Without phase folding
    cudaq.set_target('qpp-cpu')
    state1 = cudaq.get_state(kernel)
    counts1 = cudaq.estimate_resources(kernel)
    assert counts1.count('rz') == 2

    # With phase folding
    cudaq.set_target('phase-folding-bench-mins')
    state2 = cudaq.get_state(kernel)
    counts2 = cudaq.estimate_resources(kernel)
    assert counts2.count('rz') == 2

    assert_states_match(state1, state2)


@pytest.mark.parametrize("seed", [1, 2])
def test_aliasing_branch(seed):

    @cudaq.kernel
    def subkernel(r: cudaq.qubit):
        h(r)

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(3)
        cx(q[0], q[2])
        rz(1.0, q[0])
        rz(1.0, q[2])
        h(q[1])
        if mz(q[1]):
            subkernel(q.front())
        else:
            subkernel(q.back())
        rz(1.0, q[0])
        rz(1.0, q[2])

    # Need to make sure each kernel execution takes the same
    # execution path (side of the branch)

    # Without phase folding
    cudaq.set_random_seed(seed)
    cudaq.set_target('qpp-cpu')
    state1 = cudaq.get_state(kernel)
    counts1 = cudaq.estimate_resources(kernel)
    assert counts1.count('rz') == 4

    # With phase folding
    cudaq.set_random_seed(seed)
    cudaq.set_target('phase-folding-bench-mins')
    state2 = cudaq.get_state(kernel)
    counts2 = cudaq.estimate_resources(kernel)
    assert counts2.count('rz') == 4

    assert_states_match(state1, state2)
