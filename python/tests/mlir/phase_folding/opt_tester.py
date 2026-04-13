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


def test_simple():

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        p = cudaq.qubit()
        r = cudaq.qubit()
        h(q)
        h(p)
        h(r)
        rz(1.0, p)
        rz(2.0, r)
        cx(p, q)
        rz(3.0, q)
        cx(p, r)
        cx(q, p)
        h(r)
        cx(p, r)
        cx(q, p)
        rz(4.0, p)
        h(q)
        h(p)

    # First run without phase folding
    cudaq.set_target('qpp-cpu')
    cudaq.set_random_seed(20)
    state1 = cudaq.get_state(kernel)
    counts1 = cudaq.estimate_resources(kernel)
    assert counts1.count('rz') == 4

    # Now run with phase folding (bypassing minimum block length and Rz weight thresholds)
    cudaq.set_target('phase-folding-bench-mins')
    cudaq.set_random_seed(20)
    state2 = cudaq.get_state(kernel)
    counts2 = cudaq.estimate_resources(kernel)
    # Make sure optimization is actually performed
    assert counts2.count('rz') == 3

    assert_states_match(state1, state2)


def test_subkernel():

    @cudaq.kernel
    def subkernel(q: cudaq.qubit, p: cudaq.qubit):
        rz(1.0, p)
        cx(p, q)
        rz(3.0, q)

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        p = cudaq.qubit()
        r = cudaq.qubit()
        h(q)
        h(p)
        h(r)
        rz(2.0, r)
        subkernel(q, p)
        cx(p, q)
        cx(p, r)
        cx(q, p)
        h(r)
        cx(p, r)
        cx(q, p)
        rz(4.0, p)
        h(q)
        h(p)

    # Without phase folding
    cudaq.set_target('qpp-cpu')
    cudaq.set_random_seed(30)
    state1 = cudaq.get_state(kernel)
    counts1 = cudaq.estimate_resources(kernel)
    assert counts1.count('rz') == 4

    # With phase folding
    cudaq.set_target('phase-folding-bench-mins')
    cudaq.set_random_seed(30)
    state2 = cudaq.get_state(kernel)
    counts2 = cudaq.estimate_resources(kernel)
    assert counts2.count('rz') == 3

    assert_states_match(state1, state2)


def test_classical1():

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        p = cudaq.qubit()
        r = cudaq.qubit()
        rz(1.0, p)
        cx(q, p)
        rz(1.0, q)
        h(r)
        f = 2.0
        if mz(r):
            f += 3.0
        cx(q, p)
        rz(f, p)

    # Without phase folding
    cudaq.set_target('qpp-cpu')
    cudaq.set_random_seed(40)
    state1 = cudaq.get_state(kernel)
    counts1 = cudaq.estimate_resources(kernel)
    assert counts1.count('rz') == 3

    # With phase folding
    cudaq.set_target('phase-folding-bench-mins')
    cudaq.set_random_seed(40)
    state2 = cudaq.get_state(kernel)
    counts2 = cudaq.estimate_resources(kernel)
    assert counts2.count('rz') == 2

    assert_states_match(state1, state2)


# TODO: This test doesn't make sense for the simulator target
# where we don't want to do loop unrolling, so it is not
# currently run. For targets which do loop unrolling, it
# should be re-enabled.
@pytest.mark.skip(
    reason="Requires loop unrolling, not supported on simulator target")
def test_classical2():

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        p = cudaq.qubit()
        fs = [2.0, 2.0, 2.0, 2.0, 2.0]
        for i in range(5):
            rz(fs[i], q)
            cx(q, p)
            rz(fs[i], q)

    # Without phase folding
    cudaq.set_target('qpp-cpu')
    cudaq.set_random_seed(50)
    state1 = cudaq.get_state(kernel)
    counts1 = cudaq.estimate_resources(kernel)
    assert counts1.count('rz') == 10

    # With phase folding
    cudaq.set_target('phase-folding-bench-mins')
    cudaq.set_random_seed(50)
    state2 = cudaq.get_state(kernel)
    counts2 = cudaq.estimate_resources(kernel)
    assert counts2.count('rz') == 1

    assert_states_match(state1, state2)
