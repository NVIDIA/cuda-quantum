# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../../ pytest -rP  %s

import cudaq


def assert_states_match(state1, state2, tolerance=1e-5):
    assert state1.num_qubits() == state2.num_qubits()
    overlap = state1.overlap(state2)
    print(f"Real: {overlap.real}, imag: {overlap.imag}")
    assert abs(overlap.real - 1.0) < tolerance
    assert abs(overlap.imag) < tolerance


@cudaq.kernel
def kernel():
    q0 = cudaq.qvector(5)
    for i in range(10):
        x(q0[1])
        x.ctrl(q0[3], q0[2])
        x(q0[1])
        x.ctrl(q0[0], q0[4])
        rz(4.4, q0[4])
        x.ctrl(q0[0], q0[4])
        x.ctrl(q0[3], q0[0])
        swap(q0[3], q0[0])
        swap(q0[2], q0[0])
        x.ctrl(q0[3], q0[1])
        rz(3.23, q0[1])
        swap(q0[4], q0[0])
        x(q0[3])
        rz(0.75, q0[4])
        swap(q0[1], q0[0])
        x(q0[1])
        rz(0.87, q0[1])
        rz(0.08, q0[4])
        x(q0[2])
        swap(q0[4], q0[2])
        rz(2.45, q0[4])
        rz(0.34, q0[3])
        rz(0.6, q0[3])
        rz(5.61, q0[0])
        swap(q0[3], q0[4])
        x.ctrl(q0[2], q0[3])
        x.ctrl(q0[2], q0[0])
        if mz(q0[2]):
            break

            rz(5.04, q0[4])
            rz(2.84, q0[4])
            x(q0[3])
            x.ctrl(q0[2], q0[3])
            x.ctrl(q0[1], q0[3])
            x.ctrl(q0[2], q0[3])
            rz(6.1, q0[1])
            x.ctrl(q0[2], q0[4])
            rz(6.13, q0[3])
            x(q0[2])
            x.ctrl(q0[2], q0[0])
            rz(0.7, q0[4])
            x.ctrl(q0[0], q0[3])
            x(q0[2])
            x.ctrl(q0[3], q0[4])
            rz(0.63, q0[1])
            rz(4.28, q0[1])
            rz(0.91, q0[2])
            x(q0[0])
            rz(2.91, q0[3])
            x.ctrl(q0[1], q0[4])
            swap(q0[1], q0[2])
            swap(q0[3], q0[4])
            swap(q0[1], q0[4])
            x.ctrl(q0[4], q0[3])
            x.ctrl(q0[0], q0[4])
            x(q0[2])
            rz(0.53, q0[0])
            t(q0[2])
            r1(0.2, q0[3])
            x(q0[1])
            rz(0.2, q0[2])
            s(q0[3])
            x.ctrl(q0[4], q0[1])
            ry(3.27, q0[0])
            x(q0[1])
            u3(1.15, 4.64, 3.19, q0[1])
            r1(5.94, q0[1])
            z.ctrl(q0[0], q0[3])
            r1(4.78, q0[1])
            r1(2.77, q0[4])
            r1(3.46, q0[4])
            z(q0[1])
            t.adj(q0[1])
            h(q0[1])
            r1(0.38, q0[0])
            x.ctrl(q0[0], q0[3])
            y(q0[4])
            x.ctrl(q0[1], q0[4])
            r1(0.58, q0[1])
            x.ctrl(q0[0], q0[4])
            y(q0[1])


def test_phase_folding():
    cudaq.set_target('qpp-cpu')
    cudaq.set_random_seed(1)
    state1 = cudaq.get_state(kernel)

    cudaq.set_target('phase-folding-bench')
    cudaq.set_random_seed(1)
    state2 = cudaq.get_state(kernel)

    assert_states_match(state1, state2)
