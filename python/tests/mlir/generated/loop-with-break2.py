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
        rz(4.62, q0[1])
        x(q0[4])
        rz(3.65, q0[0])
        rz(4.01, q0[3])
        swap(q0[4], q0[2])
        x.ctrl(q0[0], q0[2])
        x.ctrl(q0[4], q0[2])
        rz(1.03, q0[4])
        rz(0.15, q0[1])
        rz(3.21, q0[1])
        rz(4.24, q0[4])
        rz(5.01, q0[3])
        swap(q0[2], q0[4])
        rz(1.01, q0[3])
        swap(q0[3], q0[1])
        rz(3.15, q0[3])
        x.ctrl(q0[4], q0[3])
        rz(4.55, q0[4])
        rz(5.91, q0[1])
        swap(q0[4], q0[2])
        x.ctrl(q0[4], q0[2])
        if mz(q0[2]):
            break

            swap(q0[1], q0[0])
            rz(1.72, q0[0])
            rz(4.74, q0[0])
            rz(1.54, q0[2])
            rz(2.66, q0[0])
            x(q0[2])
            rz(4.23, q0[1])
            rz(0.16, q0[0])
            x(q0[2])
            rz(4.62, q0[1])
            x(q0[4])
            rz(0.95, q0[1])
            rz(5.9, q0[2])
            swap(q0[2], q0[4])
            rz(2.82, q0[2])
            swap(q0[2], q0[3])
            swap(q0[3], q0[1])
            rz(5.27, q0[2])
            rz(3.26, q0[1])
            x.ctrl(q0[1], q0[2])
            x.ctrl(q0[4], q0[2])
            swap(q0[1], q0[0])
            rz(1.01, q0[1])
            rz(3.19, q0[1])
            swap(q0[1], q0[4])
            x.ctrl(q0[3], q0[2])
            r1(2.42, q0[2])
            h(q0[0])
            u3(1.51, 0.63, 1.14, q0[0])
            s(q0[0])
            t.adj(q0[4])
            s(q0[3])
            u3(3.65, 0.32, 2.63, q0[0])
            u3(1.14, 0.59, 5.04, q0[4])
            rz.ctrl(0.12, q0[3], q0[2], q0[0])
            r1.ctrl(5.1, q0[1], q0[0])
            x.ctrl(q0[4], q0[3])
            rx(0.15, q0[1])
            t.ctrl(q0[1], q0[0])
            rz(0.88, q0[3])
            rz(0.77, q0[3])
            rz(6.03, q0[3])
            y(q0[0])
            x.ctrl(q0[2], q0[3])
            rz.ctrl(5.44, q0[4], q0[2], q0[3])
            r1(2.48, q0[3])
            h(q0[3])
            t(q0[4])
            x(q0[4])
            h(q0[4])
            z(q0[3])
            z(q0[2])


def test_phase_folding():
    cudaq.set_target('qpp-cpu')
    cudaq.set_random_seed(1)
    state1 = cudaq.get_state(kernel)

    cudaq.set_target('phase-folding-bench')
    cudaq.set_random_seed(1)
    state2 = cudaq.get_state(kernel)

    assert_states_match(state1, state2)
