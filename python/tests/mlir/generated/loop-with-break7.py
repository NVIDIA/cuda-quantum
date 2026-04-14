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
        x(q0[4])
        rz(0.36, q0[4])
        x(q0[3])
        rz(0.57, q0[1])
        rz(0.78, q0[4])
        rz(5.95, q0[4])
        x.ctrl(q0[0], q0[1])
        x(q0[3])
        rz(3.59, q0[0])
        swap(q0[0], q0[1])
        rz(4.47, q0[4])
        x.ctrl(q0[4], q0[3])
        x.ctrl(q0[2], q0[3])
        rz(4.39, q0[1])
        rz(1.89, q0[4])
        rz(4.58, q0[2])
        rz(0.74, q0[0])
        rz(0.95, q0[2])
        rz(6.04, q0[0])
        rz(3.6, q0[4])
        x(q0[2])
        x.ctrl(q0[0], q0[3])
        swap(q0[3], q0[0])
        rz(4.07, q0[2])
        swap(q0[2], q0[3])
        x.ctrl(q0[2], q0[3])
        if mz(q0[1]):
            break

            x.ctrl(q0[0], q0[3])
            rz(3.45, q0[3])
            swap(q0[4], q0[2])
            swap(q0[3], q0[1])
            rz(0.95, q0[1])
            x(q0[4])
            rz(0.03, q0[2])
            rz(3.83, q0[2])
            rz(4.34, q0[1])
            x.ctrl(q0[4], q0[0])
            swap(q0[4], q0[3])
            rz(0.65, q0[3])
            x(q0[0])
            x.ctrl(q0[3], q0[0])
            x(q0[4])
            rz(5.96, q0[0])
            x(q0[1])
            x(q0[2])
            x.ctrl(q0[0], q0[3])
            rz(6.24, q0[3])
            rz(1.96, q0[3])
            rz(4.65, q0[2])
            h(q0[4])
            x(q0[1])
            u3(0.92, 3.41, 0.17, q0[2])
            u3(6.15, 5.42, 4.37, q0[2])
            t(q0[4])
            h(q0[2])
            s.adj(q0[4])
            s(q0[1])
            s(q0[1])
            x(q0[0])
            t(q0[3])
            s(q0[4])
            rx(0.51, q0[2])
            y(q0[1])
            s.adj(q0[2])
            ry.ctrl(0.75, q0[0], q0[2], q0[4])
            r1(3.0, q0[1])
            h(q0[3])
            rz(5.03, q0[0])
            r1(2.52, q0[3])
            x(q0[1])
            z(q0[4])
            z(q0[4])
            rz(5.02, q0[1])
            y.ctrl(q0[1], q0[4])
            x(q0[2])
            s.adj(q0[2])


def test_phase_folding():
    cudaq.set_target('qpp-cpu')
    cudaq.set_random_seed(1)
    state1 = cudaq.get_state(kernel)

    cudaq.set_target('phase-folding-bench')
    cudaq.set_random_seed(1)
    state2 = cudaq.get_state(kernel)

    assert_states_match(state1, state2)
