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
        rz(2.13, q0[0])
        rz(0.51, q0[4])
        x.ctrl(q0[4], q0[0])
        rz(5.98, q0[3])
        rz(1.5, q0[1])
        rz(5.48, q0[4])
        x(q0[3])
        swap(q0[2], q0[1])
        swap(q0[3], q0[0])
        swap(q0[3], q0[2])
        rz(1.25, q0[0])
        swap(q0[0], q0[3])
        x(q0[1])
        rz(6.06, q0[0])
        rz(1.24, q0[4])
        x(q0[0])
        rz(3.15, q0[0])
        x.ctrl(q0[4], q0[1])
        rz(4.01, q0[1])
        rz(0.91, q0[4])
        swap(q0[0], q0[4])
        rz(5.35, q0[4])
        swap(q0[4], q0[0])
        rz(5.97, q0[3])
        if mz(q0[1]):
            break
            
            rz(2.3, q0[4])
            rz(6.2, q0[1])
            x(q0[4])
            rz(0.7, q0[3])
            rz(2.04, q0[2])
            rz(0.65, q0[2])
            x(q0[4])
            x.ctrl(q0[0], q0[1])
            rz(0.6, q0[1])
            x.ctrl(q0[2], q0[4])
            rz(1.07, q0[2])
            x(q0[0])
            swap(q0[2], q0[1])
            x.ctrl(q0[1], q0[0])
            rz(4.94, q0[1])
            rz(6.07, q0[0])
            rz(3.02, q0[1])
            rz(4.7, q0[0])
            rz(0.41, q0[0])
            rz(4.8, q0[4])
            rz(4.17, q0[1])
            rz(5.53, q0[4])
            rz(5.8, q0[0])
            rz(1.09, q0[1])
            x(q0[1])
            s(q0[1])
            rx(0.52, q0[4])
            swap(q0[3], q0[2])
            swap(q0[0], q0[4])
            r1(3.29, q0[4])
            swap(q0[0], q0[2])
            x(q0[1])
            ry(4.26, q0[3])
            x(q0[2])
            s.ctrl(q0[1], q0[0], q0[3])
            x.ctrl(q0[0], q0[3], q0[2], q0[4])
            x.ctrl(q0[1], q0[4])
            x.ctrl(q0[4], q0[1], q0[0])
            x(q0[2])
            rx.ctrl(4.22, q0[1], q0[0], q0[4], q0[3])
            rx(0.69, q0[1])
            y(q0[0])
            x(q0[3])
            u3(2.32, 3.99, 1.48, q0[1])
            t.adj(q0[4])
            
            
def test_phase_folding():
    cudaq.set_target('qpp-cpu')
    cudaq.set_random_seed(1)
    state1 = cudaq.get_state(kernel)
    
    cudaq.set_target('phase-folding-bench')
    cudaq.set_random_seed(1)
    state2 = cudaq.get_state(kernel)
    
    assert_states_match(state1, state2)

