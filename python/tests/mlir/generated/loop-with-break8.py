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
        rz(0.54, q0[0])
        rz(1.32, q0[4])
        x.ctrl(q0[4], q0[3])
        rz(5.64, q0[1])
        rz(1.47, q0[3])
        rz(3.27, q0[2])
        rz(4.56, q0[3])
        x(q0[0])
        rz(0.68, q0[3])
        rz(4.34, q0[1])
        rz(5.66, q0[4])
        x(q0[0])
        x(q0[1])
        rz(2.87, q0[4])
        x.ctrl(q0[2], q0[4])
        x.ctrl(q0[0], q0[1])
        x(q0[2])
        swap(q0[4], q0[1])
        rz(3.32, q0[2])
        x(q0[4])
        x.ctrl(q0[1], q0[2])
        x(q0[1])
        rz(1.6, q0[3])
        rz(1.1, q0[3])
        rz(5.86, q0[1])
        x(q0[1])
        if mz(q0[2]):
            break
            
            rz(0.87, q0[1])
            x.ctrl(q0[1], q0[0])
            rz(0.21, q0[0])
            swap(q0[4], q0[2])
            rz(3.07, q0[4])
            rz(0.4, q0[4])
            rz(2.14, q0[1])
            x(q0[2])
            rz(0.02, q0[2])
            x(q0[4])
            rz(1.08, q0[4])
            rz(2.77, q0[3])
            rz(1.41, q0[4])
            swap(q0[3], q0[0])
            rz(4.07, q0[3])
            x.ctrl(q0[4], q0[0])
            swap(q0[3], q0[1])
            rz(2.8, q0[0])
            rz(4.45, q0[0])
            x(q0[2])
            x.ctrl(q0[1], q0[4])
            x(q0[3])
            x.ctrl(q0[3], q0[0])
            swap(q0[2], q0[3])
            rz(0.5, q0[2])
            rz(5.07, q0[2])
            x(q0[2])
            swap(q0[2], q0[0])
            rz(2.38, q0[2])
            t(q0[1])
            r1(5.88, q0[4])
            rz(4.23, q0[4])
            s.adj(q0[3])
            t.adj(q0[1])
            ry(4.9, q0[2])
            t(q0[2])
            u3(1.74, 5.58, 2.59, q0[2])
            rz.ctrl(5.11, q0[2], q0[1])
            u3(2.31, 4.91, 4.99, q0[1])
            r1(4.11, q0[0])
            r1(4.3, q0[0])
            x(q0[0])
            rx.ctrl(2.01, q0[0], q0[2])
            x.ctrl(q0[2], q0[0])
            rz(1.4, q0[0])
            rz(2.73, q0[2])
            h(q0[2])
            r1(2.71, q0[1])
            x(q0[0])
            h(q0[2])
            s.adj(q0[0])
            x.ctrl(q0[1], q0[4])
            h(q0[4])
            rz(5.54, q0[2])
            u3(0.7, 4.51, 0.52, q0[3])
            z.ctrl(q0[4], q0[1])
            h(q0[1])
            z.ctrl(q0[4], q0[1], q0[3])
            
            
def test_phase_folding():
    cudaq.set_target('qpp-cpu')
    cudaq.set_random_seed(1)
    state1 = cudaq.get_state(kernel)
    
    cudaq.set_target('phase-folding-bench')
    cudaq.set_random_seed(1)
    state2 = cudaq.get_state(kernel)
    
    assert_states_match(state1, state2)

