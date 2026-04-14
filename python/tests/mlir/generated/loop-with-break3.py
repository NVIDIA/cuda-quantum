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
        x.ctrl(q0[0], q0[4])
        swap(q0[3], q0[2])
        x(q0[3])
        x.ctrl(q0[1], q0[3])
        rz(5.45, q0[1])
        swap(q0[0], q0[1])
        x.ctrl(q0[0], q0[2])
        x(q0[4])
        swap(q0[3], q0[4])
        x.ctrl(q0[3], q0[1])
        rz(3.11, q0[1])
        rz(4.89, q0[3])
        x.ctrl(q0[2], q0[3])
        x.ctrl(q0[3], q0[1])
        swap(q0[2], q0[1])
        x(q0[4])
        x.ctrl(q0[3], q0[1])
        rz(3.03, q0[0])
        x.ctrl(q0[0], q0[2])
        rz(0.13, q0[1])
        rz(5.48, q0[3])
        rz(4.79, q0[4])
        rz(2.08, q0[4])
        swap(q0[4], q0[1])
        x(q0[0])
        if mz(q0[0]):
            break
            
            swap(q0[2], q0[4])
            rz(5.63, q0[1])
            rz(5.47, q0[3])
            rz(4.28, q0[4])
            rz(1.7, q0[4])
            swap(q0[2], q0[3])
            x.ctrl(q0[2], q0[4])
            rz(6.2, q0[3])
            rz(3.87, q0[3])
            x(q0[2])
            rz(3.83, q0[2])
            rz(0.14, q0[3])
            rz(6.05, q0[0])
            rz(1.88, q0[3])
            x(q0[1])
            rz(5.3, q0[2])
            rz(0.66, q0[3])
            swap(q0[1], q0[2])
            x.ctrl(q0[1], q0[2])
            rz(4.08, q0[3])
            rz(2.02, q0[4])
            rz(2.75, q0[1])
            x.ctrl(q0[1], q0[0], q0[2])
            x.ctrl(q0[1], q0[4])
            rz(1.15, q0[4])
            t.adj(q0[2])
            rz.ctrl(2.18, q0[2], q0[1], q0[4])
            r1(3.57, q0[2])
            x.ctrl(q0[1], q0[3])
            s(q0[0])
            u3(3.51, 6.19, 4.51, q0[3])
            x.ctrl(q0[2], q0[3])
            swap(q0[2], q0[1])
            x(q0[4])
            y(q0[1])
            r1(4.68, q0[4])
            h(q0[4])
            t.adj(q0[1])
            rz(1.59, q0[1])
            swap(q0[3], q0[0])
            rz(1.08, q0[1])
            t.adj(q0[1])
            ry(0.31, q0[4])
            t(q0[1])
            
            
def test_phase_folding():
    cudaq.set_target('qpp-cpu')
    cudaq.set_random_seed(1)
    state1 = cudaq.get_state(kernel)
    
    cudaq.set_target('phase-folding-bench')
    cudaq.set_random_seed(1)
    state2 = cudaq.get_state(kernel)
    
    assert_states_match(state1, state2)

