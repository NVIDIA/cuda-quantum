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
        rz(1.29, q0[0])
        swap(q0[1], q0[0])
        x(q0[1])
        swap(q0[2], q0[0])
        rz(3.79, q0[1])
        rz(5.2, q0[2])
        rz(4.31, q0[1])
        x(q0[1])
        x.ctrl(q0[1], q0[0])
        rz(5.91, q0[1])
        rz(4.85, q0[2])
        rz(2.83, q0[4])
        rz(2.04, q0[4])
        swap(q0[3], q0[1])
        rz(1.89, q0[3])
        x(q0[0])
        rz(5.94, q0[4])
        x(q0[0])
        rz(5.95, q0[1])
        rz(2.8, q0[3])
        x(q0[4])
        rz(1.68, q0[0])
        rz(4.89, q0[1])
        rz(2.06, q0[2])
        rz(4.48, q0[4])
        rz(1.52, q0[2])
        if mz(q0[2]):
            break
            
            swap(q0[2], q0[3])
            x(q0[3])
            swap(q0[1], q0[3])
            x.ctrl(q0[4], q0[3])
            x.ctrl(q0[4], q0[0])
            swap(q0[3], q0[0])
            rz(0.96, q0[0])
            swap(q0[3], q0[1])
            rz(4.87, q0[3])
            x.ctrl(q0[4], q0[2])
            rz(5.71, q0[4])
            rz(2.03, q0[1])
            rz(4.04, q0[1])
            swap(q0[3], q0[0])
            swap(q0[3], q0[1])
            swap(q0[0], q0[4])
            x(q0[3])
            rz(2.52, q0[4])
            swap(q0[1], q0[0])
            x.ctrl(q0[3], q0[0])
            x.ctrl(q0[3], q0[4])
            rz(1.31, q0[4])
            x(q0[0])
            rz(2.57, q0[4])
            rz(0.44, q0[1])
            rz(5.18, q0[0])
            rz(2.87, q0[2])
            x.ctrl(q0[4], q0[2])
            x.ctrl(q0[2], q0[4])
            rz(2.19, q0[2])
            ry(3.69, q0[3])
            x(q0[4])
            r1(6.06, q0[1])
            r1(3.72, q0[4])
            rz(4.65, q0[0])
            rz.ctrl(0.6, q0[3], q0[2])
            x.ctrl(q0[0], q0[1])
            r1(5.75, q0[1])
            x(q0[3])
            s.adj(q0[2])
            u3(4.8, 6.0, 2.05, q0[4])
            x.ctrl(q0[0], q0[4])
            x.ctrl(q0[2], q0[4])
            r1(3.78, q0[2])
            h(q0[1])
            y(q0[4])
            h(q0[0])
            swap(q0[4], q0[2])
            x(q0[3])
            rz.ctrl(6.2, q0[4], q0[3], q0[2], q0[0])
            h(q0[2])
            r1(2.44, q0[2])
            y(q0[3])
            s(q0[4])
            
            
def test_phase_folding():
    cudaq.set_target('qpp-cpu')
    cudaq.set_random_seed(1)
    state1 = cudaq.get_state(kernel)
    
    cudaq.set_target('phase-folding-bench')
    cudaq.set_random_seed(1)
    state2 = cudaq.get_state(kernel)
    
    assert_states_match(state1, state2)

