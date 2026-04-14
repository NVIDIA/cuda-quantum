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
        rz(2.97, q0[3])
        rz(0.59, q0[1])
        rz(2.72, q0[3])
        x(q0[3])
        rz(3.71, q0[1])
        rz(0.19, q0[2])
        rz(0.06, q0[4])
        swap(q0[3], q0[0])
        swap(q0[3], q0[1])
        rz(4.78, q0[1])
        swap(q0[3], q0[0])
        rz(0.76, q0[2])
        rz(5.88, q0[4])
        rz(1.91, q0[1])
        swap(q0[4], q0[3])
        x(q0[1])
        x.ctrl(q0[3], q0[1])
        swap(q0[2], q0[0])
        rz(0.68, q0[4])
        rz(2.33, q0[3])
        x.ctrl(q0[4], q0[2])
        x(q0[1])
        swap(q0[4], q0[1])
        rz(5.98, q0[2])
        if mz(q0[4]):
            break
            
            swap(q0[4], q0[1])
            x.ctrl(q0[0], q0[3])
            rz(3.58, q0[2])
            rz(2.6, q0[4])
            x.ctrl(q0[3], q0[0])
            rz(5.06, q0[0])
            x.ctrl(q0[0], q0[1])
            swap(q0[2], q0[0])
            swap(q0[0], q0[4])
            rz(1.57, q0[2])
            rz(1.16, q0[4])
            rz(1.0, q0[1])
            x(q0[2])
            x(q0[2])
            rz(0.15, q0[0])
            rz(5.0, q0[3])
            rz(5.65, q0[2])
            x(q0[4])
            rz(1.42, q0[0])
            rz(4.52, q0[0])
            rz(4.26, q0[4])
            x(q0[4])
            rz(4.07, q0[4])
            rz(5.05, q0[4])
            x.ctrl(q0[1], q0[2])
            swap(q0[2], q0[0])
            t.ctrl(q0[1], q0[0], q0[2])
            swap(q0[0], q0[1])
            rx.ctrl(0.62, q0[3], q0[1])
            r1(5.89, q0[4])
            r1(3.17, q0[2])
            x(q0[2])
            r1(0.11, q0[2])
            s.adj(q0[2])
            rz(1.34, q0[3])
            y(q0[3])
            swap(q0[3], q0[1])
            x.ctrl(q0[2], q0[0])
            swap(q0[1], q0[2])
            rz(3.18, q0[4])
            t(q0[2])
            rz(1.83, q0[0])
            ry(2.55, q0[1])
            r1(5.21, q0[1])
            rz(3.87, q0[0])
            r1(3.59, q0[0])
            s.ctrl(q0[2], q0[3], q0[4])
            
            
def test_phase_folding():
    cudaq.set_target('qpp-cpu')
    cudaq.set_random_seed(1)
    state1 = cudaq.get_state(kernel)
    
    cudaq.set_target('phase-folding-bench')
    cudaq.set_random_seed(1)
    state2 = cudaq.get_state(kernel)
    
    assert_states_match(state1, state2)

