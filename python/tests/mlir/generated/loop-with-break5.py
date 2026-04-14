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
        swap(q0[4], q0[0])
        swap(q0[1], q0[0])
        x(q0[3])
        x.ctrl(q0[2], q0[0])
        rz(2.56, q0[1])
        rz(1.0, q0[3])
        x(q0[4])
        rz(0.01, q0[1])
        x(q0[1])
        x(q0[1])
        rz(3.39, q0[1])
        x(q0[1])
        swap(q0[2], q0[0])
        rz(5.88, q0[1])
        rz(1.89, q0[2])
        x(q0[2])
        rz(5.14, q0[2])
        rz(1.16, q0[2])
        rz(0.36, q0[1])
        x(q0[2])
        x(q0[3])
        rz(5.29, q0[4])
        rz(3.92, q0[1])
        x(q0[1])
        swap(q0[2], q0[4])
        if mz(q0[4]):
            break
            
            rz(1.31, q0[0])
            x.ctrl(q0[3], q0[1])
            rz(0.58, q0[4])
            x(q0[0])
            swap(q0[2], q0[3])
            rz(0.51, q0[0])
            swap(q0[0], q0[1])
            x.ctrl(q0[4], q0[2])
            x(q0[0])
            x.ctrl(q0[2], q0[1])
            x.ctrl(q0[4], q0[1])
            rz(2.34, q0[3])
            rz(1.85, q0[3])
            rz(5.35, q0[4])
            x.ctrl(q0[2], q0[3])
            rz(2.95, q0[2])
            x(q0[4])
            x.ctrl(q0[2], q0[3])
            rz(1.71, q0[3])
            swap(q0[2], q0[0])
            swap(q0[4], q0[0])
            x.ctrl(q0[4], q0[2])
            rz(6.01, q0[2])
            rz(3.65, q0[1])
            ry(2.05, q0[4])
            t(q0[3])
            u3(2.19, 4.05, 4.64, q0[2])
            r1.ctrl(0.89, q0[3], q0[4], q0[2])
            h(q0[1])
            rz(4.34, q0[3])
            r1(1.9, q0[1])
            swap(q0[2], q0[0])
            s(q0[1])
            h(q0[1])
            h(q0[0])
            ry(0.7, q0[1])
            h(q0[3])
            s.adj(q0[4])
            r1(2.39, q0[3])
            x.ctrl(q0[0], q0[4])
            rz(5.38, q0[3])
            y(q0[4])
            x.ctrl(q0[2], q0[4])
            y(q0[2])
            
            
def test_phase_folding():
    cudaq.set_target('qpp-cpu')
    cudaq.set_random_seed(1)
    state1 = cudaq.get_state(kernel)
    
    cudaq.set_target('phase-folding-bench')
    cudaq.set_random_seed(1)
    state2 = cudaq.get_state(kernel)
    
    assert_states_match(state1, state2)

