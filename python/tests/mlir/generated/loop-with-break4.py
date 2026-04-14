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
        x.ctrl(q0[1], q0[0])
        rz(5.77, q0[4])
        swap(q0[1], q0[2])
        rz(5.2, q0[1])
        rz(5.21, q0[0])
        swap(q0[1], q0[4])
        rz(0.54, q0[2])
        swap(q0[4], q0[1])
        rz(1.76, q0[3])
        swap(q0[4], q0[2])
        rz(3.6, q0[2])
        swap(q0[4], q0[1])
        rz(1.81, q0[4])
        rz(1.92, q0[1])
        x(q0[0])
        rz(3.26, q0[2])
        swap(q0[1], q0[4])
        rz(3.99, q0[1])
        rz(2.24, q0[1])
        x(q0[4])
        rz(0.63, q0[2])
        if mz(q0[0]):
            break
            
            rz(0.27, q0[0])
            swap(q0[2], q0[4])
            rz(2.02, q0[2])
            rz(5.41, q0[3])
            swap(q0[0], q0[2])
            swap(q0[2], q0[1])
            rz(6.04, q0[4])
            rz(2.28, q0[0])
            rz(4.92, q0[2])
            rz(3.59, q0[2])
            rz(2.66, q0[1])
            rz(0.39, q0[0])
            x.ctrl(q0[0], q0[1])
            x.ctrl(q0[0], q0[2])
            rz(1.84, q0[4])
            x.ctrl(q0[4], q0[1])
            rz(2.58, q0[3])
            rz(2.79, q0[3])
            x.ctrl(q0[4], q0[1])
            rz(1.6, q0[0])
            rz(4.85, q0[1])
            rz(0.89, q0[2])
            rz(0.73, q0[4])
            r1(3.11, q0[0])
            x.ctrl(q0[2], q0[3])
            ry(3.32, q0[2])
            t(q0[2])
            ry(6.04, q0[0])
            s(q0[0])
            z.ctrl(q0[4], q0[2])
            r1(1.37, q0[3])
            x(q0[1])
            h(q0[0])
            t(q0[0])
            r1(3.46, q0[1])
            s(q0[1])
            swap(q0[3], q0[4])
            x(q0[0])
            x.ctrl(q0[0], q0[4])
            ry(3.82, q0[4])
            z(q0[0])
            x.ctrl(q0[0], q0[3], q0[4])
            
            
def test_phase_folding():
    cudaq.set_target('qpp-cpu')
    cudaq.set_random_seed(1)
    state1 = cudaq.get_state(kernel)
    
    cudaq.set_target('phase-folding-bench')
    cudaq.set_random_seed(1)
    state2 = cudaq.get_state(kernel)
    
    assert_states_match(state1, state2)

