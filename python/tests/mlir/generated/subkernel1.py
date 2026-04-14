# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../../ pytest -rP  %s

import cudaq
import math


def assert_states_match(state1, state2, tolerance=1e-7):
    assert state1.num_qubits() == state2.num_qubits()
    overlap = state1.overlap(state2)
    print(f"Real: {overlap.real}, imag: {overlap.imag}")
    assert abs(overlap.real - 1.0) < tolerance
    assert abs(overlap.imag) < tolerance
    
    
@cudaq.kernel
def iqft(q: cudaq.qview):
    N = q.size()
    for i in range(N // 2):
        swap(q[i], q[N - i - 1])
        
    for i in range(N - 1):
        h(q[i])
        j = i + 1
        for y in range(i, -1, -1):
            denom = 1 << (j - y)
            theta = -math.pi / denom
            r1.ctrl(theta, q[j], q[y])
            
    h(q[N - 1])
    
    
@cudaq.kernel
def kernel():
    q0 = cudaq.qvector(8)
    swap(q0[1], q0[2])
    rz(2.97, q0[7])
    rz(0.59, q0[3])
    rz(2.72, q0[6])
    x(q0[7])
    rz(3.71, q0[3])
    rz(0.19, q0[5])
    rz(5.9, q0[0])
    rz(6.09, q0[3])
    x.ctrl(q0[3], q0[7])
    x.ctrl(q0[7], q0[3])
    rz(5.82, q0[4])
    rz(1.17, q0[1])
    swap(q0[1], q0[5])
    rz(3.19, q0[6])
    x(q0[4])
    swap(q0[6], q0[4])
    x.ctrl(q0[3], q0[6])
    x.ctrl(q0[0], q0[5])
    rz(4.89, q0[1])
    x.ctrl(q0[5], q0[7])
    rz(1.94, q0[0])
    iqft(q0)
    swap(q0[3], q0[4])
    x.ctrl(q0[4], q0[6])
    rz(2.88, q0[5])
    rz(2.41, q0[0])
    swap(q0[2], q0[4])
    x(q0[0])
    rz(3.58, q0[5])
    rz(3.05, q0[6])
    rz(0.01, q0[5])
    swap(q0[5], q0[3])
    swap(q0[2], q0[4])
    swap(q0[4], q0[0])
    swap(q0[1], q0[6])
    rz(4.74, q0[0])
    rz(0.69, q0[4])
    x.ctrl(q0[2], q0[1])
    rz(4.13, q0[2])
    x(q0[5])
    rz(0.15, q0[1])
    rz(5.0, q0[6])
    rz(5.65, q0[4])
    x(q0[6])
    
    
def test_phase_folding():
    cudaq.set_target('qpp-cpu')
    cudaq.set_random_seed(1)
    state1 = cudaq.get_state(kernel)
    
    cudaq.set_target('phase-folding-bench')
    cudaq.set_random_seed(1)
    state2 = cudaq.get_state(kernel)
    
    assert_states_match(state1, state2)

