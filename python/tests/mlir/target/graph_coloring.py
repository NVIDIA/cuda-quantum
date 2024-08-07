# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../.. python3 %s
# RUN: PYTHONPATH=../../.. python3 %s --target quantinuum --emulate

import numpy as np

import cudaq


@cudaq.kernel
def init_state(qubits: cudaq.qvector, theta: float):
    ry(theta, qubits[0])
    h.ctrl(qubits[0], qubits[1])
    x(qubits[1])

    ry(theta, qubits[2])
    h.ctrl(qubits[2], qubits[3])
    x(qubits[3])

    ry(theta, qubits[4])
    h.ctrl(qubits[4], qubits[5])
    x(qubits[5])

    ry(theta, qubits[6])
    h.ctrl(qubits[6], qubits[7])
    x(qubits[7])


@cudaq.kernel
def reflect_uniform(qubits: cudaq.qvector, theta: float):
    cudaq.adjoint(init_state, qubits, theta)
    x(qubits)
    z.ctrl(qubits[0], qubits[1], qubits[2], qubits[3], qubits[4], qubits[5],
           qubits[6], qubits[7])
    x(qubits)
    init_state(qubits, theta)


@cudaq.kernel
def oracle(cs: cudaq.qvector, target: cudaq.qubit):
    x.ctrl(cs[0], ~cs[1], cs[2], ~cs[3], cs[5], target)
    x.ctrl(cs[0], ~cs[1], cs[2], ~cs[3], cs[7], target)
    x.ctrl(cs[0], ~cs[1], ~cs[3], cs[4], cs[7], target)
    x.ctrl(cs[1], cs[2], cs[3], cs[4], target)
    x.ctrl(cs[1], ~cs[2], cs[3], cs[6], target)
    x.ctrl(~cs[1], cs[2], cs[4], cs[7], target)
    x.ctrl(cs[0], cs[1], cs[2], cs[3], cs[5], target)
    x.ctrl(cs[1], ~cs[2], cs[5], cs[6], target)
    x.ctrl(~cs[1], cs[3], cs[4], target)
    x.ctrl(cs[1], cs[4], cs[7], target)
    x.ctrl(cs[0], ~cs[1], ~cs[3], cs[5], cs[6], target)
    x.ctrl(~cs[2], cs[3], ~cs[5], cs[6], target)
    x.ctrl(cs[0], cs[1], cs[2], cs[3], cs[7], target)
    x.ctrl(cs[0], cs[1], cs[3], cs[4], ~cs[7], target)
    x.ctrl(cs[2], ~cs[7], target)
    x.ctrl(cs[0], cs[1], cs[3], ~cs[5], cs[6], target)
    x.ctrl(cs[2], ~cs[3], cs[6], target)
    x.ctrl(cs[2], ~cs[5], ~cs[6], target)
    x.ctrl(~cs[2], cs[3], cs[4], cs[7], target)


@cudaq.kernel
def grover(theta: float):
    qubits = cudaq.qvector(8)
    ancilla = cudaq.qubit()

    #/ Initialization
    x(ancilla)
    h(ancilla)
    init_state(qubits, theta)

    # Iterations
    oracle(qubits, ancilla)
    reflect_uniform(qubits, theta)

    oracle(qubits, ancilla)
    reflect_uniform(qubits, theta)

    mz(qubits)


theta = 2. * np.arccos(1. / np.sqrt(3.))
result = cudaq.sample(grover, theta)

# sort the results
sortedResult = {
    k: v for k, v in sorted(result.items(), key=lambda item: item[1])
}
strings = list(sortedResult.keys())
strings.reverse()

most_probable = set()
for i in range(12):
    most_probable.add(strings[i])
    for j in range(0, 8, 2):
        print(strings[i][j:j + 2], end=' ')
    print()

assert "01101101" in most_probable
assert "10110110" in most_probable
assert "11101101" in most_probable
assert "01110110" in most_probable
assert "01100111" in most_probable
assert "01111001" in most_probable
assert "10111001" in most_probable
assert "11011110" in most_probable
assert "11100111" in most_probable
assert "10011011" in most_probable
assert "10011110" in most_probable
assert "11011011" in most_probable

# CHECK-DAG: 01 10 11 01
# CHECK-DAG: 10 11 01 10
# CHECK-DAG: 11 10 11 01
# CHECK-DAG: 01 11 01 10
# CHECK-DAG: 01 10 01 11
# CHECK-DAG: 01 11 10 01
# CHECK-DAG: 10 11 10 01
# CHECK-DAG: 11 01 11 10
# CHECK-DAG: 11 10 01 11
# CHECK-DAG: 10 01 10 11
# CHECK-DAG: 10 01 11 10
# CHECK-DAG: 11 01 10 11
