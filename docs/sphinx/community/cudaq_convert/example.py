# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# [Begin Documentation]
import cudaq
from cudaq_convert import from_qasm_str, from_qiskit
from qiskit import QuantumCircuit

# Convert an OpenQASM 2.0 program into a CUDA-Q kernel.
qasm_kernel = from_qasm_str("""
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
h q[0];
cx q[0], q[1];
cx q[1], q[2];
""")
qasm_counts = cudaq.sample(qasm_kernel)
print(qasm_counts)

# Convert a Qiskit `QuantumCircuit` into a CUDA-Q kernel.
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qiskit_kernel = from_qiskit(qc)
qiskit_counts = cudaq.sample(qiskit_kernel)
print(qiskit_counts)
# [End Documentation]

# Both circuits prepare GHZ states, so only the all-zeros and all-ones outcomes
# may appear.
qasm_ghz = qasm_counts.probability("000") + qasm_counts.probability("111")
qiskit_ghz = qiskit_counts.probability("00") + qiskit_counts.probability("11")
assert abs(qasm_ghz - 1) < 1e-9
assert abs(qiskit_ghz - 1) < 1e-9
