# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Estimate the logical resources used by an ordinary CUDA-Q kernel."""

# %%
# Import CUDA-Q and the CUDA-Q Logical target and result APIs.
import cudaq
import cudaq.logical as cql


# %%
# Author the workload as an ordinary CUDA-Q kernel.
@cudaq.kernel
def bell_pair():
    qubits = cudaq.qvector(2)
    h(qubits[0])
    x.ctrl(qubits[0], qubits[1])
    mz(qubits)


# %%
# Select the target that stops at a portable logical representation.
cudaq.set_target(cql.targets.estimator)

# %%
# Estimate the kernel and read its logical-qubit and gate counts.
estimate = cudaq.estimate(bell_pair)
resources = estimate.resources

print("Logical Bell-pair resources:")
print(f"  peak logical qubits: {resources.num_qubits}")
print(f"  Hadamard actions: {resources.count('h')}")
print(f"  controlled-X actions: {resources.count_controls('x', 1)}")
