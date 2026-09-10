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
# Select the target that stops after portable logical compilation.
cudaq.set_target(cql.targets.estimator)
cql.targets.estimator.print_stack()

# %%
# Estimate the kernel and recover the typed logical-resource annotation.
estimate = cudaq.estimate(bell_pair)
resources = cql.estimate.LogicalEstimate.from_annotations(estimate.annotations)

assert resources.logical_qubits_peak == 2
assert resources.actions == {
    "qlx_standard_h": 1,
    "qlx_standard_cx": 1,
}

print("Logical Bell-pair resources:")
print(f"  peak logical qubits: {resources.logical_qubits_peak}")
print(f"  logical action depth: {resources.action_depth_upper_bound}")
print(f"  logical actions: {dict(resources.actions)}")
