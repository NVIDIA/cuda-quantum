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
# Estimate the kernel. For logical targets, `estimate.resources` reports the
# logical program: gate counts, qubits, and depth. Measurements appear as `mz`
# and qubit preparations as `reset`.
estimate = cudaq.estimate(bell_pair)
resources = estimate.resources

assert resources.num_qubits == 2
assert resources.count("h") == 1
assert resources.count_controls("x", 1) == 1
assert resources.count("mz") == 2

print("Logical Bell-pair resources:")
print(f"  peak logical qubits: {resources.num_qubits}")
print(f"  logical operations: {resources.count()}")
print(f"  logical depth (serial upper bound): {resources.depth}")
print(f"  logical gate counts: {resources.to_dict()}")

# %%
# Going deeper: `resources` is a summary of the full logical estimate stored in
# `estimate.annotations["LOGICAL"]`. That annotation also carries details that
# `Resources` cannot express, such as idle sites, discards, synthesis demand,
# and resource requests. Rehydrate it as a typed object when you need them.
logical = cql.estimate.LogicalEstimate.from_annotations(estimate.annotations)
assert logical.logical_qubits_peak == resources.num_qubits
print(f"  logical estimate: {logical}")
