# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Estimate physical resources for a surface-code memory kernel."""

# %%
import cudaq
import cudaq.logical as cql


# %%
# Prepare and measure one logical qubit in the Z basis.
@cudaq.kernel
def logical_zero_memory(logical_qubits: int):
    qubits = cudaq.qvector(logical_qubits)
    mz(qubits)


# %%
# Select a distance-three surface-code target with room for one logical qubit.
logical_qubits = 1
target = cql.targets.surface_physical_target(
    logical_capacity=logical_qubits,
    distance=3,
)
cudaq.set_target(target)

# %%
# The target handles encoding, physical lowering, and scheduling.
estimate = cudaq.estimate(logical_zero_memory, logical_qubits)
schedule = estimate.annotations["SCHEDULE"]

print("Surface-code memory resources:")
print(f"  physical qubits: {schedule['physical_qubits']}")
print(f"  makespan: {schedule['makespan_ns']:.1f} ns")
