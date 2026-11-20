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
# Prepare and measure one or more logical qubits in the Z basis.
@cudaq.kernel
def logical_zero_memory(logical_qubits: int):
    qubits = cudaq.qvector(logical_qubits)
    mz(qubits)


# %%
# Select a distance-11 surface-code target with room for three logical qubits.
logical_qubits = 3
target = cql.targets.surface_physical_target(
    logical_capacity=logical_qubits,
    distance=11,
)
cudaq.set_target(target)

# %%
# The target handles encoding, physical lowering, and scheduling.
estimate = cudaq.estimate(logical_zero_memory, logical_qubits)
print(estimate)

#%%
schedule = estimate.annotations["SCHEDULE"]

print("Surface-code memory resources:")
print(f"  physical qubits: {schedule['physical_qubits']}")
print(f"  makespan: {schedule['makespan_ns']:.1f} ns")

# %%
# The target is a stack of layers, one per level of abstraction in CUDA-Q
# Logical. Print the stack to see which layers it is made of.
target.print_stack()

#%%
# Each layer contributes its own annotations to the estimate. Print them all
# to see the full estimates across every layer.
for e in estimate.annotations:
    print(f"{e}:")
    for item in estimate.annotations[e]:
        print(f"    {item}: {estimate.annotations[e][item]}")
