# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Estimate the resources of a basic CUDA-Q kernel using CUDA-Q Logical."""

#%%
# Use a simple memory kernel.
import cudaq


@cudaq.kernel()
def logical_zero_readout():
    qubits = cudaq.qvector(1)
    mz(qubits[0])


#%%
# Select a one-logical-qubit, distance-3 surface-code target.
from cudaq.logical.targets import surface_target

surface_3 = surface_target(distance=3, logical_capacity=1)
surface_3.print_stack()

#%%
# Compile and execute through the CUDA-Q Logical target.
cudaq.set_target(surface_3)

#%%
# Estimate resources and read CUDA-Q Logical's annotations.
from cudaq.logical.estimate import FabricCounts, LogicalEstimate

surface_estimates = cudaq.estimate(logical_zero_readout)
resources = FabricCounts.from_annotations(surface_estimates.annotations)

assert resources.patches_peak == 1
assert resources.logical_qubits_peak == 1

print("")
print("CUDA-Q logical-zero resources:")
print(f"  peak encoded patches: {resources.patches_peak}")
print(f"  peak protected logical qubits: {resources.logical_qubits_peak}")
print(f"  CUDA-Q Logical operation counts: {dict(resources.operation_counts)}")
print(f"  CUDA-Q Logical gadget calls: {dict(resources.gadget_calls)}")

# %%
