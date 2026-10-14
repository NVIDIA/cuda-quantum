# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Compare Clifford+T resources at several rotation precisions."""

# %%
# Import CUDA-Q and the CUDA-Q Logical target and result APIs.
import cudaq
import cudaq.logical as cql


# %%
# Author a kernel containing an angle that requires approximate synthesis.
@cudaq.kernel
def arbitrary_rotation():
    qubit = cudaq.qubit()
    rz(0.1234, qubit)
    mz(qubit)


# %%
# Define a helper that makes the target precision an explicit input.
def estimate_at_precision(precision: float):
    target = cql.targets.clifford_t_target(precision=precision)
    cudaq.set_target(target)
    estimate = cudaq.estimate(arbitrary_rotation)
    return cql.estimate.LogicalEstimate.from_annotations(estimate.annotations)


# %%
# Compare successively tighter synthesis tolerances.
precisions = (1.0e-2, 1.0e-6, 1.0e-10)
resources_by_precision = {
    precision: estimate_at_precision(precision) for precision in precisions
}

assert all(resources.actions["qlx_standard_t"] > 0
           for resources in resources_by_precision.values())

print("Clifford+T resources by rotation precision:")
for precision, resources in resources_by_precision.items():
    print(f"  precision={precision:g}: "
          f"T={resources.actions['qlx_standard_t']}, "
          f"depth={resources.action_depth_upper_bound}")
