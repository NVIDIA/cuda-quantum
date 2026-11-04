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
    return cudaq.estimate(arbitrary_rotation)


# %%
# Compare successively tighter synthesis tolerances.
precisions = (1.0e-2, 1.0e-6, 1.0e-10)
estimates = {
    precision: estimate_at_precision(precision) for precision in precisions
}

assert all(estimate.resources.count("t") > 0 for estimate in estimates.values())

print("Clifford+T resources by rotation precision:")
for precision, estimate in estimates.items():
    resources = estimate.resources
    print(f"  precision={precision:g}: "
          f"T={resources.count('t')}, "
          f"T-depth={resources.t_depth}, "
          f"depth={resources.depth}")

# %%
# Going deeper: the full logical estimate lives in
# `estimate.annotations["LOGICAL"]`. For example, `synthesis_demand` lists the
# gates that a fault-tolerant machine has to synthesize or distill.
tightest = estimates[precisions[-1]]
print("Synthesis demand at the tightest precision:")
print(f"  {tightest.annotations['LOGICAL']['synthesis_demand']}")
