# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Estimate an arbitrary CUDA-Q rotation in the Clifford+T gate set."""

#%%
# Optional: uncomment to enable CUDA-Q Logical debug logging and see the compiled modules.
# ```python
# import logging
#
# logger = logging.getLogger("cudaq.logical")
# logger.setLevel(logging.DEBUG)
#
# handler = logging.StreamHandler()
# handler.setFormatter(logging.Formatter(
#     fmt="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
# ))
# logger.addHandler(handler)
# ```

#%%
# Use a non-Clifford rotation to show Clifford+T synthesis.
import cudaq


@cudaq.kernel()
def arbitrary_rotation_kernel():
    qubits = cudaq.qvector(1)
    rz(0.1234, qubits[0])
    mz(qubits[0])


#%%
# Select CUDA-Q Logical's default Clifford+T target.
from cudaq.logical.targets import clifford_t

#%%
# Compile and execute through the CUDA-Q Logical target.
cudaq.set_target(clifford_t)

#%%
# Estimate the synthesized Clifford+T resources.
from cudaq.logical.estimate import LogicalEstimate

surface_estimates = cudaq.estimate(arbitrary_rotation_kernel)
resources = LogicalEstimate.from_annotations(surface_estimates.annotations)

assert resources.actions['qlx_standard_t'] > 1

print("")
print("Logical estimates:")
print(f"  action_depth_upper_bound: {resources.action_depth_upper_bound}")
print(f"  H gates: {resources.actions['qlx_standard_h']}")
print(f"  T gates: {resources.actions['qlx_standard_t']}")

#%%
