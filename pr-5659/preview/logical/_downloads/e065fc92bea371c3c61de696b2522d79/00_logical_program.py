# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Author a portable logical program without a CUDA-Q kernel."""

# %%
# Import the standalone CUDA-Q Logical authoring and compiler API.
import cudaq.logical as cql


# %%
# Define a Bell program directly with linear logical-qubit values.
@cql.program
def bell() -> tuple[bool, bool]:
    qubits = cql.allocate(2, state=cql.types.zero)
    qubits[0] = cql.h(qubits[0])
    qubits[0], qubits[1] = cql.cx(qubits[0], qubits[1])
    return cql.measure_z(qubits[0]), cql.measure_z(qubits[1])


# %%
# Compile the program and request its logical resource counts directly.
build = cql.compile(bell)
resources = cql.estimate(build, tier=cql.estimate.Tier.LOGICAL)

assert build.stage == cql.stages.P0
assert resources.logical_qubits_peak == 2
assert resources.actions == {
    "qlx_standard_h": 1,
    "qlx_standard_cx": 1,
}

print(f"Portable Bell program: {resources.logical_qubits_peak} logical qubits")
