# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Place a standalone logical program on a logical machine."""

# %%
# Import the standalone CUDA-Q Logical authoring and compiler API.
import cudaq.logical as cql


# %%
# Define a two-slot logical machine independently of any QEC code.
@cql.machine
class TwoSlotMachine:
    compute = cql.architecture.region(
        capabilities=(
            cql.architecture.capability.logical_compute,
            cql.architecture.capability.logical_measurement,
        ),
        capacity=2,
    )


# %%
# Author a portable Bell program with a named allocation.
@cql.program
def bell() -> tuple[bool, bool]:
    qubits = cql.allocate(2, state=cql.types.zero, name="data")
    qubits[0] = cql.h(qubits[0])
    qubits[0], qubits[1] = cql.cx(qubits[0], qubits[1])
    return cql.measure_z(qubits[0]), cql.measure_z(qubits[1])


# %%
# Compile and place the logical owners into the machine's compute region.
logical = cql.compile(bell)
placed = cql.compiler.place(
    logical,
    device=TwoSlotMachine,
    placement=(cql.architecture.colocate(logical.values.data),),
)

assert placed.stage == cql.stages.P1
assert {binding.space for binding in placed.placement.bindings} == {"compute"}
assert {binding.slot for binding in placed.placement.bindings} == {0, 1}
assert cql.compiler.Build.replay(
    placed.serialize()).placement == placed.placement

print("Bell data[0:2] placed on compute[0:2]")
