# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Define a Steane code and a concrete syndrome-extraction gadget."""

# %%
# Import the standalone CUDA-Q Logical code and gadget APIs.
import cudaq.logical as cql


# %%
# Define the self-dual Steane CSS code from its check supports.
@cql.code
class Steane:
    block = cql.codes.CSSBlock(data=7, sx=3, sz=3)
    d = 3
    hx = ((0, 1, 2, 3), (0, 1, 4, 5), (0, 2, 4, 6))
    hz = hx
    lx = (tuple(range(7)),)
    lz = lx


# %%
# Declare the ideal logical operation implemented by the gadget.
@cql.objective
def terminal_memory(qubit: cql.types.logical_qubit) -> None:
    cql.discard(qubit)


# %%
# Implement that operation using encoded syndrome extraction and measurement.
@cql.gadget(implements=terminal_memory)
def steane_memory(block: cql.patch[Steane]) -> None:
    block, _ = cql.extract_syndrome(block)
    block, _ = cql.mz(block.data)
    cql.discard(block)


# %%
# Materialize both definitions and inspect the gadget's static operations.
code = cql.materialize(Steane)
gadget = cql.compile(steane_memory)
counts = cql.analysis.count(gadget)

assert "fabric.code @Steane" in code.to_mlir()
assert "fabric.gadget @steane_memory" in gadget.to_mlir()
assert (Steane.n, Steane.k, Steane.d.value) == (7, 1, 3)

print("Steane [[7,1,3]] terminal-memory gadget:")
print(f"  authored operations: {dict(counts.operation_counts)}")
