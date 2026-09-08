# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Define a Steane code and inspect a concrete syndrome-extraction gadget."""

import cudaq.logical as ql


@ql.code
class Steane:
    """The self-dual ``[[7,1,3]]`` CSS code."""

    block = ql.codes.CSSBlock(data=7, sx=3, sz=3)
    d = 3
    hx = ((0, 1, 2, 3), (0, 1, 4, 5), (0, 2, 4, 6))
    hz = hx
    lx = (tuple(range(7)),)
    lz = lx


@ql.objective
def terminal_memory(q: ql.types.logical_qubit) -> None:
    ql.discard(q)


@ql.gadget(implements=terminal_memory)
def steane_memory(block: ql.patch[Steane]) -> None:
    block, _ = ql.extract_syndrome(block)
    block, _ = ql.mz(block.data)
    ql.discard(block)


code = ql.materialize(Steane)
gadget = ql.compile(steane_memory)
counts = ql.analysis.count(gadget)

assert "fabric.code @Steane" in code.to_mlir()
assert "fabric.gadget @steane_memory" in gadget.to_mlir()
assert Steane.n == 7 and Steane.k == 1 and Steane.d.value == 3
assert len(Steane.hx) + len(Steane.hz) == 6
assert sum(counts.operation_counts.values()) > 0

print("Steane [[7,1,3]] terminal-memory gadget:")
print(f"  physical data qubits per logical block: {Steane.n}")
print(f"  independent X/Z stabilizer checks: {len(Steane.hx) + len(Steane.hz)}")
print(f"  logical Z support: {Steane.lz[0]}")
print(f"  authored operations: {dict(counts.operation_counts)}")
