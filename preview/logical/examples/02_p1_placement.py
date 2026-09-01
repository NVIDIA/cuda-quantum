# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Refine one P0 build to a replayable, code-agnostic P1 placement."""

import cudaq.logical as qlx


@qlx.machine
class TwoSlotMachine:
    compute = qlx.architecture.region(
        capabilities=(
            qlx.architecture.capability.logical_compute,
            qlx.architecture.capability.logical_measurement,
        ),
        capacity=2,
    )


@qlx.program
def bell() -> tuple[bool, bool]:
    q = qlx.allocate(2, state=qlx.types.zero, name="data")
    q[0] = qlx.h(q[0])
    q[0], q[1] = qlx.cx(q[0], q[1])
    return qlx.measure_z(q[0]), qlx.measure_z(q[1])


p0 = qlx.compile(bell)
p1 = qlx.compiler.place(
    p0,
    device=TwoSlotMachine,
    placement=(qlx.architecture.colocate(p0.values.data),),
)

assert p1.stage == qlx.stages.P1
assert p1.placement.input_p0 == "bell"
assert {binding.space for binding in p1.placement.bindings} == {"compute"}
assert {binding.slot for binding in p1.placement.bindings} == {0, 1}
assert qlx.compiler.Build.replay(p1.serialize()).placement == p1.placement

print("P1 Bell: data[0:2] placed on compute[0:2]")
