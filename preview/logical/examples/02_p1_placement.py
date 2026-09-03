# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Refine one P0 build to a replayable, code-agnostic P1 placement."""

import cudaq.logical as ql


@ql.machine
class TwoSlotMachine:
    compute = ql.architecture.region(
        capabilities=(
            ql.architecture.capability.logical_compute,
            ql.architecture.capability.logical_measurement,
        ),
        capacity=2,
    )


@ql.program
def bell() -> tuple[bool, bool]:
    q = ql.allocate(2, state=ql.types.zero, name="data")
    q[0] = ql.h(q[0])
    q[0], q[1] = ql.cx(q[0], q[1])
    return ql.measure_z(q[0]), ql.measure_z(q[1])


p0 = ql.compile(bell)
p1 = ql.compiler.place(
    p0,
    device=TwoSlotMachine,
    placement=(ql.architecture.colocate(p0.values.data),),
)

assert p1.stage == ql.stages.P1
assert p1.placement.input_p0 == "bell"
assert {binding.space for binding in p1.placement.bindings} == {"compute"}
assert {binding.slot for binding in p1.placement.bindings} == {0, 1}
assert ql.compiler.Build.replay(p1.serialize()).placement == p1.placement

print("P1 Bell: data[0:2] placed on compute[0:2]")
