# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Author a portable P0 program and inspect logical resources."""

import cudaq.logical as ql


@ql.program
def bell() -> tuple[bool, bool]:
    q = ql.allocate(2, state=ql.types.zero)
    q[0] = ql.h(q[0])
    q[0], q[1] = ql.cx(q[0], q[1])
    return ql.measure_z(q[0]), ql.measure_z(q[1])


build = ql.compile(bell)
estimate = ql.estimate(build, tier=ql.estimate.Tier.LOGICAL)

assert build.stage == ql.stages.P0
assert "lvm." not in build.to_mlir()
assert estimate.logical_qubits_peak == 2
assert estimate.actions == {"qlx_standard_h": 1, "qlx_standard_cx": 1}

print(f"P0 Bell: {estimate.logical_qubits_peak} logical qubits")
