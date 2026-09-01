# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Author a portable P0 program and inspect logical resources."""

import cudaq.logical as qlx


@qlx.program
def bell() -> tuple[bool, bool]:
    q = qlx.allocate(2, state=qlx.types.zero)
    q[0] = qlx.h(q[0])
    q[0], q[1] = qlx.cx(q[0], q[1])
    return qlx.measure_z(q[0]), qlx.measure_z(q[1])


build = qlx.compile(bell)
estimate = qlx.estimate(build, tier=qlx.estimate.Tier.LOGICAL)

assert build.stage == qlx.stages.P0
assert "lvm." not in build.to_mlir()
assert estimate.logical_qubits_peak == 2
assert estimate.actions == {"qlx_standard_h": 1, "qlx_standard_cx": 1}

print(f"P0 Bell: {estimate.logical_qubits_peak} logical qubits")
