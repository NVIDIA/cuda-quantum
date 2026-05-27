# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import cudaq
import sys

cudaq.set_target("quake_fake")

width = 4


@cudaq.kernel
def loop_payload_stress() -> int:
    q = cudaq.qvector(width)

    for repeat in range(2):
        for i in range(width):
            x(q[i])

    i = 0
    while i < width:
        x(q[i])
        i += 1

    return cudaq.to_integer(cudaq.to_bools(mz(q)))


try:
    results = cudaq.run(loop_payload_stress, shots_count=3)
    assert len(results) == 3
    for result in results:
        assert result == 15, f"expected 15, got {result}"
except Exception as e:
    print(e)
    sys.exit(1)
