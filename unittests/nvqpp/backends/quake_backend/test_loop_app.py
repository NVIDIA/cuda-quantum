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

# IMPORTANT: loop-count annotations are part of this test contract. Kernel
# names ending in `_expected_N_loop(s)` tell the quake fake mock server how
# many `cc.loop` operations must be present in the submitted client payload
# before the server does its own execution-time full unroll.


# Expected remaining payload loops: the repeat loop, plus the generated
# measurement-result reduction from `to_integer(to_bools(mz(q)))`. The loops
# that index quantum data (`q[i]`) are unrolled so wire IDs are static.
@cudaq.kernel
def loop_payload_stress_expected_2_loops() -> int:
    q = cudaq.qvector(width)

    # KEEP: this classical repeat loop does not index quantum data directly.
    for repeat in range(2):
        # UNROLL: this loop indexes `q[i]`, so wire IDs must become static.
        for i in range(width):
            x(q[i])

    i = 0
    # UNROLL: this loop indexes `q[i]`, so wire IDs must become static.
    while i < width:
        x(q[i])
        i += 1

    # KEEP: this expression generates a measurement-result reduction loop.
    return cudaq.to_integer(cudaq.to_bools(mz(q)))


# Expected remaining payload loop: the loop over measurement results. This loop
# does not access quantum data, so it should stay rolled.
@cudaq.kernel
def measurement_result_loop_expected_1_loop() -> int:
    q = cudaq.qvector(width)

    x(q[0])
    x(q[2])

    bits = mz(q)
    result = 0
    # KEEP: this loop only indexes measurement results, not quantum data.
    for i in range(width):
        if bits[i]:
            result |= 1 << i

    return result


# Expected remaining payload loops: outer, group, classical accumulation, and
# measurement-result accumulation. The quantum-data loop over `q[wire]` is
# unrolled for static wire IDs.
@cudaq.kernel
def nested_mixed_loop_payload_expected_4_loops() -> int:
    q = cudaq.qvector(width)

    classical = 0
    # KEEP: this classical outer loop does not index quantum data directly.
    for outer in range(1):
        # KEEP: this classical group loop does not index quantum data directly.
        for group in range(2):
            # UNROLL: this loop indexes `q[wire]`, so wire IDs become static.
            for wire in range(width):
                x(q[wire])

            # KEEP: this loop is classical accumulation only.
            for i in range(3):
                classical += group + i

    x(q[0])

    bits = mz(q)
    result = classical
    # KEEP: this loop only indexes measurement results, not quantum data.
    for i in range(width):
        if bits[i]:
            result += 16

    return result


def check_results(kernel, expected):
    results = cudaq.run(kernel, shots_count=3)
    assert len(results) == 3
    for result in results:
        assert result == expected, f"expected {expected}, got {result}"


try:
    check_results(loop_payload_stress_expected_2_loops, 15)
    check_results(measurement_result_loop_expected_1_loop, 5)
    check_results(nested_mixed_loop_payload_expected_4_loops, 25)
except Exception as e:
    print(e)
    sys.exit(1)
