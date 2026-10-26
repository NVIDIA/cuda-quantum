# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                        #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# A loop's `else` block runs when the loop ends without a `break`. Two passes
# used to drop it. `cc-loop-unroll` never emitted the else region, and
# `cc-loop-induction-fusion` fused away a variable the else block updates.

import cudaq


def test_for_else():

    @cudaq.kernel
    def kernel() -> int:
        acc = 0
        for i in range(3):
            acc = acc + 1
        else:
            acc = acc + 10
        return acc

    assert kernel() == 13


def test_while_else():

    @cudaq.kernel
    def kernel() -> int:
        acc = 0
        i = 0
        while i < 3:
            acc = acc + 1
            i = i + 1
        else:
            acc = acc + 10
        return acc

    assert kernel() == 13


def test_break_skips_else():

    @cudaq.kernel
    def kernel() -> int:
        acc = 0
        for i in range(3):
            acc = acc + 1
            if i == 1:
                break
        else:
            acc = acc + 10
        return acc

    assert kernel() == 2


def test_zero_trip_still_runs_else():

    @cudaq.kernel
    def kernel() -> int:
        acc = 0
        for i in range(0):
            acc = acc + 1
        else:
            acc = acc + 10
        return acc

    assert kernel() == 10


def test_nested_else():

    @cudaq.kernel
    def kernel() -> int:
        acc = 0
        for i in range(2):
            for j in range(2):
                acc = acc + 1
            else:
                acc = acc + 10
        else:
            acc = acc + 100
        return acc

    assert kernel() == 124


def test_else_with_quantum_op():

    @cudaq.kernel
    def kernel() -> int:
        q = cudaq.qvector(2)
        acc = 0
        for i in range(2):
            acc = acc + 1
        else:
            x(q[0])
            acc = acc + 10
        return acc

    assert kernel() == 12
