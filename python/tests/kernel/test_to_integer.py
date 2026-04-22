# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import pytest
import os
import cudaq


def testToInteger():

    @cudaq.kernel
    def toIntegerKernel(applyX: list[int]) -> int:
        q = cudaq.qvector(len(applyX))
        for i in range(len(applyX)):
            if applyX[i]:
                x(q[i])
        return cudaq.to_integer(mz(q))

    test_cases = [
        [1, 1, 1],
        [1, 1, 1, 1],
        [1, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ]

    # See reference: targettests/execution/to_integer.cpp
    expected_results = [7, 15, 5, 1, 8]
    for applyX in test_cases:
        counts = cudaq.run(toIntegerKernel, applyX)
        # All shots should yield the same integer result
        for result in counts:
            assert result == expected_results[test_cases.index(applyX)]


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
