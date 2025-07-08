# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os
import cudaq
import pytest


def test_run_with_integer_left_shift_operator():

    @cudaq.kernel
    def kernel(n: int) -> int:
        q = cudaq.qvector(n)
        m = mz(q)
        r = 0
        for i in range(n):
            r = r & (m[i] << i)

        return r

    results = cudaq.run(kernel, 3, shots_count=2)
    assert len(results) == 2
    assert results[0] == 0
    assert results[1] == 0


def test_run_with_non_integer_left_shift_operator():

    @cudaq.kernel
    def kernel(n: int) -> int:
        q = cudaq.qvector(n)
        m = mz(q)
        r = 0
        for i in range(n):
            r = r & (m[i] << 1.0)

        return r

    with pytest.raises(RuntimeError) as e:
        results = cudaq.run(kernel, 3, shots_count=2)
    assert "unsupported operand type(s) for '<<'; only integers supported." in str(
        e.value)


def test_run_with_integer_right_shift_operator():

    @cudaq.kernel
    def kernel(n: int) -> int:
        q = cudaq.qvector(n)
        m = mz(q)
        r = 0
        for i in range(n):
            r = r & (m[i] >> i)

        return r

    results = cudaq.run(kernel, 3, shots_count=2)
    assert len(results) == 2
    assert results[0] == 0
    assert results[1] == 0


def test_run_with_integer_bitwise_or_operator():

    @cudaq.kernel
    def kernel(n: int) -> int:
        q = cudaq.qvector(n)
        m = mz(q)
        r = 0
        for i in range(n):
            r = r | (m[i] >> i)

        return r

    results = cudaq.run(kernel, 3, shots_count=2)
    assert len(results) == 2
    assert results[0] == 0
    assert results[1] == 0


def test_run_with_integer_bitwise_xor_operator():

    @cudaq.kernel
    def kernel(n: int) -> int:
        q = cudaq.qvector(n)
        m = mz(q)
        r = 0
        for i in range(n):
            r = r ^ (m[i] >> i)

        return r

    results = cudaq.run(kernel, 3, shots_count=2)
    assert len(results) == 2
    assert results[0] == 0
    assert results[1] == 0


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
