# ============================================================================ #
# Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os
import cudaq
import pytest


def test_integer_left_shift():

    @cudaq.kernel
    def kernel() -> int:
        # 3 << 2 == 12
        r = 3 << 2
        return r

    results = cudaq.run(kernel, shots_count=1)
    assert results == [12]


def test_integer_right_shift():

    @cudaq.kernel
    def kernel() -> int:
        # 8 >> 2 == 2
        r = 8 >> 2
        return r

    results = cudaq.run(kernel, shots_count=1)
    assert results == [2]


def test_integer_bitwise_and():

    @cudaq.kernel
    def kernel() -> int:
        # 6 & 3 == 2
        r = 6 & 3
        return r

    results = cudaq.run(kernel, shots_count=1)
    assert results == [2]


def test_integer_bitwise_or():

    @cudaq.kernel
    def kernel() -> int:
        # 6 | 3 == 7
        r = 6 | 3
        return r

    results = cudaq.run(kernel, shots_count=1)
    assert results == [7]


def test_integer_bitwise_xor():

    @cudaq.kernel
    def kernel() -> int:
        # 6 ^ 3 == 5
        r = 6 ^ 3
        return r

    results = cudaq.run(kernel, shots_count=1)
    assert results == [5]


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

    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel
        def kernel(n: int) -> int:
            q = cudaq.qvector(n)
            m = mz(q)
            r = 0
            for i in range(n):
                r = r & (m[i] << 1.0)

            return r

        results = cudaq.run(kernel, 3, shots_count=2)
    assert "unsupported operand type(s) for BinOp.LShift; only integers supported" in str(
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
