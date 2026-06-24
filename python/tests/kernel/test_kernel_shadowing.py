# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import pytest


def test_kernel_call_in_if_block():
    """
    Tests for issue #3963!
    Calling a kernel inside an if block would fail with
    because the AST visitor loop variable would shadow the
    kernel name in the call stack. Added checks for
    Python before 3.12 and 3.12+ to avoid this!
    """

    @cudaq.kernel
    def b():
        pass

    @cudaq.kernel
    def a(i: int):
        if i == 0:
            b()

    a(0)
    a(1)


# test if else
def test_kernel_call_in_if_else_block():

    @cudaq.kernel
    def apply_x(q: cudaq.qubit):
        x(q)

    @cudaq.kernel
    def apply_h(q: cudaq.qubit):
        h(q)

    @cudaq.kernel
    def test(cond: bool) -> bool:
        q = cudaq.qubit()
        if cond:
            apply_x(q)
        else:
            apply_h(q)
        return mz(q)

    assert cudaq.run(test, True, shots_count=10) == [True] * 10


# test nested if
def test_kernel_call_in_nested_if_block():

    @cudaq.kernel
    def flip(q: cudaq.qubit):
        x(q)

    @cudaq.kernel
    def test(a: bool, b: bool) -> bool:
        q = cudaq.qubit()
        if a:
            if b:
                flip(q)
        return mz(q)

    assert cudaq.run(test, True, True, shots_count=10) == [True] * 10
    assert cudaq.run(test, True, False, shots_count=10) == [False] * 10
    assert cudaq.run(test, False, True, shots_count=10) == [False] * 10


# test for loop
def test_kernel_call_in_for_loop():

    @cudaq.kernel
    def flip(q: cudaq.qubit):
        x(q)

    @cudaq.kernel
    def test(n: int) -> list[bool]:
        qs = cudaq.qvector(n)
        for i in range(n):
            flip(qs[i])
        return mz(qs)

    assert cudaq.run(test, 3, shots_count=10) == [[True, True, True]] * 10


# test multiple ifs (but not nested)
def test_multiple_kernels_called_in_if_blocks():

    @cudaq.kernel
    def apply_x(q: cudaq.qubit):
        x(q)

    @cudaq.kernel
    def apply_y(q: cudaq.qubit):
        y(q)

    @cudaq.kernel
    def apply_z(q: cudaq.qubit):
        z(q)

    @cudaq.kernel
    def test(choice: int) -> bool:
        q = cudaq.qubit()
        if choice == 0:
            apply_x(q)
        if choice == 1:
            apply_y(q)
        if choice == 2:
            apply_z(q)
        return mz(q)

    assert cudaq.run(test, 0, shots_count=10) == [True] * 10
    assert cudaq.run(test, 1, shots_count=10) == [True] * 10
    assert cudaq.run(test, 2, shots_count=10) == [False] * 10


# test multiple same names in one if
def test_kernel_with_same_name_as_loop_variable():

    @cudaq.kernel
    def b():
        pass

    @cudaq.kernel
    def n():
        pass

    @cudaq.kernel
    def i():
        pass

    @cudaq.kernel
    def test(cond: bool):
        if cond:
            b()
            n()
            i()

    test(True)
    test(False)


# test while loop
def test_kernel_call_in_while_loop():

    @cudaq.kernel
    def flip(q: cudaq.qubit):
        x(q)

    @cudaq.kernel
    def test() -> bool:
        q = cudaq.qubit()
        i = 0
        while i < 3:
            flip(q)
            i += 1
        return mz(q)

    assert cudaq.run(test, shots_count=10) == [True] * 10
