# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                        #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# A `for` loop that never runs leaves its variable unbound in Python. A kernel
# cannot raise `NameError`, and the read used to return whatever was in the
# variable's stack slot, so the bridge rejects it unless it can prove the loop
# runs.

import pytest

import cudaq

kDiagnostic = 'after a loop that can run zero times'


def test_dynamic_range_is_rejected():

    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel
        def kernel(n: int):
            q = cudaq.qvector(3)
            for i in range(n):
                h(q[i])
            x(q[i])

        kernel.compile()

    assert kDiagnostic in str(e.value)
    assert 'loop variable(s) i' in str(e.value)


def test_dynamic_qvector_is_rejected():

    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel
        def kernel(n: int):
            q = cudaq.qvector(n)
            for i in range(q.size()):
                h(q[i])
            x(q[i])

        kernel.compile()

    assert kDiagnostic in str(e.value)


def test_for_else_is_rejected():

    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel
        def kernel(n: int):
            q = cudaq.qvector(3)
            for i in range(n):
                h(q[i])
            else:
                h(q[0])
            x(q[i])

        kernel.compile()

    assert kDiagnostic in str(e.value)


def test_constant_range_is_allowed():

    @cudaq.kernel
    def one_arg() -> int:
        q = cudaq.qvector(3)
        for i in range(3):
            h(q[i])
        x(q[i])
        return i

    assert one_arg() == 2

    # A non-zero start or a non-unit step is accepted too. What these read
    # back is currently wrong (a separate bug: the post-loop value is the
    # normalized counter), so only check that the read is not rejected.
    @cudaq.kernel
    def two_args() -> int:
        for i in range(1, 3):
            pass
        return i

    @cudaq.kernel
    def strided() -> int:
        for i in range(0, 6, 2):
            pass
        return i

    @cudaq.kernel
    def decrementing() -> int:
        for i in range(2, -1, -1):
            pass
        return i

    two_args.compile()
    strided.compile()
    decrementing.compile()


def test_static_qvector_size_is_allowed():

    @cudaq.kernel
    def over_range() -> int:
        q = cudaq.qvector(3)
        for i in range(len(q)):
            h(q[i])
        return i

    @cudaq.kernel
    def over_enumerate() -> int:
        q = cudaq.qvector(3)
        for i, r in enumerate(q):
            h(r)
        return i

    assert over_range() == 2
    assert over_enumerate() == 2


def test_otherwise_bound_variable_is_allowed():

    @cudaq.kernel
    def assigned_before(n: int) -> int:
        q = cudaq.qvector(3)
        i = 0
        for i in range(n):
            h(q[i])
        return i

    @cudaq.kernel
    def assigned_after(n: int) -> int:
        q = cudaq.qvector(3)
        for i in range(n):
            h(q[i])
        i = 1
        return i

    @cudaq.kernel
    def bound_by_earlier_loop(n: int) -> int:
        q = cudaq.qvector(3)
        for i in range(3):
            h(q[i])
        for i in range(n):
            x(q[i])
        return i

    assert assigned_before(0) == 0
    assert assigned_after(0) == 1
    assert bound_by_earlier_loop(0) == 2


def test_loop_local_variable_is_unaffected():

    @cudaq.kernel
    def kernel(n: int) -> int:
        q = cudaq.qvector(3)
        total = 0
        for i in range(n):
            h(q[i])
            total = total + i
        return total

    assert kernel(0) == 0
    assert kernel(3) == 3
