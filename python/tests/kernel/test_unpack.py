# ============================================================================ #
# Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os
import pytest
import cudaq
from dataclasses import dataclass


@pytest.fixture(autouse=True)
def do_something():
    yield
    cudaq.__clearKernelRegistries()


def test_qubit_list():

    @cudaq.kernel
    def kernel1():
        qs1, qs2 = cudaq.qvector(2), cudaq.qvector(2)
        qs = [*qs1, *qs2]
        x(qs[:2])

    counts = cudaq.sample(kernel1)
    assert len(counts) == 1 and '1100' in counts

    @cudaq.kernel
    def kernel2():
        qs1, qs2 = cudaq.qvector(2), cudaq.qvector(2)
        qs = [*qs2, *qs1]
        x(qs[:2])

    counts = cudaq.sample(kernel2)
    assert len(counts) == 1 and '0011' in counts

    @cudaq.kernel
    def kernel3():
        q1, q2 = cudaq.qubit(), cudaq.qubit()
        qs1, qs2 = cudaq.qvector(2), cudaq.qvector(2)
        qs = [*qs1, q1, *qs2, q2]
        x(qs[2], qs[-1])

    counts = cudaq.sample(kernel3)
    assert len(counts) == 1 and '110000' in counts

    @cudaq.kernel
    def kernel4():
        q1, q2 = cudaq.qubit(), cudaq.qubit()
        qs1, qs2 = cudaq.qvector(2), cudaq.qvector(2)
        qs = [q1, *qs1, *qs2, q2]
        x(qs[2], qs[-1])

    counts = cudaq.sample(kernel4)
    assert len(counts) == 1 and '010100' in counts

    @cudaq.kernel
    def kernel5():
        qs1, qs2 = cudaq.qvector(3), cudaq.qvector(3)
        qs = [*qs1[1:], *qs2[:-1]]
        x(qs)

    counts = cudaq.sample(kernel5)
    assert len(counts) == 1 and '011110' in counts

    @dataclass(slots=True)
    class patch:
        reg1: cudaq.qview
        reg2: cudaq.qview

    @cudaq.kernel
    def kernel6():
        qs1, qs2 = cudaq.qvector(3), cudaq.qvector(3)
        p = patch(qs1, qs2)
        qs = [*p.reg1[:-1], *p.reg2[1:]]
        x(qs)

    counts = cudaq.sample(kernel6)
    assert len(counts) == 1 and '110011' in counts


def test_list_creation_failures():

    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel
        def kernel1():
            q = cudaq.qubit()
            l = [0.5, q]
            rx(l[0], l[1])

        cudaq.sample(kernel1)
    assert "non-homogenous list not allowed" in str(e.value)

    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel
        def kernel2():
            cs, q = cudaq.qvector(2), cudaq.qubit()
            l = [cs, q]
            x.ctrl(l[0], l[1])

        cudaq.sample(kernel2)
    assert "list must not contain a qvector or quantum struct" in str(e.value)
    assert "offending source -> [cs, q]" in str(e.value)

    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel
        def kernel3():
            t = cudaq.qvector(2), cudaq.qubit()
            l = [t]
            x.ctrl(l[0][0], l[0][1])

        cudaq.sample(kernel3)
    assert "list must not contain a qvector or quantum struct" in str(e.value)
    assert "offending source -> [t]" in str(e.value)

    # Unpack is currently only supported for qvectors:

    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel
        def kernel4() -> int:
            l1, l2 = [1, 2], [3, 4]
            l = [*l1, *l2]
            return len(l)

        cudaq.run(kernel4)
    assert "unpack operator `*` is only supported on qvectors" in str(e.value)
    assert "offending source -> [*l1, *l2]" in str(e.value)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
