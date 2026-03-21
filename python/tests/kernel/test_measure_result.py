# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import os
import pytest
import dataclasses


@pytest.fixture(autouse=True)
def do_something():
    yield
    cudaq.__clearKernelRegistries()


def test_return_measure_result():

    @cudaq.kernel
    def device(q: cudaq.qubit) -> cudaq.measure_result:
        x(q)
        return mz(q)

    @cudaq.kernel
    def entry_point() -> bool:
        q = cudaq.qubit()
        r = device(q)
        return r

    results = cudaq.run(entry_point, shots_count=10)
    assert len(results) == 10
    for r in results:
        assert r == True


def test_return_measure_result_list():

    @cudaq.kernel
    def kernel(q: cudaq.qview) -> list[cudaq.measure_result]:
        h(q[0])
        x.ctrl(q[0], q[1])
        return mz(q)

    @cudaq.kernel
    def entry_point() -> int:
        q = cudaq.qvector(2)
        r = kernel(q)
        return int(r[0]) + int(r[1])

    results = cudaq.run(entry_point, shots_count=10)
    assert len(results) == 10
    for r in results:
        assert r in [0, 2]


def test_direct_invocation():

    @cudaq.kernel
    def kernel() -> cudaq.measure_result:
        q = cudaq.qubit()
        x(q)
        return mz(q)

    with pytest.raises(RuntimeError) as e:
        kernel()
    assert "cannot be invoked directly" in str(e.value)

    @cudaq.kernel
    def kernel() -> list[cudaq.measure_result]:
        q = cudaq.qvector(2)
        h(q[0])
        x.ctrl(q[0], q[1])
        return mz(q)

    with pytest.raises(RuntimeError) as e:
        kernel()
    assert "cannot be invoked directly" in str(e.value)


def test_unsupported_return_types():

    @cudaq.kernel
    def kernel() -> cudaq.measure_result:
        q = cudaq.qubit()
        x(q)
        return mz(q)

    with pytest.raises(RuntimeError) as e:
        cudaq.run(kernel, shots_count=10)
    assert "Unsupported data type" in str(e.value)

    @cudaq.kernel
    def kernel() -> tuple[cudaq.measure_result, cudaq.measure_result]:
        q = cudaq.qvector(2)
        x(q[0])
        return mz(q[0]), mz(q[1])

    with pytest.raises(RuntimeError) as e:
        cudaq.run(kernel, shots_count=10)
    assert "Unsupported data type" in str(e.value)

    @dataclasses.dataclass(slots=True)
    class Result:
        r1: cudaq.measure_result
        r2: int

    @cudaq.kernel
    def kernel() -> Result:
        q = cudaq.qubit()
        x(q)
        return Result(mz(q), 42)

    with pytest.raises(RuntimeError) as e:
        cudaq.run(kernel, shots_count=10)
    assert "Unsupported data type" in str(e.value)


def test_list_from_measure():

    @cudaq.kernel
    def kernel() -> list[bool]:
        q = cudaq.qvector(3)
        h(q)
        r0 = mz(q[0])
        r1 = mz(q[1])
        if r0 and r1:
            x(q[2])
        r2 = mz(q[2])
        return [r0, r1, r2]

    results = cudaq.run(kernel, shots_count=10)
    assert len(results) == 10
    for shot in results:
        assert len(shot) == 3
        assert all(isinstance(b, bool) for b in shot)


def test_tuple_from_measure():

    @cudaq.kernel
    def kernel_direct_return() -> tuple[bool, bool]:
        q = cudaq.qvector(2)
        x(q[0])
        r0 = mz(q[0])
        r1 = mz(q[1])
        return (r0, r1)

    results = cudaq.run(kernel_direct_return, shots_count=10)
    assert len(results) == 10
    for r in results:
        assert r == (True, False)

    @cudaq.kernel
    def kernel_assign_to_var() -> tuple[bool, bool]:
        q = cudaq.qvector(2)
        x(q[0])
        r0 = mz(q[0])
        r1 = mz(q[1])
        t = (r0, r1)
        return t

    results = cudaq.run(kernel_assign_to_var, shots_count=10)
    assert len(results) == 10
    for r in results:
        assert r == (True, False)


def test_list_comprehension():

    @cudaq.kernel
    def kernel() -> list[bool]:
        q = cudaq.qvector(3)
        x(q)
        return [mz(qi) for qi in q]

    results = cudaq.run(kernel, shots_count=10)
    assert len(results) == 10
    for shot in results:
        assert len(shot) == 3
        assert all(b == True for b in shot)

    @cudaq.kernel
    def kernel() -> list[bool]:
        q = cudaq.qvector(3)
        x(q)
        return [mz(q[i]) for i in range(3)]

    with pytest.raises(RuntimeError) as e:
        cudaq.run(kernel, shots_count=10)
    assert "only supported when iterating" in str(e.value)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-srP"])
