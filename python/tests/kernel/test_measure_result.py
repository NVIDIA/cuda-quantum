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
    assert "unsupported return type from entry-point kernel" in str(e.value)

    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel
        def kernel() -> tuple[cudaq.measure_result, cudaq.measure_result]:
            q = cudaq.qvector(2)
            x(q[0])
            return mz(q[0]), mz(q[1])

        cudaq.run(kernel, shots_count=10)

    assert "Unsupported data type" in str(e.value)

    @dataclasses.dataclass(slots=True)
    class Result:
        r1: cudaq.measure_result
        r2: int

    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel
        def kernel() -> Result:
            q = cudaq.qubit()
            x(q)
            return Result(mz(q), 42)

        cudaq.run(kernel, shots_count=10)

    assert "Unsupported data type" in str(e.value)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-srP"])
