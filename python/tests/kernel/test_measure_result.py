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
    def kernel() -> cudaq.measure_result:
        q = cudaq.qubit()
        x(q)
        return mz(q)

    results = cudaq.run(kernel, shots_count=10)
    assert len(results) == 10
    for r in results:
        assert isinstance(r, cudaq.measure_result)
        assert bool(r) == True


def test_return_measure_result_list():

    @cudaq.kernel
    def kernel() -> list[cudaq.measure_result]:
        q = cudaq.qvector(2)
        h(q[0])
        x.ctrl(q[0], q[1])
        return mz(q)

    results = cudaq.run(kernel, shots_count=10)
    assert len(results) == 10
    for r in results:
        assert isinstance(r, list)
        assert len(r) == 2
        assert r[0] == r[1]


def test_return_measure_result_tuple():

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
