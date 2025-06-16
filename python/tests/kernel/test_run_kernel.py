# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os
import time
from dataclasses import dataclass

import cudaq
import numpy as np
import pytest

list_err_msg = 'does not yet support returning `list` from entry-point kernels'


def is_close(actual, expected):
    return np.isclose(actual, expected, atol=1e-6)


def is_close_array(actual, expected):
    assert len(actual) == len(expected)
    res = True
    for a, e in zip(actual, expected):
        res = res and np.isclose(e, a, atol=1e-6)
    return res


@pytest.fixture(autouse=True)
def do_something():
    yield
    cudaq.__clearKernelRegistries()


@cudaq.kernel
def simple(numQubits: int) -> int:
    qubits = cudaq.qvector(numQubits)
    h(qubits.front())
    for i, qubit in enumerate(qubits.front(numQubits - 1)):
        x.ctrl(qubit, qubits[i + 1])
    result = 0
    for i in range(numQubits):
        if mz(qubits[i]):
            result += 1
    return result


def test_simple_run_ghz():
    shots = 100
    qubitCount = 4
    results = cudaq.run(simple, qubitCount, shots_count=shots)
    print(results)
    assert len(results) == shots
    non_zero_count = 0
    for result in results:
        assert result == 0 or result == qubitCount  # 00..0 or 1...11
        if result == qubitCount:
            non_zero_count += 1

    assert non_zero_count > 0


def test_simple_run_ghz_with_noise():
    cudaq.set_target("density-matrix-cpu")
    shots = 100
    qubitCount = 4
    depol = cudaq.Depolarization2(.5)
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("cx", depol)
    results = cudaq.run(simple,
                        qubitCount,
                        shots_count=shots,
                        noise_model=noise)
    print(results)
    assert len(results) == shots
    noisy_count = 0
    for result in results:
        if result != 0 and result != qubitCount:
            noisy_count += 1
    assert noisy_count > 0
    cudaq.reset_target()


def test_run_async():
    shots = 100
    qubitCounts = [4, 5, 6, 7, 8]
    resultHandles = []
    for qubitCount in qubitCounts:
        resultHandles.append(
            cudaq.run_async(simple, qubitCount, shots_count=shots))
        print(f"({time.time()}) Launch async run for {qubitCount} qubits")

    for i in range(len(qubitCounts)):
        results = resultHandles[i].get()
        qubitCount = qubitCounts[i]
        print(f"({time.time()}) Result for {qubitCount} qubits: {results}")
        assert len(results) == shots
        non_zero_count = 0
        for result in results:
            assert result == 0 or result == qubitCount  # 00..0 or 1...11
            if result == qubitCount:
                non_zero_count += 1

        assert non_zero_count > 0


def test_run_async_with_noise():
    cudaq.set_target("density-matrix-cpu")
    shots = 100
    qubitCount = 3
    depol = cudaq.Depolarization2(.5)
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("cx", depol)
    results = cudaq.run_async(simple,
                              qubitCount,
                              shots_count=shots,
                              noise_model=noise).get()
    print(results)
    assert len(results) == shots
    noisy_count = 0
    for result in results:
        if result != 0 and result != qubitCount:
            noisy_count += 1
    assert noisy_count > 0
    cudaq.reset_target()


def test_return_bool():

    @cudaq.kernel
    def simple_bool_no_args() -> bool:
        return True

    results = cudaq.run(simple_bool_no_args, shots_count=2)
    assert len(results) == 2
    assert results[0] == True
    assert results[1] == True

    @cudaq.kernel
    def simple_bool(numQubits: int) -> bool:
        qubits = cudaq.qvector(numQubits)
        return True

    results = cudaq.run(simple_bool, 2, shots_count=2)
    assert len(results) == 2
    assert results[0] == True
    assert results[1] == True


def test_return_int():

    @cudaq.kernel
    def simple_int_no_args() -> int:
        return -43

    results = cudaq.run(simple_int_no_args, shots_count=2)
    assert len(results) == 2
    assert results[0] == -43
    assert results[1] == -43

    @cudaq.kernel
    def simple_int(numQubits: int) -> int:
        qubits = cudaq.qvector(numQubits)
        return numQubits + 1

    results = cudaq.run(simple_int, 2, shots_count=2)
    assert len(results) == 2
    assert results[0] == 3
    assert results[1] == 3


def test_return_int8():

    @cudaq.kernel
    def simple_int8_no_args() -> np.int8:
        return -43

    results = cudaq.run(simple_int8_no_args, shots_count=2)
    assert len(results) == 2
    assert results[0] == -43
    assert results[1] == -43

    @cudaq.kernel
    def simple_int8(numQubits: int) -> np.int8:
        qubits = cudaq.qvector(numQubits)
        return numQubits + 1

    results = cudaq.run(simple_int8, 2, shots_count=2)
    assert len(results) == 2
    assert results[0] == 3
    assert results[1] == 3


def test_return_int16():

    @cudaq.kernel
    def simple_int16_no_args() -> np.int16:
        return -43

    results = cudaq.run(simple_int16_no_args, shots_count=2)
    assert len(results) == 2
    assert results[0] == -43
    assert results[1] == -43

    @cudaq.kernel
    def simple_int16(numQubits: int) -> np.int16:
        qubits = cudaq.qvector(numQubits)
        return numQubits + 1

    results = cudaq.run(simple_int16, 2, shots_count=2)
    assert len(results) == 2
    assert results[0] == 3
    assert results[1] == 3


def test_return_int32():

    @cudaq.kernel
    def simple_int32_no_args() -> np.int32:
        return -43

    results = cudaq.run(simple_int32_no_args, shots_count=2)
    assert len(results) == 2
    assert results[0] == -43
    assert results[1] == -43

    @cudaq.kernel
    def simple_int32(numQubits: int) -> np.int32:
        qubits = cudaq.qvector(numQubits)
        return numQubits + 1

    results = cudaq.run(simple_int32, 2, shots_count=2)
    assert len(results) == 2
    assert results[0] == 3
    assert results[1] == 3


def test_return_int64():

    @cudaq.kernel
    def simple_int64_no_args() -> np.int64:
        return -43

    results = cudaq.run(simple_int64_no_args, shots_count=2)
    assert len(results) == 2
    assert results[0] == -43
    assert results[1] == -43

    @cudaq.kernel
    def simple_int64(numQubits: int) -> np.int64:
        qubits = cudaq.qvector(numQubits)
        return numQubits + 1

    results = cudaq.run(simple_int64, 2, shots_count=2)
    assert len(results) == 2
    assert results[0] == 3
    assert results[1] == 3


def test_return_float():

    @cudaq.kernel
    def simple_float_no_args() -> float:
        return -43.2

    results = cudaq.run(simple_float_no_args, shots_count=2)
    assert len(results) == 2
    assert results[0] == -43.2
    assert results[1] == -43.2

    @cudaq.kernel()
    def simple_float(numQubits: int) -> float:
        return numQubits + 1

    results = cudaq.run(simple_float, 2, shots_count=2)
    assert len(results) == 2
    assert results[0] == 3.0
    assert results[1] == 3.0


def test_return_float32():

    @cudaq.kernel
    def simple_float32_no_args() -> np.float32:
        return -43.2

    results = cudaq.run(simple_float32_no_args, shots_count=2)
    assert len(results) == 2
    assert is_close(results[0], -43.2)
    assert is_close(
        results[1],
        -43.2,
    )

    @cudaq.kernel
    def simple_float32(numQubits: int) -> np.float32:
        return numQubits + 1

    results = cudaq.run(simple_float32, 2, shots_count=2)
    assert len(results) == 2
    assert results[0] == 3.0
    assert results[1] == 3.0


def test_return_float64():

    @cudaq.kernel
    def simple_float64_no_args() -> np.float64:
        return -43.2

    results = cudaq.run(simple_float64_no_args, shots_count=2)
    assert len(results) == 2
    assert results[0] == -43.2
    assert results[1] == -43.2

    @cudaq.kernel
    def simple_float64(numQubits: int) -> np.float64:
        return numQubits + 1

    results = cudaq.run(simple_float64, 2, shots_count=2)
    assert len(results) == 2
    assert results[0] == 3.0
    assert results[1] == 3.0


def test_return_list_from_device_kernel():

    @cudaq.kernel
    def kernel_that_returns_list() -> list[int]:
        return [1, 2, 3]

    @cudaq.kernel
    def entry_point_kernel() -> int:
        result = kernel_that_returns_list()
        return len(result)

    results = cudaq.run(entry_point_kernel, shots_count=2)

    assert len(results) == 2
    assert results[0] == 3
    assert results[1] == 3

    @cudaq.kernel
    def incrementer(i: int) -> int:
        return i + 1

    @cudaq.kernel
    def kernel_with_list_arg(arg: list[int]) -> list[int]:
        result = arg
        for i in result:
            incrementer(i)
        return result

    @cudaq.kernel
    def caller_kernel(arg: list[int]) -> int:
        values = kernel_with_list_arg(arg)
        result = 0
        for v in values:
            result += v
        return result

    results = cudaq.run(caller_kernel, [4, 5, 6], shots_count=1)
    assert len(results) == 1
    assert results[0] == 15  # 4+1 + 5+1 + 6+1 = 15


def test_return_list_bool():

    @cudaq.kernel
    def simple_list_bool_no_args() -> list[bool]:
        return [True, False, True]

    with pytest.raises(RuntimeError) as e:
        cudaq.run(simple_list_bool_no_args, shots_count=2)
    assert list_err_msg in str(e.value)

    @cudaq.kernel
    def simple_list_bool(n: int) -> list[bool]:
        qubits = cudaq.qvector(n)
        return [True, False, True]

    with pytest.raises(RuntimeError) as e:
        cudaq.run(simple_list_bool, 2, shots_count=2)
    assert list_err_msg in str(e.value)

    @cudaq.kernel
    def simple_list_bool_args(n: int, t: list[bool]) -> list[bool]:
        qubits = cudaq.qvector(n)
        return t

    with pytest.raises(RuntimeError) as e:
        cudaq.run(simple_list_bool_args, 2, [True, False, True])
    assert list_err_msg in str(e.value)

    @cudaq.kernel
    def simple_list_bool_args_no_broadcast(t: list[bool]) -> list[bool]:
        qubits = cudaq.qvector(2)
        return t

    with pytest.raises(RuntimeError) as e:
        cudaq.run(simple_list_bool_args_no_broadcast, [True, False, True])
    assert list_err_msg in str(e.value)


def test_return_list_int():

    @cudaq.kernel
    def simple_list_int_no_args() -> list[int]:
        return [-13, 5, 42]

    with pytest.raises(RuntimeError) as e:
        cudaq.run(simple_list_int_no_args, shots_count=2)
    assert list_err_msg in str(e.value)

    @cudaq.kernel
    def simple_list_int(n: int, t: list[int]) -> list[int]:
        qubits = cudaq.qvector(n)
        return t

    with pytest.raises(RuntimeError) as e:
        cudaq.run(simple_list_int, 2, [-13, 5, 42], shots_count=2)
    assert list_err_msg in str(e.value)


def test_return_list_int8():

    @cudaq.kernel
    def simple_list_int8_no_args() -> list[np.int8]:
        return [-13, 5, 42]

    with pytest.raises(RuntimeError) as e:
        cudaq.run(simple_list_int8_no_args, shots_count=2)
    assert list_err_msg in str(e.value)

    @cudaq.kernel
    def simple_list_int8(n: int, t: list[np.int8]) -> list[np.int8]:
        qubits = cudaq.qvector(n)
        return t

    with pytest.raises(RuntimeError) as e:
        cudaq.run(simple_list_int8, 2, [-13, 5, 42], shots_count=2)
    assert list_err_msg in str(e.value)


def test_return_list_int16():

    @cudaq.kernel
    def simple_list_int16_no_args() -> list[np.int16]:
        return [-13, 5, 42]

    with pytest.raises(RuntimeError) as e:
        cudaq.run(simple_list_int16_no_args, shots_count=2)
    assert list_err_msg in str(e.value)

    @cudaq.kernel
    def simple_list_int16(n: int, t: list[np.int16]) -> list[np.int16]:
        qubits = cudaq.qvector(n)
        return t

    with pytest.raises(RuntimeError) as e:
        cudaq.run(simple_list_int16, 2, [-13, 5, 42], shots_count=2)
    assert list_err_msg in str(e.value)


def test_return_list_int32():

    @cudaq.kernel
    def simple_list_int32_no_args() -> list[np.int32]:
        return [-13, 5, 42]

    with pytest.raises(RuntimeError) as e:
        cudaq.run(simple_list_int32_no_args, shots_count=2)
    assert list_err_msg in str(e.value)

    @cudaq.kernel
    def simple_list_int32(n: int, t: list[np.int32]) -> list[np.int32]:
        qubits = cudaq.qvector(n)
        return t

    with pytest.raises(RuntimeError) as e:
        cudaq.run(simple_list_int32, 2, [-13, 5, 42], shots_count=2)
    assert list_err_msg in str(e.value)


def test_return_list_int64():

    @cudaq.kernel
    def simple_list_int64_no_args() -> list[np.int64]:
        return [-13, 5, 42]

    with pytest.raises(RuntimeError) as e:
        cudaq.run(simple_list_int64_no_args, shots_count=2)
    assert list_err_msg in str(e.value)

    @cudaq.kernel
    def simple_list_int64(n: int, t: list[np.int64]) -> list[np.int64]:
        qubits = cudaq.qvector(n)
        return t

    with pytest.raises(RuntimeError) as e:
        cudaq.run(simple_list_int64, 2, [-13, 5, 42], shots_count=2)
    assert list_err_msg in str(e.value)


def test_return_list_float():

    @cudaq.kernel
    def simple_list_float_no_args() -> list[float]:
        return [-13.2, 5., 42.99]

    with pytest.raises(RuntimeError) as e:
        cudaq.run(simple_list_float_no_args, shots_count=2)
    assert list_err_msg in str(e.value)

    @cudaq.kernel
    def simple_list_float(n: int, t: list[float]) -> list[float]:
        qubits = cudaq.qvector(n)
        return t

    with pytest.raises(RuntimeError) as e:
        cudaq.run(simple_list_float, 2, [-13.2, 5.0, 42.99], shots_count=2)
    assert list_err_msg in str(e.value)


def test_return_list_float32():

    @cudaq.kernel
    def simple_list_float32_no_args() -> list[np.float32]:
        return [-13.2, 5., 42.99]

    with pytest.raises(RuntimeError) as e:
        cudaq.run(simple_list_float32_no_args, shots_count=2)
    assert list_err_msg in str(e.value)

    @cudaq.kernel
    def simple_list_float32(n: int, t: list[np.float32]) -> list[np.float32]:
        qubits = cudaq.qvector(n)
        return t

    with pytest.raises(RuntimeError) as e:
        cudaq.run(simple_list_float32, 2, [-13.2, 5.0, 42.99], shots_count=2)
    assert list_err_msg in str(e.value)


def test_return_list_float64():

    @cudaq.kernel
    def simple_list_float64_no_args() -> list[np.float64]:
        return [-13.2, 5., 42.99]

    with pytest.raises(RuntimeError) as e:
        cudaq.run(simple_list_float64_no_args, shots_count=2)
    assert list_err_msg in str(e.value)

    @cudaq.kernel
    def simple_list_float64(n: int, t: list[np.float64]) -> list[np.float64]:
        qubits = cudaq.qvector(n)
        return t

    with pytest.raises(RuntimeError) as e:
        cudaq.run(simple_list_float64, 2, [-13.2, 5.0, 42.99], shots_count=2)
    assert list_err_msg in str(e.value)


# Test tuples
# TODO: Define spec for using tuples in kernels
# https://github.com/NVIDIA/cuda-quantum/issues/3031


def test_return_tuple_int_float():

    @cudaq.kernel
    def simple_tuple_int_float_no_args() -> tuple[int, float]:
        return (13, 42.3)

    with pytest.raises(RuntimeError) as e:
        cudaq.run(simple_tuple_int_float_no_args)
    assert 'Use of tuples is not supported in kernels' in str(e.value)

    @cudaq.kernel
    def simple_tuple_int_float(n: int, t: tuple[int,
                                                float]) -> tuple[int, float]:
        qubits = cudaq.qvector(n)
        return t

    with pytest.raises(RuntimeError) as e:
        cudaq.run(simple_tuple_int_float, 2, (13, 42.3))
    assert 'Use of tuples is not supported in kernels' in str(e.value)

    @cudaq.kernel
    def simple_tuple_int_float_assign(
            n: int, t: tuple[int, float]) -> tuple[int, float]:
        qubits = cudaq.qvector(n)
        t[0] = -14
        t[1] = 11.5
        return t

    with pytest.raises(RuntimeError) as e:
        cudaq.run(simple_tuple_int_float_assign, 2, (-13, 11.5))
    assert 'Use of tuples is not supported in kernels' in str(e.value)

    @cudaq.kernel
    def simple_tuple_int_float_error(
            n: int, t: tuple[int, float]) -> tuple[bool, float]:
        qubits = cudaq.qvector(n)
        return t

    with pytest.raises(RuntimeError) as e:
        cudaq.run(simple_tuple_int_float_error, 2, (-13, 11.5))
    assert 'Use of tuples is not supported in kernels' in str(e.value)


def test_return_tuple_float_int():

    @cudaq.kernel
    def simple_tuple_float_int_no_args() -> tuple[float, int]:
        return (42.3, 13)

    with pytest.raises(RuntimeError) as e:
        cudaq.run(simple_tuple_float_int_no_args)
    assert 'Use of tuples is not supported in kernels' in str(e.value)

    @cudaq.kernel
    def simple_tuple_float_int(n: int, t: tuple[float,
                                                int]) -> tuple[float, int]:
        qubits = cudaq.qvector(n)
        return t

    with pytest.raises(RuntimeError) as e:
        cudaq.run(simple_tuple_float_int, 2, (42.3, 13))
    assert 'Use of tuples is not supported in kernels' in str(e.value)


def test_return_tuple_bool_int():

    @cudaq.kernel
    def simple_tuple_bool_int_no_args() -> tuple[bool, int]:
        return (True, 13)

    with pytest.raises(RuntimeError) as e:
        cudaq.run(simple_tuple_bool_int_no_args)
    assert 'Use of tuples is not supported in kernels' in str(e.value)

    @cudaq.kernel
    def simple_tuple_bool_int(n: int, t: tuple[bool, int]) -> tuple[bool, int]:
        qubits = cudaq.qvector(n)
        return t

    with pytest.raises(RuntimeError) as e:
        cudaq.run(simple_tuple_bool_int, 2, (True, 13))
    assert 'Use of tuples is not supported in kernels' in str(e.value)


def test_return_tuple_int_bool():

    @cudaq.kernel
    def simple_tuple_int_bool_no_args() -> tuple[int, bool]:
        return (-13, True)

    with pytest.raises(RuntimeError) as e:
        cudaq.run(simple_tuple_int_bool_no_args)
    assert 'Use of tuples is not supported in kernels' in str(e.value)

    @cudaq.kernel
    def simple_tuple_int_bool(n: int, t: tuple[int, bool]) -> tuple[int, bool]:
        qubits = cudaq.qvector(n)
        return t

    with pytest.raises(RuntimeError) as e:
        cudaq.run(simple_tuple_int_bool, 2, (-13, True))
    assert 'Use of tuples is not supported in kernels' in str(e.value)


def test_return_tuple_int32_bool():

    @cudaq.kernel
    def simple_tuple_int32_bool_no_args() -> tuple[np.int32, bool]:
        return (-13, True)

    with pytest.raises(RuntimeError) as e:
        cudaq.run(simple_tuple_int32_bool_no_args)
    assert 'Use of tuples is not supported in kernels' in str(e.value)

    @cudaq.kernel
    def simple_tuple_int32_bool_no_args() -> tuple[np.int32, bool]:
        return (np.int32(-13), True)

    with pytest.raises(RuntimeError) as e:
        cudaq.run(simple_tuple_int32_bool_no_args)
    assert 'Use of tuples is not supported in kernels' in str(e.value)

    @cudaq.kernel
    def simple_tuple_int32_bool(
            n: int, t: tuple[np.int32, bool]) -> tuple[np.int32, bool]:
        qubits = cudaq.qvector(n)
        return t

    with pytest.raises(RuntimeError) as e:
        cudaq.run(simple_tuple_int32_bool, 2, (-13, True))
    assert 'Use of tuples is not supported in kernels' in str(e.value)


def test_return_tuple_bool_int_float():

    @cudaq.kernel
    def simple_tuple_bool_int_float_no_args() -> tuple[bool, int, float]:
        return (True, 13, 42.3)

    with pytest.raises(RuntimeError) as e:
        cudaq.run(simple_tuple_bool_int_float_no_args)
    assert 'Use of tuples is not supported in kernels' in str(e.value)

    @cudaq.kernel
    def simple_tuple_bool_int_float(
            n: int, t: tuple[bool, int, float]) -> tuple[bool, int, float]:
        qubits = cudaq.qvector(n)
        return t

    with pytest.raises(RuntimeError) as e:
        cudaq.run(simple_tuple_bool_int_float, 2, (True, 13, 42.3))
    assert 'Use of tuples is not supported in kernels' in str(e.value)


def test_return_dataclass_int_bool():

    @dataclass
    class MyClass:
        x: int
        y: bool

    @cudaq.kernel
    def simple_dataclass_int_bool_no_args() -> MyClass:
        return MyClass(-16, True)

    results = cudaq.run(simple_dataclass_int_bool_no_args, shots_count=2)
    assert len(results) == 2
    assert results[0] == MyClass(-16, True)
    assert results[1] == MyClass(-16, True)
    assert results[0].x == -16
    assert results[0].y == True

    @cudaq.kernel
    def simple_return_dataclass_int_bool(n: int, t: MyClass) -> MyClass:
        qubits = cudaq.qvector(n)
        return t

    results = cudaq.run(simple_return_dataclass_int_bool,
                        2,
                        MyClass(-16, True),
                        shots_count=2)
    assert len(results) == 2
    assert results[0] == MyClass(-16, True)
    assert results[1] == MyClass(-16, True)
    assert results[0].x == -16
    assert results[0].y == True


def test_return_dataclass_bool_int():

    @dataclass
    class MyClass:
        x: bool
        y: int

    @cudaq.kernel
    def simple_dataclass_bool_int_no_args() -> MyClass:
        return MyClass(True, 17)

    results = cudaq.run(simple_dataclass_bool_int_no_args, shots_count=2)
    assert len(results) == 2
    assert results[0] == MyClass(True, 17)
    assert results[1] == MyClass(True, 17)
    assert results[0].x == True
    assert results[0].y == 17

    @cudaq.kernel
    def simple_return_dataclass_bool_int(n: int, t: MyClass) -> MyClass:
        qubits = cudaq.qvector(n)
        return t

    results = cudaq.run(simple_return_dataclass_bool_int,
                        2,
                        MyClass(True, 17),
                        shots_count=2)
    assert len(results) == 2
    assert results[0] == MyClass(True, 17)
    assert results[1] == MyClass(True, 17)
    assert results[0].x == True
    assert results[0].y == 17


def test_return_dataclass_float_int():

    @dataclass
    class MyClass:
        x: float
        y: int

    @cudaq.kernel
    def simple_dataclass_float_int_no_args() -> MyClass:
        return MyClass(42.5, 17)

    results = cudaq.run(simple_dataclass_float_int_no_args, shots_count=2)
    assert len(results) == 2
    assert results[0] == MyClass(42.5, 17)
    assert results[1] == MyClass(42.5, 17)
    assert results[0].x == 42.5
    assert results[0].y == 17

    @cudaq.kernel
    def simple_dataclass_float_int(n: int, t: MyClass) -> MyClass:
        qubits = cudaq.qvector(n)
        return t

    results = cudaq.run(simple_dataclass_float_int,
                        2,
                        MyClass(42.5, 17),
                        shots_count=2)
    assert len(results) == 2
    assert results[0] == MyClass(42.5, 17)
    assert results[1] == MyClass(42.5, 17)
    assert results[0].x == 42.5
    assert results[0].y == 17


def test_return_dataclass_list_int_bool():

    @dataclass
    class MyClass:
        x: list[int]
        y: bool

    @cudaq.kernel
    def simple_return_dataclass(n: int, t: MyClass) -> MyClass:
        qubits = cudaq.qvector(n)
        return t

    # TODO: Support recursive aggregate types in kernels.
    # results = cudaq.run(simple_return_dataclass, 2, MyClass([0,1], 18), shots_count=2)
    # assert len(results) == 2
    # assert results[0] == MyClass([0,1], 18)
    # assert results[1] == MyClass([0,1], 18)


def test_return_dataclass_tuple_bool():

    @dataclass
    class MyClass:
        x: tuple[int, bool]
        y: bool

    @cudaq.kernel
    def simple_return_dataclass(n: int, t: MyClass) -> MyClass:
        qubits = cudaq.qvector(n)
        return t

    # TODO: Support recursive aggregate types in kernels.
    # results = cudaq.run(simple_return_dataclass, 2, MyClass((0, True), 19), shots_count=2)
    # assert len(results) == 2
    # assert results[0] == MyClass((0, True), 19)
    # assert results[1] == MyClass((0, True), 19)


def test_return_dataclass_dataclass_bool():

    @dataclass
    class MyClass1:
        x: int
        y: bool

    @dataclass
    class MyClass2:
        x: MyClass1
        y: bool

    @cudaq.kernel
    def simple_return_dataclass(n: int, t: MyClass2) -> MyClass2:
        qubits = cudaq.qvector(n)
        return t

    # TODO: Support recursive aggregate types in kernels.
    # results = cudaq.run(simple_return_dataclass, 2, MyClass2(MyClass1(0,True), 20), shots_count=2)
    # assert len(results) == 2
    # assert results[0] == MyClass2(MyClass1(0,True), 20)
    # assert results[1] == MyClass2(MyClass1(0,True), 20)


def test_run_errors():

    @cudaq.kernel
    def simple_no_return(numQubits: int):
        qubits = cudaq.qvector(numQubits)

    @cudaq.kernel
    def simple_no_args() -> int:
        qubits = cudaq.qvector(2)
        return 1

    @cudaq.kernel
    def simple(numQubits: int) -> int:
        qubits = cudaq.qvector(numQubits)
        return 1

    with pytest.raises(RuntimeError) as e:
        cudaq.run(simple_no_return, 2)
    assert '`cudaq.run` only supports kernels that return a value.' in repr(e)

    with pytest.raises(TypeError) as e:
        cudaq.run(simple, 2, shots_count=-1)
    assert 'incompatible function arguments.' in repr(e)

    with pytest.raises(RuntimeError) as e:
        cudaq.run(simple, shots_count=100)
    assert 'Invalid number of arguments passed to run:0 expected 1' in repr(e)

    with pytest.raises(RuntimeError) as e:
        print(cudaq.run(simple_no_args, 2, shots_count=100))
    assert 'Invalid number of arguments passed to run:1 expected 0' in repr(e)


def test_modify_struct():

    @dataclass
    class MyClass:
        x: int
        y: bool

    @cudaq.kernel
    def simple_struct(t: MyClass) -> MyClass:
        q = cudaq.qubit()
        t.x = 42
        return t

    results = cudaq.run(simple_struct, MyClass(-13, True), shots_count=2)
    print(results)
    assert len(results) == 2
    assert results[0] == MyClass(42, True)
    assert results[1] == MyClass(42, True)

    @dataclass
    class Foo:
        x: bool
        y: float
        z: int

    @cudaq.kernel
    def kernel(t: Foo) -> Foo:
        q = cudaq.qubit()
        t.z = 100
        t.y = 3.14
        t.x = True
        return t

    results = cudaq.run(kernel, Foo(False, 6.28, 17), shots_count=2)
    print(results)
    assert len(results) == 2
    assert results[0] == Foo(True, 3.14, 100)
    assert results[1] == Foo(True, 3.14, 100)


def test_create_and_modify_struct():

    @dataclass
    class MyClass:
        x: int
        y: bool

    @cudaq.kernel
    def simple_struct() -> MyClass:
        q = cudaq.qubit()
        t = MyClass(-13, True)
        t.x = 42
        return t

    results = cudaq.run(simple_struct, shots_count=2)
    print(results)
    assert len(results) == 2
    assert results[0] == MyClass(42, True)
    assert results[1] == MyClass(42, True)

    @dataclass
    class Bar:
        x: bool
        y: bool
        z: float

    @cudaq.kernel
    def kernel(n: int) -> Bar:
        q = cudaq.qvector(n)
        t = Bar(False, False, 4.14)
        t.x = True
        t.y = True
        return t

    results = cudaq.run(kernel, 2, shots_count=1)
    print(results)
    assert len(results) == 1
    assert results[0] == Bar(True, True, 4.14)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
