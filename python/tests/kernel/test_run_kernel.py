# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os
from dataclasses import dataclass
from typing import Callable

import cudaq
import numpy as np
import warnings
import pytest

skipIfBraketNotInstalled = pytest.mark.skipif(
    not (cudaq.has_target("braket")),
    reason='Could not find `braket` in installation')


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
    shots = 20
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
        result = arg.copy()
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
    assert results[0] == 15  # 4 + 5 + 6 = 15


def test_return_list_bool():

    @cudaq.kernel
    def simple_list_bool_no_args() -> list[bool]:
        return [True, False, True]

    results = cudaq.run(simple_list_bool_no_args, shots_count=2)
    assert len(results) == 2
    assert results[0] == [True, False, True]
    assert results[1] == [True, False, True]

    @cudaq.kernel
    def simple_list_bool(n: int) -> list[bool]:
        qubits = cudaq.qvector(n)
        return [True, False, True]

    results = cudaq.run(simple_list_bool, 2, shots_count=2)
    assert len(results) == 2
    assert results[0] == [True, False, True]
    assert results[1] == [True, False, True]

    @cudaq.kernel
    def simple_list_bool_args(n: int, t: list[bool]) -> list[bool]:
        qubits = cudaq.qvector(n)
        return t.copy()

    results = cudaq.run(simple_list_bool_args,
                        2, [True, False, True],
                        shots_count=2)
    assert len(results) == 2
    assert results[0] == [True, False, True]
    assert results[1] == [True, False, True]

    @cudaq.kernel
    def simple_list_bool_args_no_broadcast(t: list[bool]) -> list[bool]:
        qubits = cudaq.qvector(2)
        return t.copy()

    results = cudaq.run(simple_list_bool_args_no_broadcast, [True, False, True],
                        shots_count=2)
    assert len(results) == 2
    assert results[0] == [True, False, True]
    assert results[1] == [True, False, True]


def test_return_list_int():

    @cudaq.kernel
    def simple_list_int_no_args() -> list[int]:
        return [-13, 5, 42]

    results = cudaq.run(simple_list_int_no_args, shots_count=2)
    assert len(results) == 2
    assert results[0] == [-13, 5, 42]
    assert results[1] == [-13, 5, 42]

    @cudaq.kernel
    def simple_list_int(n: int, t: list[int]) -> list[int]:
        qubits = cudaq.qvector(n)
        return t.copy()

    results = cudaq.run(simple_list_int, 2, [-13, 5, 42], shots_count=2)
    assert len(results) == 2
    assert results[0] == [-13, 5, 42]
    assert results[1] == [-13, 5, 42]


def test_return_list_int8():

    @cudaq.kernel
    def simple_list_int8_no_args() -> list[np.int8]:
        return [-13, 5, 42]

    results = cudaq.run(simple_list_int8_no_args, shots_count=2)
    assert len(results) == 2
    assert results[0] == [-13, 5, 42]
    assert results[1] == [-13, 5, 42]

    @cudaq.kernel
    def simple_list_int8(n: int, t: list[np.int8]) -> list[np.int8]:
        qubits = cudaq.qvector(n)
        return t.copy()

    results = cudaq.run(simple_list_int8, 2, [-13, 5, 42], shots_count=2)
    assert len(results) == 2
    assert results[0] == [-13, 5, 42]
    assert results[1] == [-13, 5, 42]


def test_return_list_int16():

    @cudaq.kernel
    def simple_list_int16_no_args() -> list[np.int16]:
        return [-13, 5, 42]

    results = cudaq.run(simple_list_int16_no_args, shots_count=2)
    assert len(results) == 2
    assert results[0] == [-13, 5, 42]
    assert results[1] == [-13, 5, 42]

    @cudaq.kernel
    def simple_list_int16(n: int, t: list[np.int16]) -> list[np.int16]:
        qubits = cudaq.qvector(n)
        return t.copy()

    results = cudaq.run(simple_list_int16, 2, [-13, 5, 42], shots_count=2)
    assert len(results) == 2
    assert results[0] == [-13, 5, 42]
    assert results[1] == [-13, 5, 42]


def test_return_list_int32():

    @cudaq.kernel
    def simple_list_int32_no_args() -> list[np.int32]:
        return [-13, 5, 42]

    results = cudaq.run(simple_list_int32_no_args, shots_count=2)
    assert len(results) == 2
    assert results[0] == [-13, 5, 42]
    assert results[1] == [-13, 5, 42]

    @cudaq.kernel
    def simple_list_int32(n: int, t: list[np.int32]) -> list[np.int32]:
        qubits = cudaq.qvector(n)
        return t.copy()

    results = cudaq.run(simple_list_int32, 2, [-13, 5, 42], shots_count=2)
    assert len(results) == 2
    assert results[0] == [-13, 5, 42]
    assert results[1] == [-13, 5, 42]


def test_return_list_int64():

    @cudaq.kernel
    def simple_list_int64_no_args() -> list[np.int64]:
        return [-13, 5, 42]

    results = cudaq.run(simple_list_int64_no_args, shots_count=2)
    assert len(results) == 2
    assert results[0] == [-13, 5, 42]
    assert results[1] == [-13, 5, 42]

    @cudaq.kernel
    def simple_list_int64(n: int, t: list[np.int64]) -> list[np.int64]:
        qubits = cudaq.qvector(n)
        return t.copy()

    results = cudaq.run(simple_list_int64, 2, [-13, 5, 42], shots_count=2)
    assert len(results) == 2
    assert results[0] == [-13, 5, 42]
    assert results[1] == [-13, 5, 42]


def test_return_list_float():

    @cudaq.kernel
    def simple_list_float_no_args() -> list[float]:
        return [-13.2, 5., 42.99]

    results = cudaq.run(simple_list_float_no_args, shots_count=2)
    assert len(results) == 2
    assert is_close_array(results[0], [-13.2, 5., 42.99])
    assert is_close_array(results[1], [-13.2, 5., 42.99])

    @cudaq.kernel
    def simple_list_float(n: int, t: list[float]) -> list[float]:
        qubits = cudaq.qvector(n)
        return t.copy()

    results = cudaq.run(simple_list_float,
                        2, [-13.2, 5.0, 42.99],
                        shots_count=2)
    assert len(results) == 2
    assert is_close_array(results[0], [-13.2, 5., 42.99])
    assert is_close_array(results[1], [-13.2, 5., 42.99])


def test_return_list_float32():

    @cudaq.kernel
    def simple_list_float32_no_args() -> list[np.float32]:
        return [-13.2, 5., 42.99]

    results = cudaq.run(simple_list_float32_no_args, shots_count=2)
    assert len(results) == 2
    assert is_close_array(results[0], [-13.2, 5., 42.99])
    assert is_close_array(results[1], [-13.2, 5., 42.99])

    @cudaq.kernel
    def simple_list_float32(n: int, t: list[np.float32]) -> list[np.float32]:
        qubits = cudaq.qvector(n)
        return t.copy()

    results = cudaq.run(simple_list_float32,
                        2, [-13.2, 5.0, 42.99],
                        shots_count=2)
    assert len(results) == 2
    assert is_close_array(results[0], [-13.2, 5., 42.99])
    assert is_close_array(results[1], [-13.2, 5., 42.99])


def test_return_list_float64():

    @cudaq.kernel
    def simple_list_float64_no_args() -> list[np.float64]:
        return [-13.2, 5., 42.99]

    results = cudaq.run(simple_list_float64_no_args, shots_count=2)
    assert len(results) == 2
    assert is_close_array(results[0], [-13.2, 5., 42.99])
    assert is_close_array(results[1], [-13.2, 5., 42.99])

    @cudaq.kernel
    def simple_list_float64(n: int, t: list[np.float64]) -> list[np.float64]:
        qubits = cudaq.qvector(n)
        return t.copy()

    results = cudaq.run(simple_list_float64,
                        2, [-13.2, 5.0, 42.99],
                        shots_count=2)
    assert len(results) == 2
    assert is_close_array(results[0], [-13.2, 5., 42.99])
    assert is_close_array(results[1], [-13.2, 5., 42.99])


def test_return_list_large_size():
    # Returns a large list (dynamic size) to stress test the code generation

    @cudaq.kernel
    def kernel_with_dynamic_int_array_input(n: int, t: list[int]) -> list[int]:
        qubits = cudaq.qvector(n)
        return t.copy()

    @cudaq.kernel
    def kernel_with_dynamic_float_array_input(n: int,
                                              t: list[float]) -> list[float]:
        qubits = cudaq.qvector(n)
        return t.copy()

    @cudaq.kernel
    def kernel_with_dynamic_bool_array_input(n: int,
                                             t: list[bool]) -> list[bool]:
        qubits = cudaq.qvector(n)
        return t.copy()

    # Test with various sizes (validate dynamic output logging)
    for array_size in [10, 15, 100, 167, 1000]:
        input_array = list(np.random.randint(-1000, 1000, size=array_size))
        results = cudaq.run(kernel_with_dynamic_int_array_input,
                            2,
                            input_array,
                            shots_count=2)
        assert len(results) == 2
        assert results[0] == input_array
        assert results[1] == input_array

        input_array_float = list(
            np.random.uniform(-1000.0, 1000.0, size=array_size))
        results = cudaq.run(kernel_with_dynamic_float_array_input,
                            2,
                            input_array_float,
                            shots_count=2)
        assert len(results) == 2
        assert is_close_array(results[0], input_array_float)
        assert is_close_array(results[1], input_array_float)

        input_array_bool = []
        for _ in range(array_size):
            input_array_bool.append(True if np.random.rand() > 0.5 else False)
        results = cudaq.run(kernel_with_dynamic_bool_array_input,
                            2,
                            input_array_bool,
                            shots_count=2)
        assert len(results) == 2
        assert results[0] == input_array_bool
        assert results[1] == input_array_bool


def test_return_dynamics_measure_results():

    @cudaq.kernel
    def measure_all_qubits(numQubits: int) -> list[bool]:
        # Number of qubits is dynamic
        qubits = cudaq.qvector(numQubits)
        for i in range(numQubits):
            if i % 2 == 0:
                x(qubits[i])

        return mz(qubits)

    for numQubits in [1, 3, 5, 11, 20]:
        shots = 2
        results = cudaq.run(measure_all_qubits, numQubits, shots_count=shots)
        assert len(results) == shots
        for res in results:
            assert len(res) == numQubits
            for i in range(numQubits):
                if i % 2 == 0:
                    assert res[i] == True
                else:
                    assert res[i] == False


def test_return_tuple_int_float():

    @cudaq.kernel
    def simple_tuple_int_float_no_args() -> tuple[int, float]:
        return (-13, 42.3)

    result = cudaq.run(simple_tuple_int_float_no_args, shots_count=1)
    assert len(result) == 1 and result[0] == (-13, 42.3)

    @cudaq.kernel
    def simple_tuple_int_float(n: int, t: tuple[int,
                                                float]) -> tuple[int, float]:
        qubits = cudaq.qvector(n)
        return t

    result = cudaq.run(simple_tuple_int_float, 2, (-13, 42.3), shots_count=1)
    assert len(result) == 1 and result[0] == (-13, 42.3)

    @cudaq.kernel
    def simple_tuple_int_float_assign(
            n: int, t: tuple[int, float]) -> tuple[int, float]:
        qubits = cudaq.qvector(n)
        t[0] = -14
        t[1] = 11.5
        return t

    with pytest.raises(RuntimeError) as e:
        cudaq.run(simple_tuple_int_float_assign, 2, (-13, 11.5))
    assert 'tuple value cannot be modified' in str(e.value)

    @cudaq.kernel
    def simple_tuple_int_float_conversion(
            n: int, t: tuple[int, float]) -> tuple[bool, float]:
        qubits = cudaq.qvector(n)
        return t

    result = cudaq.run(simple_tuple_int_float_conversion,
                       2, (-13, 42.3),
                       shots_count=1)
    assert len(result) == 1 and result[0] == (True, 42.3)


def test_return_tuple_float_int():

    @cudaq.kernel
    def simple_tuple_float_int_no_args() -> tuple[float, int]:
        return (42.3, 13)

    result = cudaq.run(simple_tuple_float_int_no_args, shots_count=1)
    assert len(result) == 1 and result[0] == (42.3, 13)

    @cudaq.kernel
    def simple_tuple_float_int(n: int, t: tuple[float,
                                                int]) -> tuple[float, int]:
        qubits = cudaq.qvector(n)
        return t

    result = cudaq.run(simple_tuple_float_int, 2, (42.3, 13), shots_count=1)
    assert len(result) == 1 and result[0] == (42.3, 13)


def test_return_tuple_bool_int():

    @cudaq.kernel
    def simple_tuple_bool_int_no_args() -> tuple[bool, int]:
        return (True, 13)

    result = cudaq.run(simple_tuple_bool_int_no_args, shots_count=1)
    assert len(result) == 1 and result[0] == (True, 13)

    @cudaq.kernel
    def simple_tuple_bool_int(n: int, t: tuple[bool, int]) -> tuple[bool, int]:
        qubits = cudaq.qvector(n)
        return t

    result = cudaq.run(simple_tuple_bool_int, 2, (True, 13), shots_count=1)
    assert len(result) == 1 and result[0] == (True, 13)


def test_return_tuple_int_bool():

    @cudaq.kernel
    def simple_tuple_int_bool_no_args() -> tuple[int, bool]:
        return (-13, True)

    result = cudaq.run(simple_tuple_int_bool_no_args, shots_count=1)
    assert len(result) == 1 and result[0] == (-13, True)

    @cudaq.kernel
    def simple_tuple_int_bool(n: int, t: tuple[int, bool]) -> tuple[int, bool]:
        qubits = cudaq.qvector(n)
        return t

    result = cudaq.run(simple_tuple_int_bool, 2, (-13, True), shots_count=1)
    assert len(result) == 1 and result[0] == (-13, True)


def test_return_tuple_int32_bool():

    @cudaq.kernel
    def simple_tuple_int32_bool_no_args() -> tuple[np.int32, bool]:
        return (-13, True)

    result = cudaq.run(simple_tuple_int32_bool_no_args, shots_count=1)
    assert len(result) == 1 and result[0] == (-13, True)

    @cudaq.kernel
    def simple_tuple_int32_bool_no_args1() -> tuple[np.int32, bool]:
        return (np.int32(-13), True)

    result = cudaq.run(simple_tuple_int32_bool_no_args1, shots_count=1)
    assert len(result) == 1 and result[0] == (-13, True)

    @cudaq.kernel
    def simple_tuple_int32_bool(
            n: int, t: tuple[np.int32, bool]) -> tuple[np.int32, bool]:
        qubits = cudaq.qvector(n)
        return t

    result = cudaq.run(simple_tuple_int32_bool, 2, (-13, True), shots_count=1)
    assert len(result) == 1 and result[0] == (-13, True)


def test_return_tuple_bool_int_float():

    @cudaq.kernel
    def simple_tuple_bool_int_float_no_args() -> tuple[bool, int, float]:
        return (True, 13, 42.3)

    result = cudaq.run(simple_tuple_bool_int_float_no_args, shots_count=1)
    assert len(result) == 1 and result[0] == (True, 13, 42.3)

    @cudaq.kernel
    def simple_tuple_bool_int_float(
            n: int, t: tuple[bool, int, float]) -> tuple[bool, int, float]:
        qubits = cudaq.qvector(n)
        return t

    result = cudaq.run(simple_tuple_bool_int_float,
                       2, (True, 13, 42.3),
                       shots_count=1)
    assert len(result) == 1 and result[0] == (True, 13, 42.3)


def test_return_dataclass_int_bool():

    @dataclass(slots=True)
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

    @dataclass(slots=True)
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

    @dataclass(slots=True)
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

    @dataclass(slots=True)
    class MyClass:
        x: list[int]
        y: bool

    def simple_return_dataclass(n: int, t: MyClass) -> MyClass:
        qubits = cudaq.qvector(n)
        return t

    # TODO: Support recursive aggregate types in kernels.
    # results = cudaq.run(simple_return_dataclass, 2, MyClass([0,1], 18), shots_count=2)
    # assert len(results) == 2
    # assert results[0] == MyClass([0,1], 18)
    # assert results[1] == MyClass([0,1], 18)


def test_return_dataclass_tuple_bool():

    @dataclass(slots=True)
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

    @dataclass(slots=True)
    class MyClass1:
        x: int
        y: bool

    @dataclass(slots=True)
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

    @dataclass(slots=True)
    class MyClass:
        x: int
        y: bool

    @cudaq.kernel
    def simple_struc_err(t: MyClass) -> MyClass:
        q = cudaq.qubit()
        # If we allowed this, the expected behavior for Python
        # would be that t is modified also in the caller without
        # having to return it. We hence give an error to make it
        # clear that changes to structs don't propagate past
        # function boundaries.
        t.x = 42
        return t

    with pytest.raises(RuntimeError) as e:
        cudaq.run(simple_struc_err, MyClass(-13, True), shots_count=2)
    assert 'value cannot be modified - use `.copy(deep)` to create a new value that can be modified' in repr(
        e)
    assert '(offending source -> t.x)' in repr(e)

    @cudaq.kernel
    def simple_structA(arg: MyClass) -> MyClass:
        q = cudaq.qubit()
        t = arg.copy()
        t.x = 42
        return t

    results = cudaq.run(simple_structA, MyClass(-13, True), shots_count=2)
    print(results)
    assert len(results) == 2
    assert results[0] == MyClass(42, True)
    assert results[1] == MyClass(42, True)

    @dataclass(slots=True)
    class Foo:
        x: bool
        y: float
        z: int

    @cudaq.kernel
    def kernelB(arg: Foo) -> Foo:
        q = cudaq.qubit()
        t = arg.copy()
        t.z = 100
        t.y = 3.14
        t.x = True
        return t

    results = cudaq.run(kernelB, Foo(False, 6.28, 17), shots_count=2)
    print(results)
    assert len(results) == 2
    assert results[0] == Foo(True, 3.14, 100)
    assert results[1] == Foo(True, 3.14, 100)


def test_create_and_modify_struct():

    @dataclass(slots=True)
    class MyClass:
        x: int
        y: bool

    @cudaq.kernel
    def simple_strucC() -> MyClass:
        q = cudaq.qubit()
        t = MyClass(-13, True)
        t.x = 42
        return t

    results = cudaq.run(simple_strucC, shots_count=2)
    print(results)
    assert len(results) == 2
    assert results[0] == MyClass(42, True)
    assert results[1] == MyClass(42, True)

    @dataclass(slots=True)
    class Bar:
        x: bool
        y: bool
        z: float

    @cudaq.kernel
    def kerneD(n: int) -> Bar:
        q = cudaq.qvector(n)
        t = Bar(False, False, 4.14)
        t.x = True
        t.y = True
        return t

    results = cudaq.run(kerneD, 2, shots_count=1)
    print(results)
    assert len(results) == 1
    assert results[0] == Bar(True, True, 4.14)


def test_unsupported_return_type():

    @cudaq.kernel
    def kernel_with_no_args() -> complex:
        return 1 + 2j

    with pytest.raises(RuntimeError) as e:
        cudaq.run(kernel_with_no_args, shots_count=2)
    assert 'unsupported return type' in str(e.value)

    @cudaq.kernel
    def kernel_with_args(real: float, imag: float) -> complex:
        return complex(real, imag)

    with pytest.raises(RuntimeError) as e:
        cudaq.run(kernel_with_args, 1.0, 2.0, shots_count=2)
    assert 'unsupported return type' in str(e.value)


def test_run_and_sample_and_direct_call():

    @cudaq.kernel
    def bell_pair() -> int:
        q = cudaq.qvector(2)
        h(q[0])
        cx(q[0], q[1])
        res = mz(q[0]) + 2 * mz(q[1])
        return res

    run_results = cudaq.run(bell_pair, shots_count=10)
    assert len(run_results) == 10

    direct_call_result = bell_pair()
    assert direct_call_result is not None

    with pytest.raises(RuntimeError) as error:
        cudaq.sample_async(bell_pair, shots_count=10)
    assert (
        "The `sample_async` API only supports kernels that return None (void)"
        in repr(error))


# NOTE: Ref - https://github.com/NVIDIA/cuda-quantum/issues/1925
@pytest.mark.parametrize("target",
                         ["density-matrix-cpu", "nvidia", "qpp-cpu", "stim"])
def test_supported_simulators(target):

    def can_set_target(name):
        target_installed = True
        try:
            cudaq.set_target(name)
        except RuntimeError:
            target_installed = False
        return target_installed

    if can_set_target(target):
        test_simple_run_ghz()
    else:
        pytest.skip("target not available")

    cudaq.reset_target()


def test_unsupported_targets_0():
    try:
        cudaq.set_target("dynamics")
        with pytest.raises(RuntimeError) as e:
            test_simple_run_ghz()
        assert "Quantum gate simulation is not supported" in repr(e)
    except RuntimeError:
        pytest.skip("target not available")
    finally:
        cudaq.reset_target()

    try:
        cudaq.set_target("orca")
        with pytest.raises(RuntimeError) as e:
            test_simple_run_ghz()
        assert "No QPUs are available for this target" in repr(e)
    except RuntimeError:
        pytest.skip("target not available")
    finally:
        cudaq.reset_target()


@pytest.mark.parametrize("target, env_var",
                         [("anyon", ""), ("infleqtion", "SUPERSTAQ_API_KEY"),
                          ("ionq", "IONQ_API_KEY"), ("quantinuum", "")])
@pytest.mark.parametrize("emulate", [True, False])
def test_unsupported_targets_1(target, env_var, emulate):
    if env_var:
        os.environ[env_var] = "foobar"

    if target == 'quantinuum' and not emulate:
        pytest.skip("This target needs additional setup.")
    else:
        cudaq.set_target(target, emulate=emulate)

    with pytest.raises(RuntimeError) as e:
        test_simple_run_ghz()
    assert "not yet supported on this target" in repr(e)
    os.environ.pop(env_var, None)
    cudaq.reset_target()


@skipIfBraketNotInstalled
@pytest.mark.parametrize("target", ["braket", "quera"])
def test_unsupported_targets_2(target):
    cudaq.set_target(target)
    with pytest.raises(RuntimeError) as e:
        test_simple_run_ghz()
    assert "not yet supported on this target" in repr(e)
    cudaq.reset_target()


def test_dataclass_slots_success():

    @dataclass(slots=True)
    class SlotsClass:
        x: int
        y: int

    @cudaq.kernel
    def kernel_with_slots_dataclass() -> SlotsClass:
        return SlotsClass(3, 4)

    results = cudaq.run(kernel_with_slots_dataclass, shots_count=2)
    assert len(results) == 2
    assert all(isinstance(result, SlotsClass) for result in results)
    assert results == [SlotsClass(3, 4), SlotsClass(3, 4)]


def test_dataclasses_dot_dataclass_slots_success():
    import dataclasses

    @dataclasses.dataclass(slots=True)
    class SlotsClass:
        x: int
        y: int

    @cudaq.kernel
    def kernel_with_slots_dataclass() -> SlotsClass:
        return SlotsClass(3, 4)

    results = cudaq.run(kernel_with_slots_dataclass, shots_count=2)
    assert len(results) == 2
    assert all(isinstance(result, SlotsClass) for result in results)
    assert results == [SlotsClass(3, 4), SlotsClass(3, 4)]


def test_dataclass_slots_success():

    @dataclass(slots=True)
    class SlotsClass:
        x: int
        y: int

    @cudaq.kernel
    def kernel_with_slots_dataclass() -> SlotsClass:
        return SlotsClass(3, 4)

    results = cudaq.run(kernel_with_slots_dataclass, shots_count=2)
    assert len(results) == 2
    assert all(isinstance(result, SlotsClass) for result in results)
    assert results == [SlotsClass(3, 4), SlotsClass(3, 4)]


def test_dataclasses_dot_dataclass_slots_success():
    import dataclasses

    @dataclasses.dataclass(slots=True)
    class SlotsClass:
        x: int
        y: int

    @cudaq.kernel
    def kernel_with_slots_dataclass() -> SlotsClass:
        return SlotsClass(3, 4)

    results = cudaq.run(kernel_with_slots_dataclass, shots_count=2)
    assert len(results) == 2
    assert all(isinstance(result, SlotsClass) for result in results)
    assert results == [SlotsClass(3, 4), SlotsClass(3, 4)]


def test_dataclass_user_defined_method_raises_error():

    @dataclass(slots=True)
    class SlotsClass:
        x: int
        y: int

        def doSomething(self):
            pass

    @cudaq.kernel
    def kernel_with_slots_dataclass() -> SlotsClass:
        return SlotsClass(3, 4)

    with pytest.raises(RuntimeError) as e:
        results = cudaq.run(kernel_with_slots_dataclass, shots_count=2)
    assert 'struct types with user specified methods are not allowed.' in str(
        e.value)


def test_dataclasses_dot_dataclass_user_defined_method_raises_error():
    import dataclasses

    @dataclasses.dataclass(slots=True)
    class SlotsClass:
        x: int
        y: int

        def doSomething(self):
            pass

    @cudaq.kernel
    def kernel_with_slots_dataclass() -> SlotsClass:
        return SlotsClass(3, 4)

    with pytest.raises(RuntimeError) as e:
        results = cudaq.run(kernel_with_slots_dataclass, shots_count=2)
    assert 'struct types with user specified methods are not allowed.' in str(
        e.value)


def test_shots_count():

    @cudaq.kernel
    def kernel() -> bool:
        q = cudaq.qubit()
        h(q)
        return mz(q)

    results = cudaq.run(kernel)
    assert len(results) == 100  # default shots count
    results = cudaq.run(kernel, shots_count=53)
    assert len(results) == 53


def test_return_from_if_loop_with_true_condition():

    @cudaq.kernel
    def kernel(cond: bool) -> int:
        if cond:
            return 1

    with pytest.raises(RuntimeError) as e:
        results = cudaq.run(kernel, True, shots_count=1)
    assert 'cudaq.kernel functions with return type annotations must have a return statement.' in str(
        e.value)


def test_return_from_if_loop_with_false_condition():

    @cudaq.kernel
    def kernel(cond: bool) -> int:
        if cond:
            return 1

    with pytest.raises(RuntimeError) as e:
        results = cudaq.run(kernel, False, shots_count=1)
    assert 'cudaq.kernel functions with return type annotations must have a return statement.' in str(
        e.value)


def test_return_from_if_loop_with_false_condition_and_return_from_parent_scope(
):

    @cudaq.kernel
    def kernel(cond: bool) -> int:
        if cond:
            return 1
        return 0

    results = cudaq.run(kernel, False, shots_count=1)
    assert len(results) == 1
    assert results[0] == 0


def test_return_from_if_and_else_loop_with_true_condition():

    @cudaq.kernel
    def kernel(cond: bool) -> int:
        if cond:
            return 1
        else:
            return -1

    results = cudaq.run(kernel, True, shots_count=1)
    assert len(results) == 1
    assert results[0] == 1


def test_return_from_if_and_else_loop_with_false_condition():

    @cudaq.kernel
    def kernel(cond: bool) -> int:
        if cond:
            return 1
        else:
            return -1

    results = cudaq.run(kernel, False, shots_count=1)
    assert len(results) == 1
    assert results[0] == -1


def test_return_from_if_and_else_loop_with_true_condition_in_for_loop():

    @cudaq.kernel
    def kernel(cond: bool) -> int:
        if cond:
            for i in range(6):
                if i == 0:
                    return 1
                else:
                    return -1
        else:
            return -1

    results = cudaq.run(kernel, True, shots_count=1)
    assert len(results) == 1
    assert results[0] == 1


def test_return_from_if_and_else_loop_having_for_with_no_return():

    @cudaq.kernel
    def kernel(cond: bool) -> int:
        if cond:
            for i in range(6):
                if i == 0:
                    return 1
        else:
            return -1

    with pytest.raises(RuntimeError) as e:
        results = cudaq.run(kernel, True, shots_count=1)
    assert 'cudaq.kernel functions with return type annotations must have a return statement.' in str(
        e.value)


def test_return_from_if_and_else_loop_with_for_loop_and_false_condition():

    @cudaq.kernel
    def kernel(cond: bool) -> int:
        if cond:
            for i in range(6):
                if i == 0:
                    return 1
                else:
                    return -1
        else:
            return -1

    results = cudaq.run(kernel, False, shots_count=1)
    assert len(results) == 1
    assert results[0] == -1


def test_return_from_if_and_else_loop_with_true_condition_in_while_loop():

    @cudaq.kernel
    def kernel(cond: bool) -> int:
        if cond:
            i = 0
            while i < 6:
                if i == 0:
                    return 1
                else:
                    return -1
                i = i + 1
        else:
            return -1

    results = cudaq.run(kernel, True, shots_count=1)
    assert len(results) == 1
    assert results[0] == 1


def test_return_from_if_and_else_loop_having_while_with_no_return():

    @cudaq.kernel
    def kernel(cond: bool) -> int:
        if cond:
            i = 0
            while i < 6:
                if i == 0:
                    return 1
                i = i + 1
        else:
            return -1

    with pytest.raises(RuntimeError) as e:
        results = cudaq.run(kernel, True, shots_count=1)
    assert 'cudaq.kernel functions with return type annotations must have a return statement.' in str(
        e.value)


def test_return_from_if_and_else_loop_with_for_loop_and_false_condition():

    @cudaq.kernel
    def kernel(cond: bool) -> int:
        if cond:
            for i in range(6):
                if i == 0:
                    return 1
                else:
                    return -1
        else:
            return -1

    results = cudaq.run(kernel, False, shots_count=1)
    assert len(results) == 1
    assert results[0] == -1


def test_return_from_for_loop_with_else_block():

    @cudaq.kernel
    def kernel() -> int:
        for i in range(6):
            if i % 2 == 0:
                return 1
        else:
            return -1

    results = cudaq.run(kernel, shots_count=1)
    assert len(results) == 1
    assert results[0] == 1


def test_return_from_else_block_after_a_for_loop():

    @cudaq.kernel
    def kernel() -> int:
        for i in range(6):
            if i % 2 == 10:
                return 1
        else:
            return -1

    results = cudaq.run(kernel, shots_count=1)
    assert len(results) == 1
    results[0] == -1


def test_return_from_while_loop_with_else_block():

    @cudaq.kernel
    def kernel() -> int:
        i = 0
        while i < 6:
            if i % 2 == 0:
                return 1
            i = i + 1
        else:
            return -1

    results = cudaq.run(kernel, shots_count=1)
    assert len(results) == 1
    assert results[0] == 1


def test_return_from_else_block_after_a_while_loop():

    @cudaq.kernel
    def kernel() -> int:
        i = 0
        while i < 6:
            if i % 2 == 10:
                return 1
            i = i + 1
        else:
            return -1

    results = cudaq.run(kernel, shots_count=1)
    assert len(results) == 1
    assert results[0] == -1


def test_return_from_outside_the_for_loop():

    @cudaq.kernel
    def kernel() -> int:
        for i in range(6):
            if i % 2 == 10:
                return 1
        return 0

    results = cudaq.run(kernel, shots_count=1)
    assert len(results) == 1
    assert results[0] == 0


def test_return_with_true_condition_with_variable_defined_outside_the_loop():

    @cudaq.kernel
    def kernel(cond: bool) -> int:
        result = 0
        if cond:
            result = 1
        return result

    results = cudaq.run(kernel, True, shots_count=1)
    assert len(results) == 1
    assert results[0] == 1


def test_return_with_false_condition_with_variable_defined_outside_the_loop():

    @cudaq.kernel
    def kernel(cond: bool) -> int:
        result = 0
        if cond:
            result = 1
        return result

    results = cudaq.run(kernel, False, shots_count=1)
    assert len(results) == 1
    assert results[0] == 0


def test_run_with_callable():
    '''
    Test running a kernel with a callable as a argument.
    '''

    @cudaq.kernel
    def kernel(state_prep: Callable[[cudaq.qvector], None], N: int) -> int:
        qubits = cudaq.qvector(N)
        state_prep(qubits)
        meas = mz(qubits)
        res = 0
        for m in meas:
            if m:
                res += 1
        return res

    @cudaq.kernel
    def prep_1_state(qubits: cudaq.qvector):
        x(qubits)

    for num_qubits in [1, 2, 3, 4]:
        results = cudaq.run(kernel, prep_1_state, num_qubits, shots_count=10)
        assert len(results) == 10
        for r in results:
            assert r == num_qubits


def test_return_nested_lists():
    """
    Test returning nested lists from a kernel. 
    This is currently unsupported and should raise an error.
    """

    @cudaq.kernel
    def nested_list_kernel() -> list[list[int]]:
        return [[1, 2], [3, 4]]

    with pytest.raises(RuntimeError) as e:
        results = cudaq.run(nested_list_kernel, shots_count=2)

    assert "`cudaq.run` does not yet support returning nested `list`" in str(
        e.value)


def test_return_list_of_structs():
    """
    Test returning a list of dataclass structs from a kernel. 
    This is currently unsupported and should raise an error.
    """

    @dataclass(slots=True)
    class SomeStruct:
        a: int
        b: bool

    @cudaq.kernel
    def list_of_structs_kernel() -> list[SomeStruct]:
        return [SomeStruct(1, True), SomeStruct(2, False)]

    with pytest.raises(RuntimeError) as e:
        results = cudaq.run(list_of_structs_kernel, shots_count=2)

    assert "`cudaq.run` does not yet support returning `list` of `dataclass`" in str(
        e.value)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
