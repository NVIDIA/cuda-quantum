# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os
from dataclasses import dataclass

import cudaq
import numpy as np
import pytest


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


def test_return_bool():

    @cudaq.kernel
    def simple_bool_no_args() -> bool:
        return True

    result = simple_bool_no_args()
    assert result == True

    @cudaq.kernel
    def simple_bool(numQubits: int) -> bool:
        qubits = cudaq.qvector(numQubits)
        return True

    result = simple_bool(2)
    assert result == True


def test_return_int():

    @cudaq.kernel
    def simple_int_no_args() -> int:
        return -43

    result = simple_int_no_args()
    assert result == -43

    @cudaq.kernel
    def simple_int(numQubits: int) -> int:
        qubits = cudaq.qvector(numQubits)
        return numQubits + 1

    result = simple_int(2)
    assert result == 3


def test_return_int8():

    @cudaq.kernel
    def simple_int8_no_args() -> np.int8:
        return -43

    result = simple_int8_no_args()
    assert result == -43

    @cudaq.kernel
    def simple_int8(numQubits: int) -> np.int8:
        qubits = cudaq.qvector(numQubits)
        return numQubits + 1

    result = simple_int8(2)
    assert result == 3


def test_return_int16():

    @cudaq.kernel
    def simple_int16_no_args() -> np.int16:
        return -43

    result = simple_int16_no_args()
    assert result == -43

    @cudaq.kernel
    def simple_int16(numQubits: int) -> np.int16:
        qubits = cudaq.qvector(numQubits)
        return numQubits + 1

    result = simple_int16(2)
    assert result == 3


def test_return_int32():

    @cudaq.kernel
    def simple_int32_no_args() -> np.int32:
        return -43

    result = simple_int32_no_args()
    assert result == -43

    @cudaq.kernel
    def simple_int32(numQubits: int) -> np.int32:
        qubits = cudaq.qvector(numQubits)
        return numQubits + 1

    result = simple_int32(2)
    assert result == 3


def test_return_int64():

    @cudaq.kernel
    def simple_int64_no_args() -> np.int64:
        return -43

    result = simple_int64_no_args()
    assert result == -43

    @cudaq.kernel
    def simple_int64(numQubits: int) -> np.int64:
        qubits = cudaq.qvector(numQubits)
        return numQubits + 1

    result = simple_int64(2)
    assert result == 3


def test_return_float():

    @cudaq.kernel
    def simple_float_no_args() -> float:
        return -43.2

    result = simple_float_no_args()
    assert result == -43.2

    @cudaq.kernel()
    def simple_float(numQubits: int) -> float:
        return numQubits + 1

    result = simple_float(2)
    assert result == 3.0


def test_return_float32():

    @cudaq.kernel
    def simple_float32_no_args() -> np.float32:
        return -43.2

    result = simple_float32_no_args()
    assert is_close(result, -43.2)

    @cudaq.kernel
    def simple_float32(numQubits: int) -> np.float32:
        return numQubits + 1

    result = simple_float32(2)
    assert result == 3.0


def test_return_float64():

    @cudaq.kernel
    def simple_float64_no_args() -> np.float64:
        return -43.2

    result = simple_float64_no_args()
    assert result == -43.2

    @cudaq.kernel
    def simple_float64(numQubits: int) -> np.float64:
        return numQubits + 1

    result = simple_float64(2)
    assert result == 3.0


def test_return_list_bool():

    @cudaq.kernel
    def simple_list_bool_no_args() -> list[bool]:
        return [True, False, True]

    result = simple_list_bool_no_args()
    assert result == [True, False, True]

    @cudaq.kernel
    def simple_list_bool(n: int, t: list[bool]) -> list[bool]:
        qubits = cudaq.qvector(n)
        return t.copy()

    result = simple_list_bool(2, [True, False, True])
    assert result == [True, False, True]


def test_return_list_int():

    @cudaq.kernel
    def simple_list_int_no_args() -> list[int]:
        return [-13, 5, 42]

    result = simple_list_int_no_args()
    assert result == [-13, 5, 42]

    @cudaq.kernel
    def simple_list_int(n: int, t: list[int]) -> list[int]:
        qubits = cudaq.qvector(n)
        return t.copy()

    result = simple_list_int(2, [-13, 5, 42])
    assert result == [-13, 5, 42]


def test_return_list_int32():

    @cudaq.kernel
    def simple_list_int32_no_args() -> list[np.int32]:
        return [-13, 5, 42]

    result = simple_list_int32_no_args()
    assert result == [-13, 5, 42]

    @cudaq.kernel
    def simple_list_int32(n: int, t: list[np.int32]) -> list[np.int32]:
        qubits = cudaq.qvector(n)
        return t.copy()

    result = simple_list_int32(2, [-13, 5, 42])
    assert result == [-13, 5, 42]


def test_return_list_int16():

    @cudaq.kernel
    def simple_list_int16_no_args() -> list[np.int16]:
        return [-13, 5, 42]

    result = simple_list_int16_no_args()
    assert result == [-13, 5, 42]

    @cudaq.kernel
    def simple_list_int16(n: int, t: list[np.int16]) -> list[np.int16]:
        qubits = cudaq.qvector(n)
        return t.copy()

    result = simple_list_int16(2, [-13, 5, 42])
    assert result == [-13, 5, 42]


def test_return_list_int8():

    @cudaq.kernel
    def simple_list_int8_no_args() -> list[np.int8]:
        return [-13, 5, 42]

    result = simple_list_int8_no_args()
    assert result == [-13, 5, 42]

    @cudaq.kernel
    def simple_list_int8(n: int, t: list[np.int8]) -> list[np.int8]:
        qubits = cudaq.qvector(n)
        return t.copy()

    result = simple_list_int8(2, [-13, 5, 42])
    assert result == [-13, 5, 42]


def test_return_list_int64():

    @cudaq.kernel
    def simple_list_int64_no_args() -> list[np.int64]:
        return [-13, 5, 42]

    result = simple_list_int64_no_args()
    assert result == [-13, 5, 42]

    @cudaq.kernel
    def simple_list_int64(n: int, t: list[np.int64]) -> list[np.int64]:
        qubits = cudaq.qvector(n)
        return t.copy()

    result = simple_list_int64(2, [-13, 5, 42])
    assert result == [-13, 5, 42]


def test_return_list_float():

    @cudaq.kernel
    def simple_list_float_no_args() -> list[float]:
        return [-13.2, 5., 42.99]

    result = simple_list_float_no_args()
    assert result == [-13.2, 5.0, 42.99]

    @cudaq.kernel
    def simple_list_float(n: int, t: list[float]) -> list[float]:
        qubits = cudaq.qvector(n)
        return t.copy()

    result = simple_list_float(2, [-13.2, 5.0, 42.99])
    assert result == [-13.2, 5.0, 42.99]


def test_return_list_float32():

    @cudaq.kernel
    def simple_list_float32_no_args() -> list[np.float32]:
        return [-13.2, 5., 42.99]

    result = simple_list_float32_no_args()
    print(result)
    assert is_close_array(result, [-13.2, 5.0, 42.99])

    @cudaq.kernel
    def simple_list_float32(n: int, t: list[np.float32]) -> list[np.float32]:
        qubits = cudaq.qvector(n)
        return t.copy()

    result = simple_list_float32(2, [-13.2, 5.0, 42.99])
    assert is_close_array(result, [-13.2, 5.0, 42.99])


def test_return_list_float64():

    @cudaq.kernel
    def simple_list_float64_no_args() -> list[np.float64]:
        return [-13.2, 5., 42.99]

    result = simple_list_float64_no_args()
    assert result == [-13.2, 5.0, 42.99]

    @cudaq.kernel
    def simple_list_float64(n: int, t: list[np.float64]) -> list[np.float64]:
        qubits = cudaq.qvector(n)
        return t.copy()

    result = simple_list_float64(2, [-13.2, 5.0, 42.99])
    assert result == [-13.2, 5.0, 42.99]


def test_return_tuple_int_float():

    @cudaq.kernel
    def simple_tuple_int_float_no_args() -> tuple[int, float]:
        return (-13, 42.3)

    result = simple_tuple_int_float_no_args()
    assert result == (-13, 42.3)

    @cudaq.kernel
    def simple_tuple_int_float(n: int, t: tuple[int,
                                                float]) -> tuple[int, float]:
        qubits = cudaq.qvector(n)
        return t

    result = simple_tuple_int_float(2, (-13, 42.3))
    assert result == (-13, 42.3)

    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel
        def simple_tuple_int_float_assign(
                n: int, t: tuple[int, float]) -> tuple[int, float]:
            qubits = cudaq.qvector(n)
            t[0] = -14
            t[1] = 11.5
            return t

        simple_tuple_int_float_assign(2, (-13, 42.3))
    assert 'tuple value cannot be modified' in str(e.value)


def test_return_tuple_float_int():

    @cudaq.kernel
    def simple_tuple_float_int_no_args() -> tuple[float, int]:
        return (42.3, 13)

    result = simple_tuple_float_int_no_args()
    assert result == (42.3, 13)

    @cudaq.kernel
    def simple_tuple_float_int(n: int, t: tuple[float,
                                                int]) -> tuple[float, int]:
        qubits = cudaq.qvector(n)
        return t

    result = simple_tuple_float_int(2, (42.3, 13))
    assert result == (42.3, 13)


def test_return_tuple_bool_int():

    @cudaq.kernel
    def simple_tuple_bool_int_no_args() -> tuple[bool, int]:
        return (True, 13)

    result = simple_tuple_bool_int_no_args()
    assert result == (True, 13)

    @cudaq.kernel
    def simple_tuple_bool_int(n: int, t: tuple[bool, int]) -> tuple[bool, int]:
        qubits = cudaq.qvector(n)
        return t

    result = simple_tuple_bool_int(2, (True, 13))
    assert result == (True, 13)


def test_return_tuple_int_bool():

    @cudaq.kernel
    def simple_tuple_int_bool_no_args() -> tuple[int, bool]:
        return (-13, True)

    result = simple_tuple_int_bool_no_args()
    assert result == (-13, True)

    @cudaq.kernel
    def simple_tuple_int_bool(n: int, t: tuple[int, bool]) -> tuple[int, bool]:
        qubits = cudaq.qvector(n)
        return t

    result = simple_tuple_int_bool(2, (-13, True))
    assert result == (-13, True)


def test_return_tuple_int32_bool():

    @cudaq.kernel
    def simple_tuple_int32_bool_no_args() -> tuple[np.int32, bool]:
        return (-13, True)

    result = simple_tuple_int32_bool_no_args()
    # See https://github.com/NVIDIA/cuda-quantum/issues/3524
    assert result == (-13, True)

    @cudaq.kernel
    def simple_tuple_int32_bool_no_args1() -> tuple[np.int32, bool]:
        return (np.int32(-13), True)

    result = simple_tuple_int32_bool_no_args1()
    # Note: printing the kernel correctly shows the MLIR
    # values return type as "tuple" {i32, i1}, but we don't
    # actually create numpy values even when these are requested
    # in the signature.
    # See https://github.com/NVIDIA/cuda-quantum/issues/3524
    assert result == (-13, True)

    @cudaq.kernel
    def simple_tuple_int32_bool(
            n: int, t: tuple[np.int32, bool]) -> tuple[np.int32, bool]:
        qubits = cudaq.qvector(n)
        return t

    result = simple_tuple_int32_bool(2, (np.int32(-13), True))
    # See https://github.com/NVIDIA/cuda-quantum/issues/3524
    assert result == (-13, True)


def test_return_tuple_bool_int_float():

    @cudaq.kernel
    def simple_tuple_bool_int_float_no_args() -> tuple[bool, int, float]:
        return (True, 13, 42.3)

    result = simple_tuple_bool_int_float_no_args()
    assert result == (True, 13, 42.3)

    @cudaq.kernel
    def simple_tuple_bool_int_float(
            n: int, t: tuple[bool, int, float]) -> tuple[bool, int, float]:
        qubits = cudaq.qvector(n)
        return t

    result = simple_tuple_bool_int_float(2, (True, 13, 42.3))
    assert result == (True, 13, 42.3)


def test_return_dataclass_int_bool():

    @dataclass(slots=True)
    class MyClass:
        x: int
        y: bool

    @cudaq.kernel
    def simple_dataclass_int_bool_no_args() -> MyClass:
        return MyClass(-16, True)

    result = simple_dataclass_int_bool_no_args()
    assert result == MyClass(-16, True)
    assert result.x == -16
    assert result.y == True

    @cudaq.kernel
    def test_return_dataclass(n: int, t: MyClass) -> MyClass:
        qubits = cudaq.qvector(n)
        return t

    result = test_return_dataclass(2, MyClass(-16, True))
    assert result == MyClass(-16, True)
    assert result.x == -16
    assert result.y == True


def test_return_dataclass_bool_int():

    @dataclass(slots=True)
    class MyClass2:
        x: bool
        y: int

    @cudaq.kernel
    def simple_dataclass_bool_int_no_args() -> MyClass2:
        return MyClass2(True, 17)

    result = simple_dataclass_bool_int_no_args()
    assert result == MyClass2(True, 17)
    assert result.x == True
    assert result.y == 17

    @cudaq.kernel
    def test_return_dataclass(n: int, t: MyClass2) -> MyClass2:
        qubits = cudaq.qvector(n)
        return t

    result = test_return_dataclass(2, MyClass2(True, 17))
    assert result == MyClass2(True, 17)
    assert result.x == True
    assert result.y == 17


def test_return_dataclass_float_int():

    @dataclass(slots=True)
    class MyClass:
        x: float
        y: int

    @cudaq.kernel
    def simple_dataclass_float_int_no_args() -> MyClass:
        return MyClass(42.5, 17)

    result = simple_dataclass_float_int_no_args()
    assert result == MyClass(42.5, 17)
    assert result.x == 42.5
    assert result.y == 17

    @cudaq.kernel
    def test_return_dataclass(n: int, t: MyClass) -> MyClass:
        qubits = cudaq.qvector(n)
        return t

    result = test_return_dataclass(2, MyClass(42.5, 17))

    assert result == MyClass(42.5, 17)
    assert result.x == 42.5
    assert result.y == 17


def test_return_dataclass_list_int_bool():

    @dataclass(slots=True)
    class MyClass:
        x: list[int]
        y: bool

    @cudaq.kernel
    def test_return_dataclass(n: int, t: MyClass) -> MyClass:
        qubits = cudaq.qvector(n)
        return t.copy(deep=True)

    # TODO: Support recursive aggregate types in kernels.
    # result = test_return_dataclass(2, MyClass([0,1], 18))
    #
    # assert result == MyClass([0,1], 18)


def test_return_dataclass_tuple_bool():

    @dataclass(slots=True)
    class MyClass:
        x: tuple[int, bool]
        y: bool

    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel
        def test_return_dataclass(n: int, t: MyClass) -> MyClass:
            qubits = cudaq.qvector(n)
            return t

        result = test_return_dataclass(2, MyClass((0, True), 19))
    assert 'Type not supported' in str(e.value)
    #assert result == MyClass((0, True), 19)


def test_return_dataclass_dataclass_bool():

    @dataclass(slots=True)
    class MyClass1:
        x: int
        y: bool

    @dataclass(slots=True)
    class MyClass2:
        x: MyClass1
        y: bool

    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel
        def test_return_dataclass(n: int, t: MyClass2) -> MyClass2:
            qubits = cudaq.qvector(n)
            return t

        result = test_return_dataclass(2, MyClass2(MyClass1(0, True), 20))
    # FIXME!!!
    # The bridge is incorrectly deciding there are recursive types here!
    assert 'Type not supported' in str(e.value)


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
    assert 'a runnable kernel must return a value.' in repr(e)

    with pytest.raises(RuntimeError) as e:
        cudaq.run(simple, 2, shots_count=-1)
    assert 'Invalid `shots_count`' in repr(e)

    with pytest.raises(RuntimeError) as e:
        cudaq.run(simple, shots_count=100)
    assert 'Invalid number of arguments passed to run' in repr(e)

    with pytest.raises(RuntimeError) as e:
        print(cudaq.run(simple_no_args, 2, shots_count=100))
    assert 'Invalid number of arguments passed to run' in repr(e)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
