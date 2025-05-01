# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os, time

import pytest
import numpy as np
from typing import Callable, List, Tuple
<<<<<<< HEAD
from dataclasses import dataclass
=======
>>>>>>> bf7a909fb5ad28b9c1ec211007d1e082e64a4fd0

import cudaq


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
        qubits = cudaq.qvector(2)
        return True

    # TODO: seg fault on running any kernel with no args
    # results = cudaq.run(simple_bool_no_args, shots_count=2)
    # assert len(results) == 2
    # assert results[0] == True
    # assert results[1] == True

    @cudaq.kernel
    def simple_bool(numQubits: int) -> bool:
        qubits = cudaq.qvector(numQubits)
        return True

    results = cudaq.run(simple_bool, 2, shots_count=2)
    assert len(results) == 2
    assert results[0] == True
    assert results[1] == True


def test_return_integral():

    @cudaq.kernel
    def simple_int(numQubits: int) -> int:
        qubits = cudaq.qvector(numQubits)
        return numQubits + 1

    results = cudaq.run(simple_int, 2, shots_count=2)
    assert len(results) == 2
    assert results[0] == 3
    assert results[1] == 3

    @cudaq.kernel
    def simple_int32(numQubits: int) -> np.int32:
        qubits = cudaq.qvector(numQubits)
        return numQubits + 1

    results = cudaq.run(simple_int32, 2, shots_count=2)
    assert len(results) == 2
    assert results[0] == 3
    assert results[1] == 3

    @cudaq.kernel
    def simple_int64(numQubits: int) -> np.int64:
        qubits = cudaq.qvector(numQubits)
        return numQubits + 1

    results = cudaq.run(simple_int64, 2, shots_count=2)
    assert len(results) == 2
    assert results[0] == 3
    assert results[1] == 3


def test_return_floating():

    @cudaq.kernel()
    def simple_float(numQubits: int) -> float:
        return numQubits + 1

    results = cudaq.run(simple_float, 2, shots_count=2)
    assert len(results) == 2
    assert results[0] == 3.0
    assert results[1] == 3.0

    @cudaq.kernel
    def simple_float32(numQubits: int) -> np.float32:
        return numQubits + 1

    results = cudaq.run(simple_float32, 2, shots_count=2)
    assert len(results) == 2
    assert results[0] == 3.0
    assert results[1] == 3.0

    @cudaq.kernel
    def simple_float64(numQubits: int) -> np.float64:
        return numQubits + 1

    results = cudaq.run(simple_float64, 2, shots_count=2)
    assert len(results) == 2
    assert results[0] == 3.0
    assert results[1] == 3.0


def test_return_list():

    @cudaq.kernel
    def simple_list_bool(n: int) -> list[bool]:
        qubits = cudaq.qvector(n)
        result = [True, False]
        return result

    results = cudaq.run(simple_list_bool, 2, shots_count=2)
    assert len(results) == 2
    assert results[0] == [True, False]
    assert results[1] == [True, False]

    @cudaq.kernel
    def simple_list_int(n: int) -> list[int]:
        qubits = cudaq.qvector(n)
        result = [1, 0]
        return result

    results = cudaq.run(simple_list_int, 2, shots_count=2)
    assert len(results) == 2
    assert results[0] == [1, 0]
    assert results[1] == [1, 0]

    @cudaq.kernel
    def simple_list_int32(n: int) -> list[np.int32]:
        qubits = cudaq.qvector(n)
        result = [1, 0]
        return result

    results = cudaq.run(simple_list_int32, 2, shots_count=2)
    assert len(results) == 2
    assert results[0] == [1, 0]
    assert results[1] == [1, 0]

    @cudaq.kernel
    def simple_list_int64(n: int) -> list[np.int64]:
        qubits = cudaq.qvector(n)
        result = [1, 0]
        return result

    results = cudaq.run(simple_list_int64, 2, shots_count=2)
    assert len(results) == 2
    assert results[0] == [1, 0]
    assert results[1] == [1, 0]

    @cudaq.kernel
    def simple_list_float(n: int) -> list[float]:
        qubits = cudaq.qvector(n)
        result = [1.0, 0.0]
        return result

    results = cudaq.run(simple_list_float, 2, shots_count=2)
    assert len(results) == 2
    assert results[0] == [1.0, 0.0]
    assert results[1] == [1.0, 0.0]

    @cudaq.kernel
    def simple_list_float32(n: int) -> list[np.float32]:
        qubits = cudaq.qvector(n)
        result = [1.0, 0.0]
        return result

    results = cudaq.run(simple_list_float32, 2, shots_count=2)
    assert len(results) == 2
    assert results[0] == [1.0, 0.0]
    assert results[1] == [1.0, 0.0]

    @cudaq.kernel
    def simple_list_float64(n: int) -> list[np.float64]:
        qubits = cudaq.qvector(n)
        result = [1.0, 0.0]
        return result

    results = cudaq.run(simple_list_float64, 2, shots_count=2)
    assert len(results) == 2
    assert results[0] == [1.0, 0.0]
    assert results[1] == [1.0, 0.0]


def test_return_tuple():

    @cudaq.kernel
    def simple_tuple_int_float(n: int, t: tuple[int,
                                                float]) -> tuple[int, float]:
        qubits = cudaq.qvector(n)
        return t

    results = cudaq.run(simple_tuple_int_float, 2, (13, 42.3), shots_count=2)
    assert len(results) == 2
    assert results[0] == (13, 42.3)
    assert results[1] == (13, 42.3)

    @cudaq.kernel
    def simple_tuple_float_int(n: int, t: tuple[float,
                                                int]) -> tuple[float, int]:
        qubits = cudaq.qvector(n)
        return t

    results = cudaq.run(simple_tuple_float_int, 2, (42.3, 13), shots_count=2)
    assert len(results) == 2
    assert results[0] == (42.3, 13)
    assert results[1] == (42.3, 13)

    @cudaq.kernel
    def simple_tuple_bool_int(n: int, t: tuple[bool, int]) -> tuple[bool, int]:
        qubits = cudaq.qvector(n)
        return t

    # TODO: fix alignment
    results = cudaq.run(simple_tuple_bool_int, 2, (True, 13), shots_count=2)
    assert len(results) == 2
    #assert results[0] == (True, 13)
    #assert results[1] == (True, 13)

    @cudaq.kernel
    def simple_tuple_int_bool(n: int, t: tuple[int, bool]) -> tuple[int, bool]:
        qubits = cudaq.qvector(n)
        return t

    # TODO: fix alignment
    results = cudaq.run(simple_tuple_int_bool, 2, (13, True), shots_count=2)
    assert len(results) == 2
    # assert results[0] == (13, True)
    # assert results[1] == (13, True)

    @cudaq.kernel
    def simple_tuple_bool_int_float(
            n: int, t: tuple[bool, int, float]) -> tuple[bool, int, float]:
        qubits = cudaq.qvector(n)
        return t

    # TODO: fix alignment
    results = cudaq.run(simple_tuple_bool_int_float,
                        2, (True, 13, 42.3),
                        shots_count=2)
    assert len(results) == 2
    #assert results[0] == (True, 13, 42.3)
    #assert results[1] == (True, 13, 42.3)


def test_return_dataclass_int_bool():

    @dataclass
    class MyClass:
        x: int
        y: bool

    @cudaq.kernel
    def test_return_dataclass(n: int, t: MyClass) -> MyClass:
        qubits = cudaq.qvector(n)
        return t

    results = cudaq.run(test_return_dataclass,
                        2,
                        MyClass(16, True),
                        shots_count=2)
    assert len(results) == 2
    # TODO: fix alignment
    # assert results[0] == MyClass(16, True)
    # assert results[1] == MyClass(16, True)


def test_return_dataclass_bool_int():

    @dataclass
    class MyClass:
        x: bool
        y: int

    @cudaq.kernel
    def test_return_dataclass(n: int, t: MyClass) -> MyClass:
        qubits = cudaq.qvector(n)
        return t

    results = cudaq.run(test_return_dataclass,
                        2,
                        MyClass(True, 17),
                        shots_count=2)
    assert len(results) == 2
    # TODO: fix alignment
    # assert results[0] == MyClass(True, 17)
    # assert results[1] == MyClass(True, 17)


def test_return_dataclass_float_int():

    @dataclass
    class MyClass:
        x: float
        y: int

    @cudaq.kernel
    def test_return_dataclass(n: int, t: MyClass) -> MyClass:
        qubits = cudaq.qvector(n)
        return t

    results = cudaq.run(test_return_dataclass,
                        2,
                        MyClass(42.5, 17),
                        shots_count=2)
    assert len(results) == 2
    assert results[0] == MyClass(42.5, 17)
    assert results[1] == MyClass(42.5, 17)


def test_return_dataclass_list_int_bool():

    @dataclass
    class MyClass:
        x: list[int]
        y: bool

    @cudaq.kernel
    def test_return_dataclass(n: int, t: MyClass) -> MyClass:
        qubits = cudaq.qvector(n)
        return t

    # TODO: RuntimeError: Tuple size mismatch in value and label
    # results = cudaq.run(test_return_dataclass, 2, MyClass([0,1], 18), shots_count=2)
    # assert len(results) == 2
    # assert results[0] == MyClass([0,1], 18)
    # assert results[1] == MyClass([0,1], 18)


def test_return_dataclass_tuple_bool():

    @dataclass
    class MyClass:
        x: tuple[int, bool]
        y: bool

    @cudaq.kernel
    def test_return_dataclass(n: int, t: MyClass) -> MyClass:
        qubits = cudaq.qvector(n)
        return t

    # TODO: error: recursive struct types are not allowed in kernels.
    # results = cudaq.run(test_return_dataclass, 2, MyClass((0, True), 19), shots_count=2)
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
    def test_return_dataclass(n: int, t: MyClass2) -> MyClass2:
        qubits = cudaq.qvector(n)
        return t

    # TODO: error: recursive struct types are not allowed in kernels.
    # results = cudaq.run(test_return_dataclass, 2, MyClass2(MyClass1(0,True), 20), shots_count=2)
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
    assert 'cudaq.run only supports kernels that return a value.' in repr(e)

    with pytest.raises(TypeError) as e:
        cudaq.run(simple, 2, shots_count=-1)
    assert 'incompatible function arguments.' in repr(e)

    with pytest.raises(RuntimeError) as e:
        cudaq.run(simple, shots_count=100)
    assert 'Invalid number of arguments passed to run:0 expected 1' in repr(e)

    with pytest.raises(RuntimeError) as e:
        print(cudaq.run(simple_no_args, 2, shots_count=100))
    assert 'Invalid number of arguments passed to run:1 expected 0' in repr(e)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
