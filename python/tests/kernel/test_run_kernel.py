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
from typing import Callable, List

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


def test_return_noargs():

    @cudaq.kernel()
    def simple() -> bool:
        qubits = cudaq.qvector(2)
        return True

    results = cudaq.run(simple, shots_count=0)
    assert len(results) == 0


def test_return_integral():

    @cudaq.kernel()
    def simple_bool(numQubits: int) -> bool:
        qubits = cudaq.qvector(numQubits)
        return True

    results = cudaq.run(simple_bool, 2, shots_count=2)
    assert len(results) == 2
    assert results[0] == True
    assert results[1] == True

    @cudaq.kernel()
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
    print(results)
    assert len(results) == 2
    assert results[0] == 3
    assert results[1] == 3

    @cudaq.kernel
    def simple_int64(numQubits: int) -> np.int64:
        qubits = cudaq.qvector(numQubits)
        return numQubits + 1

    results = cudaq.run(simple_int64, 2, shots_count=2)
    print(results)
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
    print(results)
    assert len(results) == 2
    assert results[0] == 3.0
    assert results[1] == 3.0

    @cudaq.kernel
    def simple_float64(numQubits: int) -> np.float64:
        return numQubits + 1

    results = cudaq.run(simple_float64, 2, shots_count=2)
    print(results)
    assert len(results) == 2
    assert results[0] == 3.0
    assert results[1] == 3.0


def test_run_errors():

    @cudaq.kernel
    def simple_no_return(numQubits: int):
        qubits = cudaq.qvector(numQubits)

    @cudaq.kernel
    def simple_no_args() -> int:
        return 1

    @cudaq.kernel
    def simple(numQubits: int) -> int:
        return 1

    with pytest.raises(RuntimeError) as e:
        cudaq.run(simple_no_return, 2)
    assert 'cudaq.run only supports kernels that return a value.' in repr(e)

    with pytest.raises(RuntimeError) as e:
        cudaq.run(simple, 2, shots_count=-1)
    assert 'Invalid shots_count. Must be non-negative.' in repr(e)

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
