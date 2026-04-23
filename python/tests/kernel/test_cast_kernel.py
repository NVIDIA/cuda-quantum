# ============================================================================ #
# Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import numpy as np
import pytest


# bool <-> int32
def testBoolInt32():

    @cudaq.kernel
    def kernelBoolInt32() -> np.int32:
        return True

    assert cudaq.run(kernelBoolInt32, shots_count=1) == [1]

    @cudaq.kernel
    def kernelInt32Bool(i: np.int32) -> bool:
        return i

    assert cudaq.run(kernelInt32Bool, -1, shots_count=1) == [True]


# bool <-> int64
def testBoolInt64():

    @cudaq.kernel
    def kernelBoolInt64() -> int:
        return True

    assert cudaq.run(kernelBoolInt64, shots_count=1) == [1]

    @cudaq.kernel
    def kernelInt64Bool() -> bool:
        return -1

    assert cudaq.run(kernelInt64Bool, shots_count=1) == [True]


# bool <-> float32
def testBoolFloat32():

    @cudaq.kernel
    def kernelBoolFloat64() -> np.float32:
        return True

    assert cudaq.run(kernelBoolFloat64, shots_count=1) == [1.0]

    @cudaq.kernel
    def kernelFloat32Bool(f: np.float32) -> bool:
        return 1.2

    assert cudaq.run(kernelFloat32Bool, -1.2, shots_count=1) == [True]


# bool <-> float64
def testBoolFloat64():

    @cudaq.kernel
    def kernelBoolFloat64() -> float:
        return True

    assert cudaq.run(kernelBoolFloat64, shots_count=1) == [1.0]

    @cudaq.kernel
    def kernelFloat64Bool() -> bool:
        return 1.2

    assert cudaq.run(kernelFloat64Bool, shots_count=1) == [True]


# int32 <-> int64
def testInt32Int64():

    @cudaq.kernel
    def kernelInt32Int64(i: np.int32) -> int:
        return i

    assert cudaq.run(kernelInt32Int64, -2, shots_count=1) == [-2]

    @cudaq.kernel
    def kernelInt64Int32() -> np.int32:
        return -2

    assert cudaq.run(kernelInt64Int32, shots_count=1) == [-2]


# int32 <-> float32
def testInt32Float32():

    @cudaq.kernel
    def kernelInt32Float32(i: np.int32) -> np.float32:
        return i

    assert cudaq.run(kernelInt32Float32, -2, shots_count=1) == [-2.0]

    @cudaq.kernel
    def kernelFloat32Int32(f: np.float32) -> np.int32:
        return f

    assert cudaq.run(kernelFloat32Int32, -1.2, shots_count=1) == [-1]


# int32 <-> float64
def testInt32Float64():

    @cudaq.kernel
    def kernelInt32Float64(i: np.int32) -> float:
        return -2

    assert cudaq.run(kernelInt32Float64, -2, shots_count=1) == [-2.0]

    @cudaq.kernel
    def kernelFloat64Int32() -> np.int32:
        return -1.2

    assert cudaq.run(kernelFloat64Int32, shots_count=1) == [-1]


# int64 <-> float32
def testInt64Float32():

    @cudaq.kernel
    def kernelInt64Float32() -> np.float32:
        return -3

    assert cudaq.run(kernelInt64Float32, shots_count=1) == [-3]

    @cudaq.kernel
    def kernelFloat32Int64(f: np.float32) -> int:
        return f

    assert cudaq.run(kernelFloat32Int64, -2.3, shots_count=1) == [-2]


# int64 <-> float64
def testInt64Float64():

    @cudaq.kernel
    def kernelInt64Float64() -> float:
        return -2

    assert cudaq.run(kernelInt64Float64, shots_count=1) == [-2.0]

    @cudaq.kernel
    def kernelFloat64Int64() -> int:
        return -1.2

    assert cudaq.run(kernelFloat64Int64, shots_count=1) == [-1]


# float32 <-> float64
def testFloat32Float64():

    @cudaq.kernel
    def kernelFloat32Float64(f: np.float32) -> float:
        return f

    assert cudaq.run(kernelFloat32Float64, -2.0, shots_count=1) == [-2.0]

    @cudaq.kernel
    def kernelFloat64Float32(f: float) -> np.float32:
        return f

    assert cudaq.run(kernelFloat64Float32, -2.0, shots_count=1) == [-2.0]


def test_multiplication():

    @cudaq.kernel
    def mult_check(angle: float) -> float:
        M_PI = 3.1415926536
        phase = 2 * M_PI * angle
        return phase

    result = cudaq.run(mult_check, 0.1, shots_count=1)
    assert result[0] == pytest.approx(0.6283185307179586)
