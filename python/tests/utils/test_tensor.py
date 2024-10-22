# ============================================================================ #
# Copyright (c) 2024 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest
import numpy as np
import cudaq


def tensor_class(request):
    return request.param


def test_tensor_creation():
    shape = [2, 3, 4]
    t = cudaq.Tensor(shape, dtype=np.int32)
    assert t.shape() == shape
    assert t.size() == 24


def test_tensor_indexing():
    t = cudaq.Tensor([2, 3])
    t[0, 0] = 1
    t[0, 1] = 2
    t[0, 2] = 3
    t[1, 0] = 4
    t[1, 1] = 5
    t[1, 2] = 6

    assert t[0, 0] == 1
    assert t[0, 1] == 2
    assert t[0, 2] == 3
    assert t[1, 0] == 4
    assert t[1, 1] == 5
    assert t[1, 2] == 6


def test_tensor_numpy_conversion():
    shape = [2, 3]
    t = cudaq.Tensor(shape)
    for i in range(2):
        for j in range(3):
            t[i, j] = i * 3 + j

    arr = np.array(t, copy=False)
    assert arr.shape == tuple(shape)
    assert np.array_equal(arr, np.array([[0, 1, 2], [3, 4, 5]]))


def test_numpy_to_tensor():
    arr = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.complex128)
    t = cudaq.Tensor(arr.shape, dtype=np.complex128)
    a = np.array(t, copy=False, dtype=np.complex128)
    np.array(t, copy=False, dtype=np.complex128)[:] = arr
    for i in range(2):
        for j in range(3):
            assert t[i, j] == arr[i, j]


def test_tensor_operations():
    t = cudaq.Tensor([2, 2])
    t[0, 0] = 1
    t[0, 1] = 2
    t[1, 0] = 3
    t[1, 1] = 4

    arr = np.array(t, copy=False)

    # Test addition
    result = arr + 1
    assert np.array_equal(result, np.array([[2, 3], [4, 5]]))

    # Test multiplication
    result = arr * 2
    assert np.array_equal(result, np.array([[2, 4], [6, 8]]))

    # Test matrix multiplication
    result = np.dot(arr, arr)
    assert np.array_equal(result, np.array([[7, 10], [15, 22]]))


def test_tensor_large_data():
    shape = [100, 100, 100]
    t = cudaq.Tensor(shape)

    # Fill with some
    arr = np.array(t, copy=False)
    arr[:] = np.arange(1000000).reshape(shape)

    # Verify some values
    assert t[0, 0, 0] == 0
    assert t[50, 50, 50] == 505050
    assert t[99, 99, 99] == 999999


def test_tensor_errors():
    t = cudaq.Tensor([2, 2])

    # Test index out of bounds
    with pytest.raises(RuntimeError):
        t[2, 0] = 1

    with pytest.raises(RuntimeError):
        _ = t[0, 2]

    # Test wrong number of indices
    with pytest.raises(TypeError):
        t[0] = 1

    with pytest.raises(RuntimeError):
        _ = t[0, 0, 0]
