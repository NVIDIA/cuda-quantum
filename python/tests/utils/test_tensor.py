# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
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


def test_tensor_creation_from_data():
    t = cudaq.Tensor(data=[[0, 1], [2, 3]])
    assert t.shape() == [2, 2]
    assert t.size() == 4
    assert t.rank() == 2
    assert t[0, 0] == 0
    assert t[0, 1] == 1
    t[0, 0] = 2
    t[1, 1] = 4
    assert t[0, 0] == 2
    assert t[1, 1] == 4


def test_tensor_creation_from_data_int32():
    t = cudaq.Tensor(data=[[0, 1], [2, 3]], dtype=np.int32)
    assert t.shape() == [2, 2]
    assert t.size() == 4
    assert t.rank() == 2
    assert t[0, 0] == 0
    assert t[0, 1] == 1
    t[0, 0] = 2
    t[1, 1] = 4
    assert t[0, 0] == 2
    assert t[1, 1] == 4

    t = cudaq.Tensor(data=[0, 1], dtype=np.int32)
    assert t.shape() == [2]
    assert t.size() == 2
    assert t.rank() == 1
    assert t[0] == 0
    assert t[1] == 1
    t[0] = 2
    t[1] = 4
    assert t[0] == 2
    assert t[1] == 4


def test_tensor_creation_from_data_float32():
    t = cudaq.Tensor(data=[[0, 1], [2, 3]], dtype=np.float32)
    assert t.shape() == [2, 2]
    assert t.size() == 4
    assert t.rank() == 2
    assert t[0, 0] == 0
    assert t[0, 1] == 1
    t[0, 0] = 2
    t[1, 1] = 4
    assert t[0, 0] == 2
    assert t[1, 1] == 4

    t = cudaq.Tensor(data=[0, 1], dtype=np.float32)
    assert t.shape() == [2]
    assert t.size() == 2
    assert t.rank() == 1
    assert t[0] == 0
    assert t[1] == 1
    t[0] = 2
    t[1] = 4
    assert t[0] == 2
    assert t[1] == 4


def test_tensor_creation_from_data_complex64():
    t = cudaq.Tensor(data=[[0, 1], [2, 3]], dtype=np.complex64)
    assert t.shape() == [2, 2]
    assert t.size() == 4
    assert t.rank() == 2
    assert t[0, 0] == 0
    assert t[0, 1] == 1
    t[0, 0] = 2
    t[1, 1] = 4
    assert t[0, 0] == 2
    assert t[1, 1] == 4

    t = cudaq.Tensor(data=[0, 1], dtype=np.complex64)
    assert t.shape() == [2]
    assert t.size() == 2
    assert t.rank() == 1
    assert t[0] == 0
    assert t[1] == 1
    t[0] = 2
    t[1] = 4
    assert t[0] == 2
    assert t[1] == 4


def test_tensor_creation():
    shape = [2, 3, 4]
    t = cudaq.Tensor(shape)
    assert t.shape() == shape
    assert t.size() == 24

    t = cudaq.Tensor(shape=shape)
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


def test_tensor_indexing_int32():
    t = cudaq.Tensor([2, 3], dtype=np.int32)
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


def test_tensor_indexing_float32():
    shape = [2, 3]
    t = cudaq.Tensor(shape, dtype=np.float32)
    t[0, 0] = 1.1
    t[0, 1] = 2.2
    t[0, 2] = 3.3
    t[1, 0] = 4.4
    t[1, 1] = 5.5
    t[1, 2] = 6.6

    assert np.isclose(t[0, 0], 1.1)
    assert np.isclose(t[0, 1], 2.2)
    assert np.isclose(t[0, 2], 3.3)
    assert np.isclose(t[1, 0], 4.4)
    assert np.isclose(t[1, 1], 5.5)
    assert np.isclose(t[1, 2], 6.6)


def test_tensor_indexing_float64():
    shape = [2, 3]
    t = cudaq.Tensor(shape, dtype=np.float64)
    t[0, 0] = 1.1
    t[0, 1] = 2.2
    t[0, 2] = 3.3
    t[1, 0] = 4.4
    t[1, 1] = 5.5
    t[1, 2] = 6.6

    assert np.isclose(t[0, 0], 1.1)
    assert np.isclose(t[0, 1], 2.2)
    assert np.isclose(t[0, 2], 3.3)
    assert np.isclose(t[1, 0], 4.4)
    assert np.isclose(t[1, 1], 5.5)
    assert np.isclose(t[1, 2], 6.6)


def test_tensor_indexing_complex64():
    shape = [2, 3]
    t = cudaq.Tensor(shape, dtype=np.complex64)
    t[0, 0] = 1.1 + 1j
    t[0, 1] = 2.2 + 2j
    t[0, 2] = 3.3 + 3j
    t[1, 0] = 4.4 + 4j
    t[1, 1] = 5.5 + 5j
    t[1, 2] = 6.6 + 6j

    assert np.isclose(t[0, 0], 1.1 + 1j)
    assert np.isclose(t[0, 1], 2.2 + 2j)
    assert np.isclose(t[0, 2], 3.3 + 3j)
    assert np.isclose(t[1, 0], 4.4 + 4j)
    assert np.isclose(t[1, 1], 5.5 + 5j)
    assert np.isclose(t[1, 2], 6.6 + 6j)


def test_tensor_indexing_complex128():
    shape = [2, 3]
    t = cudaq.Tensor(shape, dtype=np.complex128)
    t[0, 0] = 1.1 + 1j
    t[0, 1] = 2.2 + 2j
    t[0, 2] = 3.3 + 3j
    t[1, 0] = 4.4 + 4j
    t[1, 1] = 5.5 + 5j
    t[1, 2] = 6.6 + 6j

    assert np.isclose(t[0, 0], 1.1 + 1j)
    assert np.isclose(t[0, 1], 2.2 + 2j)
    assert np.isclose(t[0, 2], 3.3 + 3j)
    assert np.isclose(t[1, 0], 4.4 + 4j)
    assert np.isclose(t[1, 1], 5.5 + 5j)
    assert np.isclose(t[1, 2], 6.6 + 6j)


def test_tensor_numpy_conversion():
    shape = [2, 3]
    t = cudaq.Tensor(shape)
    for i in range(2):
        for j in range(3):
            t[i, j] = i * 3 + j

    arr = np.array(t, copy=False)
    assert arr.shape == tuple(shape)
    assert np.array_equal(arr, np.array([[0, 1, 2], [3, 4, 5]]))


def test_numpy_to_tensor_float64():
    arr = np.array([[0., 1., 2.], [3., 4., 5.]], dtype=np.float64)
    t = cudaq.Tensor(arr.shape, dtype=np.float64)
    a = np.array(t, copy=False, dtype=np.float64)
    np.array(t, copy=False, dtype=np.float64)[:] = arr
    for i in range(2):
        for j in range(3):
            assert t[i, j] == arr[i, j]


def test_numpy_to_tensor_complex128():
    arr = np.array([[0. + 1j, 1. + 1j, 2. + 1j], [3. + 1j, 4. + 1j, 5. + 1j]],
                   dtype=np.complex128)
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


def test_tensor_copy():
    t = cudaq.Tensor([2, 2])
    arr = np.array([[1, 2], [3, 4]])
    t.copy(arr)

    assert t[0, 0] == arr[0, 0]
    assert t[0, 1] == arr[0, 1]
    assert t[1, 0] == arr[1, 0]
    assert t[1, 1] == arr[1, 1]

    arr[0, 0] = 100
    assert t[0, 0] == 1


def test_tensor_borrow():
    t = cudaq.Tensor([2, 2])
    arr = np.array([[1, 2], [3, 4]])
    t.borrow(arr)

    assert t[0, 0] == arr[0, 0]
    assert t[0, 1] == arr[0, 1]
    assert t[1, 0] == arr[1, 0]
    assert t[1, 1] == arr[1, 1]

    arr[0, 0] = 100
    assert t[0, 0] == 1


def test_tensor_take_float64():
    t = cudaq.Tensor([2, 2])
    arr = np.array([[1, 2], [3, 4]])
    t.take(arr)

    assert t[0, 0] == arr[0, 0]
    assert t[0, 1] == arr[0, 1]
    assert t[1, 0] == arr[1, 0]
    assert t[1, 1] == arr[1, 1]

    arr[0, 0] = 100
    assert t[0, 0] == 1


def test_tensor_dump():
    t = cudaq.Tensor([2, 2])
    arr = np.array([[1, 2], [3, 4]])
    t.take(arr)
    s = t.__str__()
    assert s == '[[1. 2.]\n [3. 4.]]'
    t.dump()


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
    with pytest.raises(RuntimeError):
        t[0] = 1

    with pytest.raises(RuntimeError):
        _ = t[0, 0, 0]
