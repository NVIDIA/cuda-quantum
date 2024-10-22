# ============================================================================ #
# Copyright (c) 2022-2024 NVIDIA Corporation & Affiliates.                     #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import numpy as np
from ..mlir._mlir_libs._quakeDialects.cudaq_runtime import utils


class Tensor:
    """
    A multi-dimensional array class for CUDA-QX quantum computing libraries.

    This class wraps CUDA-QX core tensor types and provides a NumPy-like interface
    for quantum tensor operations.

    Attributes:
        _tensor: The underlying CUDA-Q core tensor object.
        _dtype: The data type of the tensor elements.
    """

    def __init__(self, data=None, shape=None, dtype=None):
        """
        Initialize a Tensor object.

        Args:
            data (array-like, optional): Initial data for the tensor. Can be a list, tuple, 
                                         numpy array, or CUDA-Q core tensor.
            shape (tuple, optional): Shape of the tensor if creating an empty tensor.
            dtype (numpy.dtype, optional): Data type of the tensor elements.

        Raises:
            RuntimeError: If initializing from a CUDA-Q core tensor without specifying dtype.
            ValueError: If data is not a supported type or if dtype is unsupported.
        """
        self._tensor = None
        self._dtype = None

        if isinstance(data,
                      (utils.Tensor_float, utils.Tensor_int32, utils.Tensor_int64,
                       utils.Tensor_complex64, utils.Tensor_complex128, utils.Tensor_uint8)):
            if dtype is None:
                raise RuntimeError(
                    "Tensor init from CUDA-QX Tensor must provide dtype.")
            self._tensor = data
            self._dtype = dtype
            return

        if data is not None:
            if isinstance(data, (list, tuple, np.ndarray)):
                data = np.array(data)
                if dtype is None:
                    dtype = data.dtype
            else:
                raise ValueError("Data must be a list, tuple, or numpy array")

            self._create_tensor(data, dtype)
        elif shape is not None:
            if dtype is None:
                dtype = np.float64
            self._create_empty_tensor(shape, dtype)
        else:
            if dtype is None:
                dtype = np.float64
            self._create_empty_tensor([], dtype)

    def _create_tensor(self, data, dtype):
        """
        Create a tensor from given data and dtype.

        Args:
            data (numpy.ndarray): The data to initialize the tensor with.
            dtype (numpy.dtype): The data type of the tensor elements.

        Raises:
            ValueError: If the dtype is unsupported.
        """
        self._dtype = dtype
        if dtype == np.float64:
            self._tensor = utils.Tensor_float(data)
        elif dtype == np.complex64:
            self._tensor = utils.Tensor_complex64(data)
        elif dtype == np.int32:
            self._tensor = utils.Tensor_int32(data)
        elif dtype == np.int64:
            self._tensor = utils.Tensor_int64(data)
        elif dtype == np.complex128:
            self._tensor = utils.Tensor_complex128(data)
        elif dtype == np.uint8:
            self._tensor = utils.Tensor_uint8(data)
        else:
            raise ValueError("Unsupported dtype")

    def _create_empty_tensor(self, shape, dtype):
        """
        Create an empty tensor with given shape and dtype.

        Args:
            shape (tuple): The shape of the tensor.
            dtype (numpy.dtype): The data type of the tensor elements.

        Raises:
            ValueError: If the dtype is unsupported.
        """
        self._dtype = dtype
        if dtype == np.float64:
            self._tensor = utils.Tensor_float(shape)
        elif dtype == np.complex64:
            self._tensor = utils.Tensor_complex64(shape)
        elif dtype == np.int32:
            self._tensor = utils.Tensor_int32(shape)
        elif dtype == np.int64:
            self._tensor = utils.Tensor_int64(shape)
        elif dtype == np.complex128:
            self._tensor = utils.Tensor_complex128(shape)
        elif dtype == np.uint8:
            self._tensor = utils.Tensor_uint8(shape)
        else:
            raise ValueError("Unsupported dtype")

    def __getitem__(self, key):
        """
        Get item(s) from the tensor using NumPy-like indexing.

        Args:
            key: Index or slice object.

        Returns:
            The value(s) at the specified index/indices.
        """
        return self.at(list(key))

    def __setitem__(self, key, value):
        """
        Set item(s) in the tensor using NumPy-like indexing.

        Args:
            key: Index or slice object.
            value: Value(s) to set.
        """
        self._tensor[key] = value

    def __array__(self, dtype=None):
        """
        Convert the tensor to a NumPy array.

        Args:
            dtype (numpy.dtype, optional): The desired dtype of the resulting array.

        Returns:
            numpy.ndarray: A NumPy array representation of the tensor.

        Raises:
            RuntimeError: If the requested dtype doesn't match the tensor's dtype.
        """
        if dtype is None:
            return np.array(self._tensor, copy=False, dtype=dtype)
        else:
            if dtype != self._dtype:
                raise RuntimeError(
                    f"invalid numpy dtype conversion ({dtype} vs {self._dtype})"
                )
            return np.array(self._tensor, copy=False, dtype=dtype)

    def rank(self):
        """
        Get the rank (number of dimensions) of the tensor.

        Returns:
            int: The rank of the tensor.
        """
        return self._tensor.rank()

    def size(self):
        """
        Get the total number of elements in the tensor.

        Returns:
            int: The total number of elements.
        """
        return self._tensor.size()

    def shape(self):
        """
        Get the shape of the tensor.

        Returns:
            tuple: The shape of the tensor.
        """
        return self._tensor.shape()

    def at(self, indices):
        """
        Get the value at the specified indices.

        Args:
            indices (list): List of indices.

        Returns:
            The value at the specified indices.
        """
        return self._tensor.at(indices)

    def copy(self, data):
        """
        Copy data into the tensor.

        Args:
            data: The array-like data to copy into the tensor.
        """
        np_data = np.array(data, dtype=self._scalar_type_to_numpy(self._dtype))
        self._tensor.copy(np_data)

    def take(self, data):
        """
        Take data from the tensor.

        Args:
            data: The array-like data to take from the tensor.
        """
        np_data = np.array(data, dtype=self._scalar_type_to_numpy(self._dtype))
        self._tensor.take(np_data)

    def borrow(self, data):
        """
        Borrow data from the tensor.

        Args:
            data: The array-like data to borrow from the tensor.
        """
        np_data = np.array(data, dtype=self._scalar_type_to_numpy(self._dtype))
        self._tensor.borrow(np_data)

    def dump(self):
        """
        Dump the contents of the tensor.
        """
        self._tensor.dump()

    def __str__(self):
        """
        Print a NumPy-like string representation of the tensor 
        """
        return str(np.array(self._tensor, copy=False))

    @property
    def dtype(self):
        """
        Get the data type of the tensor.

        Returns:
            numpy.dtype: The data type of the tensor elements.
        """
        return self._dtype
