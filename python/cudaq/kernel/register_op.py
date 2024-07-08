# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from functools import partialmethod
import numpy as np
from typing import Callable, List

from .utils import globalRegisteredOperations
from .kernel_builder import PyKernel, __generalCustomOperation


def register_operation(operation_name: str, unitary):
    global globalRegisteredOperations
    """
    Register a new quantum operation at runtime. Users must provide the unitary
    matrix as a 1D NumPy array in row-major format with MSB qubit ordering. 
    ```python
        cudaq.register_operation("myOp", unitary)
        @cudaq.kernel
        def kernel():
            ...
            myOp(...)
            ...
    ```
    """
    if operation_name == None:
        raise RuntimeError("custom operation name not provided.")

    if isinstance(unitary, Callable):
        raise RuntimeError("parameterized custom operations not yet supported.")

    if isinstance(unitary, np.ndarray):
        if (len(unitary.shape) != unitary.ndim):
            raise RuntimeError(
                "provide a 1D array for the matrix representation in row-major format."
            )
        matrix = unitary
    elif isinstance(unitary, List):
        matrix = np.array(unitary)
    else:
        raise RuntimeError("unknown type of unitary.")

    # TODO: Flatten the matrix if not flattened
    assert (matrix.ndim == len(matrix.shape))

    # Size must be a power of 2
    assert (matrix.size != 0)
    assert (matrix.size & (matrix.size - 1) == 0)

    # Register the operation name so JIT AST can get it.
    globalRegisteredOperations[operation_name] = matrix

    # Make available to kernel builder object
    setattr(PyKernel, operation_name,
            partialmethod(__generalCustomOperation, operation_name))

    return
