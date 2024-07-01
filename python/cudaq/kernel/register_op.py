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


def register_operation(operation_name: str, num_targets: int, num_params: int,
                       unitary):
    global globalRegisteredOperations
    """
    Register a new quantum operation at runtime. Users must 
    provide the unitary matrix as a 2D NumPy array. The operation 
    name is inferred from the name of the assigned variable. 
    ```python
        cudaq.register_operation(myOp, 1, 0, unitary)
        @cudaq.kernel
        def kernel():
            ...
            myOp(...)
            ...
    ```
    """
    if operation_name == None:
        raise RuntimeError("custom operation name not provided.")

    if num_targets < 1:
        raise RuntimeError("at least one target required.")

    if num_targets > 8:
        raise RuntimeError("custom operations on upto 8 qubits supported.")

    if num_params > 0:
        raise RuntimeError("parameterized custom operations not yet supported.")

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

    # Check size of matrix
    assert (num_targets == np.log2(np.sqrt(matrix.size)))

    # Register the operation name so JIT AST can get it.
    globalRegisteredOperations[operation_name] = matrix

    # Make available to kernel builder object
    setattr(PyKernel, operation_name,
            partialmethod(__generalCustomOperation, operation_name))

    return
