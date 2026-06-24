# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import math
import numpy as np
from functools import partialmethod
from typing import Callable, List

from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime
from .kernel_builder import PyKernel, __generalCustomOperation
from .utils import globalRegisteredOperations


def register_operation(operation_name: str, unitary):
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

    global globalRegisteredOperations

    if not operation_name or not operation_name.strip():
        raise RuntimeError("custom operation name not provided.")

    if isinstance(unitary, Callable):
        raise RuntimeError("parameterized custom operations not yet supported.")

    if isinstance(unitary, np.matrix) or isinstance(unitary, List):
        matrix = np.array(unitary)
    elif isinstance(unitary, np.ndarray):
        matrix = unitary
    else:
        raise RuntimeError("unknown type of unitary.")

    matrix = matrix.flatten()

    # Size must be a perfect square and power of 2; at least 4 for 1-qubit operation
    if matrix.size < 4 or (matrix.size & (matrix.size - 1)) != 0 or (math.isqrt(
            matrix.size)**2) != matrix.size:
        raise RuntimeError(
            "invalid matrix size, required 2^N * 2^N for N-qubit operation.")

    # Register the operation name so JIT AST can get it.
    globalRegisteredOperations[operation_name] = matrix

    # Make available to kernel builder object
    setattr(PyKernel, operation_name,
            partialmethod(__generalCustomOperation, operation_name))
    # Let the runtime know about this registered operation.
    # Note: the matrix generator/construction is not known by the ExecutionManager in this case since we don't expect the ExecutionManager to be involved.
    cudaq_runtime.register_custom_operation(operation_name)

    return
