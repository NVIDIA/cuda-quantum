# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from functools import partialmethod
import inspect
from typing import Callable, NamedTuple

from ..kernel.ast_bridge import globalRegisteredUnitaries
from ..kernel.kernel_builder import PyKernel, __generalCustomOperation

CustomQuantumOperation = NamedTuple("CustomQuantumOperation",
                                    [('operation_name', str), ('unitary', any),
                                     ('num_parameters', int)])


def register_operation(unitary, operation_name=None):
    global globalRegisteredUnitaries
    """
    Register a new quantum operation at runtime. Users must 
    provide the unitary matrix as a 2D NumPy array. The operation 
    name is inferred from the name of the assigned variable. 

    ```python
        myOp = cudaq.register_operation(unitary)

        @cudaq.kernel
        def kernel():
            ...
            myOp(...)
            ...
    ```
    """
    if operation_name == None:
        lastFrame = inspect.currentframe().f_back
        frameInfo = inspect.getframeinfo(lastFrame)
        codeContext = frameInfo.code_context[0]
        if not '=' in codeContext:
            raise RuntimeError(
                "[register_operation] operation_name not given and variable name not set."
            )
        operation_name = codeContext.split('=')[0].strip()

    numParameters = 0
    if isinstance(unitary, Callable):
        numParameters = len(inspect.getfullargspec(unitary).args)
    registeredOp = CustomQuantumOperation(operation_name, unitary,
                                          numParameters)

    # Register the operation name so JIT AST can get it.
    globalRegisteredUnitaries[operation_name] = unitary
    # Make available to kernel builder object
    setattr(PyKernel, operation_name,
            partialmethod(__generalCustomOperation, operation_name))

    return registeredOp
