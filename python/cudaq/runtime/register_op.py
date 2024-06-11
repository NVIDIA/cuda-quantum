# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import inspect
import numpy as np
from typing import List, Callable

# Keep a global registry of all registered custom operations.
globalRegisteredOperations = {}

class UnitaryOperation:

    def __init__(self, name, num_params, matrix):
        self.name = name
        # self.num_targets = num_targets
        self.num_params = num_params
        self.matrix = matrix

    def getUnitary(self, params : List[float] = None):
        assert (self.num_params == len(params))
        if params and isinstance(self.matrix, Callable):
            return self.matrix(params)
        return self.matrix


def register_operation(unitary, operation_name=None):
    global globalRegisteredOperations
    
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

    registeredOp = UnitaryOperation(operation_name, numParameters, unitary)

    # Register the operation name so JIT AST can get it.
    globalRegisteredOperations[operation_name] = registeredOp

    # # Make available to kernel builder object
    # setattr(PyKernel, operation_name,
    #         partialmethod(__generalCustomOperation, operation_name))
    
    return registeredOp