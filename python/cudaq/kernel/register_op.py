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

    def __init__(self, name, num_targets, num_params, matrix):
        self.name = name
        self.num_targets = num_targets
        self.num_params = num_params
        self.matrix = matrix

        if isinstance(matrix, Callable):
            dummy_params = [0.0] * self.num_params
            expected_targets = int(np.log2(np.sqrt(matrix(dummy_params).size)))
        else:
            expected_targets = int(np.log2(np.sqrt(matrix.size)))
        assert (self.num_targets == expected_targets)

        self.gen_func_name = self.name + "_generator_" + str(self.num_targets)

    def getUnitary(self, params: List[float] = None):
        if params and isinstance(self.matrix, Callable):
            assert (self.num_params == len(params))
            return self.matrix(params)
        return self.matrix

    def getGeneratorFunc(self):
        # outputArr = np.array()
        def generator(outputArr, inputParams):
            outputArr = self.getUnitary(inputParams)
            return

        generator.__name__ = self.gen_func_name
        generator.__qualname__ = self.gen_func_name
        return generator


def register_operation(num_targets, num_params, unitary, operation_name=None):
    global globalRegisteredOperations
    """
    Register a new quantum operation at runtime. Users must 
    provide the unitary matrix as a 2D NumPy array. The operation 
    name is inferred from the name of the assigned variable. 
    ```python
        myOp = cudaq.register_operation(num_targets, num_params, unitary)
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

    registeredOp = UnitaryOperation(operation_name, num_targets, num_params,
                                    unitary)

    # Register the operation name so JIT AST can get it.
    globalRegisteredOperations[operation_name] = registeredOp

    # # Make available to kernel builder object
    # setattr(PyKernel, operation_name,
    #         partialmethod(__generalCustomOperation, operation_name))

    return registeredOp
