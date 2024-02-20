# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from ..mlir._mlir_libs._quakeDialects import cudaq_runtime
from ..kernel.kernel_builder import PyKernel
from ..kernel.kernel_decorator import PyKernelDecorator
from ..mlir.dialects import quake, cc

import numpy as np
from typing import List


def __isBroadcast(kernel, *args):
    # kernel could be a PyKernel or PyKernelDecorator
    if isinstance(kernel, PyKernel):
        argTypes = kernel.mlirArgTypes
        if len(argTypes) == 0 or len(args) == 0:
            return False

        firstArg = args[0]
        firstArgTypeIsStdvec = cc.StdvecType.isinstance(argTypes[0])
        if (isinstance(firstArg, list) or
                isinstance(firstArg, List)) and not firstArgTypeIsStdvec:
            return True

        if hasattr(firstArg, "shape"):
            shape = firstArg.shape
            if len(shape) == 1 and not firstArgTypeIsStdvec:
                return True

            if len(shape) == 2:
                return True

        return False

    elif isinstance(kernel, PyKernelDecorator):
        argTypes = kernel.signature
        if len(argTypes) == 0 or len(args) == 0:
            return False
        firstArg = args[0]
        firstArgType = next(iter(argTypes))
        firstArgTypeIsStdvec = argTypes[firstArgType] in [
            list, list[float], list[complex], list[int], np.ndarray, List,
            List[float], List[complex], List[int]
        ]
        if (isinstance(firstArg, list) or
                isinstance(firstArg, List)) and not firstArgTypeIsStdvec:
            return True

        if hasattr(firstArg, "shape"):
            shape = firstArg.shape
            if len(shape) == 1 and not firstArgTypeIsStdvec:
                return True

            if len(shape) == 2:
                return True

        return False


def __createArgumentSet(*args):
    nArgSets = len(args[0])
    argSet = []
    for j in range(nArgSets):
        currentArgs = [0 for i in range(len(args))]
        for i, arg in enumerate(args):

            if isinstance(arg, list) or isinstance(arg, List):
                currentArgs[i] = arg[j]

            if hasattr(arg, "tolist"):
                shape = arg.shape
                if len(shape) == 2:
                    currentArgs[i] = arg[j].tolist()
                else:
                    currentArgs[i] = arg.tolist()[j]

        argSet.append(tuple(currentArgs))
    return argSet
