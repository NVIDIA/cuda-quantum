# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from cudaq.kernel.kernel_builder import PyKernel
from cudaq.kernel.kernel_decorator import isa_kernel_decorator
from cudaq.kernel.utils import mlirTypeToPyType
from cudaq.mlir.ir import Type as MlirType
from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime
from cudaq.mlir.dialects import cc

import numpy as np
from typing import List


def __isBroadcast(kernel, *args):
    # kernel could be a PyKernel or kernel decorator
    if isinstance(kernel, PyKernel):
        argTypes = kernel.mlirArgTypes
    elif isa_kernel_decorator(kernel):
        argTypes = kernel.arg_types()
    else:
        return False

    if len(args) == 0 or len(argTypes) == 0:
        return False

    firstArg = args[0]
    num_nested_lists_arg = _count_nested_lists(firstArg)
    num_nested_lists_type = _count_nested_lists(argTypes[0])
    return num_nested_lists_arg == num_nested_lists_type + 1


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


def _count_nested_lists(obj: np.ndarray | MlirType | list) -> int:
    """
    Count the level of nesting of a list-like object.

    Supports `np.ndarray`, `MlirType`, and `list`.
    """
    count = 0
    while True:
        if isinstance(obj, list):
            count += 1
            if len(obj) == 0:
                break
            obj = obj[0]
        elif isinstance(obj, MlirType) and cc.StdvecType.isinstance(obj):
            count += 1
            obj = cc.StdvecType.getElementType(obj)
        elif hasattr(obj, "shape"):
            count += len(obj.shape)
            break
        else:
            break
    return count
