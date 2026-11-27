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
from cudaq.kernel.utils import mlirTypeToPyType, nvqppPrefix
from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime
from cudaq.mlir.dialects import cc

import numpy as np
from typing import List


def _kernel_has_conditionals_on_measure(kernel) -> bool:
    """Return True if @p kernel branches on a measurement result.

    Shared by primitives that need to reject measurement-dependent
    control flow with their own diagnostic. The caller is responsible for
    raising the API-specific error message; this helper only returns the
    boolean detection result.
    """
    if isa_kernel_decorator(kernel):
        if not kernel.supports_compilation():
            return False
        for operation in kernel.qkeModule.body.operations:
            op_name = getattr(operation.name,
                              'value', operation.name) if hasattr(
                                  operation, 'name') else None
            if (op_name == nvqppPrefix + kernel.uniqName and
                    'qubitMeasurementFeedback' in operation.attributes):
                return True
        return False
    if isinstance(kernel, PyKernel):
        return kernel.conditionalOnMeasure
    return False


def __isBroadcast(kernel, *args):
    isBuilderKernel = isinstance(kernel, PyKernel)
    if isBuilderKernel:
        argTypes = kernel.mlirArgTypes
    elif isa_kernel_decorator(kernel):
        argTypes = kernel.arg_types()
    else:
        return False

    if len(argTypes) == 0 or len(args) == 0:
        return False

    # A 2D array in any position implies a broadcast. Keep the legacy
    # diagnostics for this path.
    if any(hasattr(arg, "shape") and len(arg.shape) == 2 for arg in args):
        for i, arg in enumerate(args):
            if isinstance(arg, (int, float, bool, str)):
                operation = "sample or observe" if isBuilderKernel else "observe"
                raise RuntimeError(
                    f"2D array argument provided for an {operation} broadcast, but argument {i} ({type(arg)}) must be a list."
                )
        return True

    # A flat sequence is a broadcast only if the kernel does not expect a
    # sequence as its first argument.
    if cc.SequenceType.isinstance(argTypes[0]):
        return False

    firstArg = args[0]
    isBroadcast = (isinstance(firstArg, (list, List)) or
                   (hasattr(firstArg, "shape") and len(firstArg.shape) == 1))
    if isBroadcast:
        for i, arg in enumerate(args):
            if isinstance(arg, (int, float, bool, str, complex, np.generic)):
                raise RuntimeError(
                    f"Argument {i} ({type(arg)}) must be a list or ndarray for sample or observe broadcasting."
                )
    return isBroadcast


def __createArgumentSet(*args):
    nArgSets = len(args[0])
    if nArgSets == 0:
        return []

    # Materialize array-like arguments once.
    materializedArgs = []
    arrayRanks = []
    for i, arg in enumerate(args):
        if hasattr(arg, "tolist"):
            arrayRank = len(arg.shape)
            arrayRanks.append(arrayRank)
            nProvided = arg.shape[0] if arrayRank > 0 else 0
            if nProvided < nArgSets:
                raise RuntimeError(
                    f"Broadcast argument {i} provides {nProvided} argument sets, but {nArgSets} are required."
                )
            # A later matrix argument may have more rows than the first
            # argument that defines `nArgSets`. Do not materialize unused rows.
            m = arg[:nArgSets] if arrayRank == 2 else arg
            materializedArgs.append(m.tolist())
        else:
            if (isinstance(arg, list) or isinstance(arg, List)) and \
                    len(arg) < nArgSets:
                raise RuntimeError(
                    f"Broadcast argument {i} provides {len(arg)} argument sets, but {nArgSets} are required."
                )
            arrayRanks.append(None)
            materializedArgs.append(arg)

    argSet = []
    for j in range(nArgSets):
        currentArgs = [0 for i in range(len(args))]
        for i, arg in enumerate(args):

            if isinstance(arg, list) or isinstance(arg, List):
                currentArgs[i] = arg[j]
            elif arrayRanks[i] is not None:
                currentArgs[i] = materializedArgs[i][j]
            else:
                currentArgs[i] = arg

        argSet.append(tuple(currentArgs))
    return argSet
