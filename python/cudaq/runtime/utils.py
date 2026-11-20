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


def __rejectScalarBroadcastArgs(*args):
    # Same check as the 2D-array path above for broadcasts triggered by a
    # 1D list/array argument.
    for i, arg in enumerate(args):
        if isinstance(arg, (int, float, bool, str, complex, np.generic)):
            raise RuntimeError(
                f"Argument {i} ({type(arg)}) must be a list or ndarray for sample or observe broadcasting."
            )


def __isBroadcast(kernel, *args):
    # kernel could be a PyKernel or kernel decorator
    if isinstance(kernel, PyKernel):
        argTypes = kernel.mlirArgTypes
        if len(argTypes) == 0 or len(args) == 0:
            return False

        # Quick check, if we have a 2d array anywhere, we know this is a broadcast
        isDefinitelyBroadcast = True in [
            hasattr(arg, "shape") and len(arg.shape) == 2 for arg in args
        ]

        if isDefinitelyBroadcast:
            # Error check, did the user pass a single value for any of the other arguments
            for i, arg in enumerate(args):
                if isinstance(arg, (int, float, bool, str)):
                    raise RuntimeError(
                        f"2D array argument provided for an sample or observe broadcast, but argument {i} ({type(arg)}) must be a list."
                    )

        firstArg = args[0]
        firstArgTypeIsFlatSequence = cc.SequenceType.isinstance(argTypes[0])
        isBroadcast = isDefinitelyBroadcast or (
            (isinstance(firstArg, list) or isinstance(firstArg, List)) and
            not firstArgTypeIsFlatSequence) or (hasattr(firstArg, "shape") and (
                (len(firstArg.shape) == 1 and not firstArgTypeIsFlatSequence) or
                len(firstArg.shape) == 2))

        if isBroadcast and not isDefinitelyBroadcast:
            __rejectScalarBroadcastArgs(*args)

        if isBroadcast:
            return True

        return False

    elif isa_kernel_decorator(kernel):
        argTypes = kernel.arg_types()
        if len(argTypes) == 0 or len(args) == 0:
            return False

        # Quick check, if we have a 2d array anywhere, we know this
        # is a broadcast
        isDefinitelyBroadcast = True in [
            hasattr(arg, "shape") and len(arg.shape) == 2 for arg in args
        ]

        if isDefinitelyBroadcast:
            # Error check, did the user pass a single value for any of the other arguments
            for i, arg in enumerate(args):
                if isinstance(arg, (int, float, bool, str)):
                    raise RuntimeError(
                        f"2D array argument provided for an observe broadcast, but argument {i} ({type(arg)}) must be a list."
                    )

        firstArg = args[0]
        firstArgTypeIsFlatSequence = cc.SequenceType.isinstance(argTypes[0])
        isBroadcast = isDefinitelyBroadcast or (
            (isinstance(firstArg, list) or isinstance(firstArg, List)) and
            not firstArgTypeIsFlatSequence) or (hasattr(firstArg, "shape") and (
                (len(firstArg.shape) == 1 and not firstArgTypeIsFlatSequence) or
                len(firstArg.shape) == 2))

        if isBroadcast and not isDefinitelyBroadcast:
            __rejectScalarBroadcastArgs(*args)

        if isBroadcast:
            return True

        return False


def __createArgumentSet(*args):
    nArgSets = len(args[0])
    if nArgSets == 0:
        return []

    # Materialize array-like arguments once.
    materializedArgs = []
    arrayRanks = []
    for arg in args:
        if hasattr(arg, "tolist"):
            arrayRank = len(arg.shape)
            arrayRanks.append(arrayRank)
            # A later matrix argument may have more rows than the first
            # argument that defines `nArgSets`. Do not materialize unused rows.
            m = arg[:nArgSets] if arrayRank == 2 else arg
            materializedArgs.append(m.tolist())
        else:
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
