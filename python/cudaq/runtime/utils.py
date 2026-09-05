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
        if (isinstance(firstArg, list) or
                isinstance(firstArg, List)) and not firstArgTypeIsFlatSequence:
            return True

        if hasattr(firstArg, "shape"):
            shape = firstArg.shape
            if len(shape) == 1 and not firstArgTypeIsFlatSequence:
                return True

            if len(shape) == 2:
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
        if (isinstance(firstArg, list) or
                isinstance(firstArg, List)) and not firstArgTypeIsFlatSequence:
            return True

        if hasattr(firstArg, "shape"):
            shape = firstArg.shape
            if len(shape) == 1 and not firstArgTypeIsFlatSequence:
                return True

            if len(shape) == 2:
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
            handled = False

            if isinstance(arg, list) or isinstance(arg, List):
                currentArgs[i] = arg[j]
                handled = True

            if arrayRanks[i] is not None:
                currentArgs[i] = materializedArgs[i][j]
                handled = True

            if not handled:
                # A plain scalar argument (e.g. a fixed `int`/`float` kernel
                # parameter passed alongside a broadcast list/array) is not a
                # per-call value to index into; hold it constant across every
                # generated argument set instead of leaving the `0` placeholder.
                currentArgs[i] = arg

        argSet.append(tuple(currentArgs))
    return argSet
