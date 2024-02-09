# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from ..mlir._mlir_libs._quakeDialects import cudaq_runtime


def get_state(kernel, *args):
    import numpy as np
    ctx = cudaq_runtime.ExecutionContext("extract-state")
    cudaq_runtime.setExecutionContext(ctx)
    kernel(*args)
    cudaq_runtime.resetExecutionContext()
    res = ctx.simulationData
    return np.reshape(np.asarray(res[1]), res[0])
