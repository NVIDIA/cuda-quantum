# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from enum import Enum

from ..mlir._mlir_libs._quakeDialects import cudaq_runtime


class InitialState(Enum):
    """
    Enum to specify the initial quantum state.
    """
    ZERO = 1
    UNIFORM = 2


InitialStateArgT = cudaq_runtime.State | InitialState
