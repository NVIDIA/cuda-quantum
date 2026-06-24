# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from ..mlir._mlir_libs._quakeDialects import cudaq_runtime
from enum import Enum

# Enum to specify the initial quantum state.
InitialState = cudaq_runtime.InitialStateType

InitialStateArgT = cudaq_runtime.State | InitialState


class IntermediateResultSave(Enum):
    '''
    Enum to specify how intermediate results should be saved during the dynamics evolution.
    '''
    # Options for saving intermediate results.
    # NONE: Do not save any intermediate results.
    NONE = 1
    # ALL: Save all intermediate results.
    ALL = 2
    # EXPECTATION_VALUE: Save only the expectation values of the observables.
    EXPECTATION_VALUE = 3
