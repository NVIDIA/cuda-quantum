# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from ..mlir._mlir_libs._quakeDialects import cudaq_runtime

# Enum to specify the initial quantum state.
InitialState = cudaq_runtime.InitialStateType

InitialStateArgT = cudaq_runtime.State | InitialState
