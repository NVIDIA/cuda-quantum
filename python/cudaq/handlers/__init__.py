# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime
from .target_handler import DefaultTargetHandler, PhotonicsTargetHandler

# Registry of target handlers
TARGET_HANDLERS = {'orca-photonics': PhotonicsTargetHandler()}


def get_target_handler():
    """Get the appropriate target handler based on current target"""
    try:
        target_name = cudaq_runtime.get_target().name
        return TARGET_HANDLERS.get(target_name, DefaultTargetHandler())
    except RuntimeError:
        return DefaultTargetHandler()
