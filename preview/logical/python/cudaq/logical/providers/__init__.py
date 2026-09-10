# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Stable callback ABI for serialized CUDA-Q Logical definitions.

This module deliberately contains no provider registry. Each exported object
is an ordinary compiler callback that a definition resolves by its versioned
Python entry point.
"""

from __future__ import annotations

from ..qec.lattice_surgery import mpp_compiler
from ..targets import mlir

__all__ = [
    "mlir",
    "mpp_compiler",
]
