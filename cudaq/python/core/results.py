# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Result types for kernel launches.

Currently only the types for `cudaq.estimate` are exported here.
"""

# Re-export the existing C++ bindings from their shared extension; the types are
# registered there once, rather than separately for each programming model.
from cudaq.mlir._mlir_libs._backends import (
    EstimateResult,
    Resources,
)

__all__ = ["EstimateResult", "Resources"]
