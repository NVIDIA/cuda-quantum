# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Target descriptions and resource estimates shared by programming models.

Logical compilation and resource estimation use these types without loading the
execution frontend. Frontend APIs re-export the same types to preserve class
identity across both import paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

# Re-export the existing C++ bindings from their shared extension; the types are
# registered there once, rather than separately for each programming model.
from cudaq.mlir._mlir_libs._backends import (
    CompileTarget,
    EstimateResult,
    PipelineConfig,
    Resources,
)

# RuntimeEndpoint belongs to the execution frontend. It is needed here only for
# type checking; postponed annotations avoid importing that frontend at runtime.
if TYPE_CHECKING:
    from cudaq._experimental.runtime_endpoint import RuntimeEndpoint

__all__ = [
    "CompileTarget", "CustomTarget", "EstimateResult", "PipelineConfig",
    "Resources"
]


@dataclass
class CustomTarget:
    """A compile target and runtime endpoint installed together.

    Defined here so logical targets can use this container without importing
    the frontend. cudaq._experimental.custom_target re-exports this same class.

    Args:
      runtime_endpoint: The endpoint that receives compiled kernels.
      compile_target: The machine model kernels are compiled against.
    """

    runtime_endpoint: RuntimeEndpoint
    compile_target: CompileTarget
