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

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# Re-export the existing C++ bindings from their shared extension; the types are
# registered there once, rather than separately for each programming model.
from cudaq.mlir._mlir_libs._backends import (
    CompileTarget,
    EstimateResult,
    PipelineConfig,
    Resources,
)

__all__ = [
    "CompileTarget", "CustomTarget", "EstimateResult", "PipelineConfig",
    "Resources", "RuntimeEndpoint"
]


# This base protocol describes capabilities, not execution methods. Keeping it
# here lets CustomTarget expose concrete annotations without importing the
# frontend. Policy-specific protocols (SupportsSample, etc.) stay there.
@runtime_checkable
class RuntimeEndpoint(Protocol):
    """A runtime endpoint is a Python object that can serve kernel launches.

    Implement one or several of the children protocols for each supported
    launch policy.

    Although not required, it is recommended for user-defined endpoints to
    inherit explicitly from this base class. This ensures all default
    attribute values are inherited:

    ```python
    class MyEndpoint(RuntimeEndpoint):
        def sample(self, module, args, **kwargs):
            pass

    ep = MyEndpoint()
    print(ep.is_simulator)  # True
    print(ep.is_remote)    # False
    print(ep.is_emulated)  # False
    print(ep.supports_jit) # True
    ```

    Set ``supports_jit = False`` if the endpoint consumes the
    ``CompiledModule``'s MLIR artifact itself. The runtime then skips local
    code generation, which is otherwise built and discarded.
    """

    is_simulator: bool = True
    is_remote: bool = False
    is_emulated: bool = False
    supports_jit: bool = True


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
