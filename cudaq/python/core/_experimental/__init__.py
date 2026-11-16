# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Experimental CUDA-Q types available without the execution frontend.

These core target types can be used to define custom targets, compilation
pipelines and runtime endpoints without loading the execution frontend. Frontend
APIs re-export the same types to preserve class identity across both import
paths.

.. warning::

   This API is experimental. There is currently no check that the compile target
   and runtime endpoint are compatible; mismatches surface as hard to diagnose
   compilation or execution errors.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from cudaq.core._target import (Codegen, CompilationStage, CompileTarget,
                                CostModel, CustomPipelineLowering,
                                LoopUnrolling, NoiseSupport, QuakeMappedProfile,
                                QuakeReferenceSemanticsProfile,
                                QuakeUnmappedProfile, UncheckedProfile)

__all__ = [
    "Codegen",
    "CompilationStage",
    "CompileTarget",
    "CostModel",
    "CustomPipelineLowering",
    "CustomTarget",
    "LoopUnrolling",
    "NoiseSupport",
    "QuakeMappedProfile",
    "QuakeReferenceSemanticsProfile",
    "QuakeUnmappedProfile",
    "RuntimeEndpoint",
    "UncheckedProfile",
]


# This base protocol describes capabilities, not execution methods. Keeping it
# here lets CustomTarget expose concrete annotations without importing the
# frontend. Policy-specific protocols (`SupportsSample`, etc.) stay there.
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

    A custom target combines the two halves of a target config in one
    definition: a compile target that defines how kernels are compiled and a
    runtime endpoint that defines where they run. Install one with
    :func:`cudaq.set_target`:

    ```python
    import cudaq
    from dataclasses import dataclass, field

    from cudaq._experimental import CompileTarget, CustomTarget, RuntimeEndpoint


    class MyEndpoint(RuntimeEndpoint):

        def sample(self, module, args, **kwargs):
            return cudaq.SampleResult({"00": kwargs["shots_count"]})


    @dataclass
    class MyCustomTarget(CustomTarget):
        runtime_endpoint: RuntimeEndpoint = field(default_factory=MyEndpoint)
        compile_target: CompileTarget = field(default_factory=CompileTarget)


    cudaq.set_target(MyCustomTarget())
    cudaq.sample(my_kernel)
    ```

    You must currently mark any custom class explicitly as a valid target by
    inheriting from :class:`CustomTarget`. Otherwise, :func:`cudaq.set_target`
    will throw an error.

    Args:
      runtime_endpoint: The endpoint that receives compiled kernels.
      compile_target: The machine model kernels are compiled against.
    """

    runtime_endpoint: RuntimeEndpoint
    compile_target: CompileTarget
