# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Define a CUDA-Q compile target in Python.

A compile target is the *compilation* half of a backend: it fixes the MLIR pass
pipelines, the code generation and the capabilities that kernels are compiled
against. It says nothing about where the compiled kernel runs -- that is the
active target's QPU, or a :func:`set_runtime_endpoint` endpoint.

Build one and register it with :func:`set_compile_target`:

```python
import cudaq
from cudaq._experimental import CompileTarget, set_compile_target

target = CompileTarget()
target.pipeline_config.override_pass_pipeline = (
    "canonicalize,decomposition{enable-patterns=SwapToCX},canonicalize")

set_compile_target(target)
cudaq.sample(my_kernel)      # compiled with that pipeline
```

Setting a compile target manually overrides the QPU-provided compile target that
would have been used otherwise. To revert to a QPU-provided compile target, call
``cudaq.set_target('target-name')`` or ``cudaq.reset_target()``.

.. warning::

   This API is experimental. Nothing checks that the pipeline you configure
   produces IR that the backend understands; mismatches surface as hard to
   diagnose compilation or execution errors.
"""

from cudaq.mlir._mlir_libs._quakeDialects.cudaq_runtime import (
    CompileTarget,
    CompiledModule,
    PipelineConfig,
)
import cudaq.mlir._mlir_libs._quakeDialects.cudaq_runtime as _cudaq_runtime

__all__ = [
    "CompileTarget",
    "CompiledModule",
    "PipelineConfig",
    "set_compile_target",
]


def set_compile_target(target: CompileTarget) -> None:
    """Compile kernels with `target` instead of the active target's own.

    Args:
      target: The :class:`CompileTarget` to compile subsequent kernel launches
        with.

    Raises:
      TypeError: If `target` is not a :class:`CompileTarget`.
    """
    if not isinstance(target, CompileTarget):
        raise TypeError(
            f"{type(target).__name__} is not a compile target: expected a "
            f"cudaq._experimental.CompileTarget.")
    _cudaq_runtime.set_compile_target(target)
