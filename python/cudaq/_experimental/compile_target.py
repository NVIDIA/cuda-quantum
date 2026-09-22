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
runtime endpoint half of a :class:`CustomTarget`.

Build one and install it together with a runtime endpoint via
``cudaq.set_target``:

```python
import cudaq
from cudaq._experimental import CompileTarget, CustomTarget, RuntimeEndpoint

class MyEndpoint(RuntimeEndpoint):
    def sample(self, module, args, **kwargs):
        return cudaq.SampleResult({"00": kwargs["shots_count"]})

cudaq.set_target(CustomTarget(
    compile_target=CompileTarget(),
    runtime_endpoint=MyEndpoint(),
))
cudaq.sample(my_kernel)      # compiled with that pipeline, launched to endpoint
```

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

__all__ = [
    "CompileTarget",
    "CompiledModule",
    "PipelineConfig",
]
