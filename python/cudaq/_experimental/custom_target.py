# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Pair a compile target with a runtime endpoint.

A custom target combines the two halves of a target config in one definition: a
compile target that defines how kernels are compiled and a runtime endpoint that
defines where they run. Install one with :func:`cudaq.set_target`:

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

.. warning::

   This API is experimental. There is currently no check that the compile target
   and runtime endpoint are compatible; mismatches surface as hard to diagnose
   compilation or execution errors.
"""

from dataclasses import dataclass

from .compile_target import CompileTarget
from .runtime_endpoint import RuntimeEndpoint

__all__ = [
    "CustomTarget",
]


@dataclass
class CustomTarget:
    """A compile target and runtime endpoint installed together.

    Args:
      runtime_endpoint: The endpoint that receives compiled kernels.
      compile_target: The machine model kernels are compiled against.
    """

    runtime_endpoint: RuntimeEndpoint
    compile_target: CompileTarget
