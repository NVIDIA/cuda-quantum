# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Define a CUDA-Q runtime endpoint in Python.

A runtime endpoint is the *launch* half of a backend: it receives an already
compiled kernel and executes it. It says nothing about how the kernel was
compiled -- that is specified by the compile target half of a
:class:`CustomTarget`.

An endpoint is any Python object implementing one or more of the
protocols below. Install it together with a compile target via
``cudaq.set_target``:

```python
import cudaq
from cudaq import SampleResult
from cudaq._experimental import CompileTarget, CustomTarget, RuntimeEndpoint

class MyEndpoint(RuntimeEndpoint):

    def sample(self, module, arguments, *, shots_count, **options):
        submit_somewhere(module, list(arguments))
        return SampleResult({"00": shots_count})

cudaq.set_target(CustomTarget(
    compile_target=CompileTarget(),
    runtime_endpoint=MyEndpoint(),
))
cudaq.sample(my_kernel)      # dispatched to endpoint.sample
```

Calling ``cudaq.set_target(...)`` or ``cudaq.reset_target()`` replaces the
platform's QPUs and thereby removes the endpoint again; there is no separate
uninstall call.

.. warning::

   This API is experimental. There is currently no way for the runtime to check
   that the compile target and runtime endpoint are compatible. Mismatches
   between the two will result in hard to diagnose errors.
"""

from typing import Protocol, runtime_checkable

from cudaq.mlir._mlir_libs._quakeDialects.cudaq_runtime import (
    CompiledModule,
    DEMResult,
    EstimateResult,
    KernelArgs,
    NoiseModel,
    ObserveResult,
    SampleResult,
    SpinOperator,
)

__all__ = [
    "CompiledModule",
    "DEMResult",
    "EstimateResult",
    "KernelArgs",
    "NoiseModel",
    "ObserveResult",
    "SampleResult",
    "SpinOperator",
    "SupportsDem",
    "SupportsEstimate",
    "SupportsObserve",
    "SupportsSample",
]


@runtime_checkable
class RuntimeEndpoint(Protocol):
    """A runtime endpoint is a Python object that can serve kernel launches.
    
    Implement one or several of the children protocols for each supported
    launch policy.
    
    Although not required, it is recommended for user-defined endpoints to
    inherit explicitly from this base class. This ensures all default
    attributes values are inherited:

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


@runtime_checkable
class SupportsSample(RuntimeEndpoint, Protocol):
    """An endpoint that can serve ``cudaq.sample``."""

    def sample(self, module: CompiledModule, args: KernelArgs,
               **kwargs) -> SampleResult:
        """Execute a compiled kernel and return its measurement counts.

        Keyword arguments: ``shots_count`` (int) and ``explicit_measurements``
        (bool).
        """
        ...


@runtime_checkable
class SupportsObserve(RuntimeEndpoint, Protocol):
    """An endpoint that can serve ``cudaq.observe``."""

    def observe(self, module: CompiledModule, args: KernelArgs,
                **kwargs) -> ObserveResult:
        """Execute a compiled kernel and return an expectation value.

        Keyword arguments: ``spin_operator`` (:class:`SpinOperator`) and, when
        the launch specifies one, ``shots_count`` (int).
        """
        ...


@runtime_checkable
class SupportsDem(RuntimeEndpoint, Protocol):
    """An endpoint that can serve ``cudaq.dem_from_kernel``."""

    def dem_from_kernel(self, module: CompiledModule, args: KernelArgs,
                        **kwargs) -> DEMResult:
        """Analyze a compiled kernel and return its detector error model.

        Keyword arguments: ``noise_model`` (:class:`NoiseModel` or ``None``)
        and the DEM options accepted by ``dem_from_kernel`` --
        ``decompose_errors`` (bool), ``fold_loops`` (bool),
        ``allow_gauge_detectors`` (bool),
        ``approximate_disjoint_errors_threshold`` (float),
        ``ignore_decomposition_failures`` (bool),
        ``block_decomposition_from_introducing_remnant_edges`` (bool) and
        ``return_measurement_matrices`` (bool).
        """
        ...


@runtime_checkable
class SupportsEstimate(RuntimeEndpoint, Protocol):
    """An endpoint that can serve ``cudaq.estimate``."""

    def estimate(self, module: CompiledModule, args: KernelArgs,
                 **kwargs) -> EstimateResult:
        """Estimate the resources a compiled kernel would use.

        Keyword arguments: ``choice``, a callable returning a `bool` that
        resolves each measurement so that kernels branching on measurement
        results take a definite path.
        """
        ...


_PROTOCOLS = (SupportsSample, SupportsObserve, SupportsDem, SupportsEstimate)
