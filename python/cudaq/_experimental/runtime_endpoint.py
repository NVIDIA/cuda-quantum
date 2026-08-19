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
compiled -- that is specified by the active target.

An endpoint is any Python object implementing one or more of the
protocols below. Register it with :func:`set_runtime_endpoint`:
    
.. code-block:: python

    import cudaq
    from cudaq import SampleResult
    from cudaq._experimental import RuntimeEndpoint, set_runtime_endpoint

    class MyEndpoint(RuntimeEndpoint):

        def sample(self, module, arguments, *, shots_count, **options):
            submit_somewhere(module, list(arguments))
            return SampleResult({"00": shots_count})

    endpoint = MyEndpoint()
    set_runtime_endpoint(endpoint)
    cudaq.sample(my_kernel)      # dispatched to endpoint.sample

Calling ``cudaq.set_target(...)`` or ``cudaq.reset_target()`` replaces the
platform's QPUs and thereby removes the endpoint again; there is no separate
uninstall call.

.. warning::

   This API is experimental. There is currently no way for the runtime to check
   that the active target's compilation settings produce IR the endpoint
   understands. Mismatches between the target's compilation settings and the endpoint's
   requirements will result in hard to diagnose errors. Currently,
   the default local-simulator compile target is used whenever a custom
   runtime endpoint is registered.
"""

from typing import Protocol, runtime_checkable

from cudaq.mlir._mlir_libs._quakeDialects.cudaq_runtime import (
    CompiledModule,
    KernelArgs,
    ObserveResult,
    SampleResult,
    SpinOperator,
    set_runtime_endpoint,
)
import cudaq.mlir._mlir_libs._quakeDialects.cudaq_runtime as _cudaq_runtime

__all__ = [
    "CompiledModule",
    "KernelArgs",
    "ObserveResult",
    "SampleResult",
    "SpinOperator",
    "SupportsObserve",
    "SupportsSample",
    "set_runtime_endpoint",
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
    ```
    """

    is_simulator: bool = True
    is_remote: bool = False
    is_emulated: bool = False


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


_PROTOCOLS = (SupportsSample, SupportsObserve)
