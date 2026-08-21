# ============================================================================ #
# Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime
from cudaq.kernel.kernel_decorator import (mk_decorator, isa_kernel_decorator)
from cudaq.util import trace

EstimateResult = cudaq_runtime.EstimateResult

# We currently have two redundant APIs for resource estimation that only
# differ in the return type:
#  - the older `estimate_resources()` functions return `Resources` types
#    directly,
#  - the newer `estimate()` functions return `EstimateResult` types.
#
# The goal is to deprecate the `estimate_resources()` functions and migrate
# callers to the `estimate()` functions. However, the shape of
# `EstimateResult` has not settled yet, so we keep both APIs for now and will
# make a clean break once we are ready.


@trace.traced
def estimate(kernel, *args, **kwargs):
    """
    Performs resource counting on the given quantum kernel expression and
    returns an accounting of how many times each gate was applied, in addition
    to the total number of gates and qubits used.

    Args:
      choice (Any): A choice function called to determine the outcome of
          measurements, in case control flow depends on measurements. Should
          only return either `True` or `False`. Invoking the kernel within
          the choice function is forbidden. Default: returns `True` or `False`
          with 50% probability.
      kernel (:class:`Kernel`): The :class:`Kernel` to count resources on
      *arguments (Optional[Any]): The concrete values to evaluate the kernel
          function at. Leave empty if the kernel doesn't accept any arguments.
      **kwargs: Endpoint-specific options are forwarded unchanged to a
          configured runtime endpoint. This includes resource-estimation
          options such as ``tier``.

    Returns:
      :class:`cudaq.EstimateResult`: A data-type containing the resource count
          results for the :class:`Kernel`.
    """
    if isa_kernel_decorator(kernel):
        decorator = kernel
    else:
        decorator = mk_decorator(kernel)
    processedArgs, module = decorator.prepare_call(*args)
    choice = kwargs.get("choice", None)
    endpoint_options = {
        key: value for key, value in kwargs.items() if key != "choice"
    }
    return cudaq_runtime.estimate_impl(decorator.uniqName, module, choice,
                                       endpoint_options, *processedArgs)


@trace.traced
def estimate_resources(kernel, *args, **kwargs):
    """
    Performs resource counting on the given quantum kernel expression and
    returns an accounting of how many times each gate was applied, in addition
    to the total number of gates and qubits used.

    Args:
      choice (Any): A choice function called to determine the outcome of
          measurements, in case control flow depends on measurements. Should
          only return either `True` or `False`. Invoking the kernel within
          the choice function is forbidden. Default: returns `True` or `False`
          with 50% probability.
      kernel (:class:`Kernel`): The :class:`Kernel` to count resources on
      *arguments (Optional[Any]): The concrete values to evaluate the kernel
          function at. Leave empty if the kernel doesn't accept any arguments.

    Returns:
      :class:`Resources`:  A dictionary containing the resource count results
          for the :class:`Kernel`. Any annotation stored on the
          :class:`cudaq.EstimateResult` instance is discarded. Use
          :func:`cudaq.estimate` to get the full result. The returned instance
          corresponds to the `resources` attribute of the return value of
          :func:`cudaq.estimate`.
    """
    return estimate(kernel, *args, **kwargs).resources
