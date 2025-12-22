# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime
from cudaq.kernel.kernel_decorator import (mk_decorator, isa_kernel_decorator)


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
          for the :class:`Kernel`.
    """
    if isa_kernel_decorator(kernel):
        decorator = kernel
    else:
        decorator = mk_decorator(kernel)
    specMod, processedArgs = decorator.handle_call_arguments(*args)
    returnTy = (decorator.returnType
                if decorator.returnType else decorator.get_none_type())
    choice = kwargs.get("choice", None)
    return cudaq_runtime.estimate_resources_impl(decorator.uniqName, specMod,
                                                 returnTy, choice,
                                                 *processedArgs)
