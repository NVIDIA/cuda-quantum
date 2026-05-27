# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime
from cudaq.kernel.kernel_decorator import (mk_decorator, isa_kernel_decorator)
from cudaq.util import trace
from .utils import _kernel_has_conditionals_on_measure


def _detail_check_conditionals_on_measure(kernel):
    if not _kernel_has_conditionals_on_measure(kernel):
        return
    kernel_name = kernel.name if hasattr(kernel, 'name') else '<unknown>'
    raise RuntimeError(
        f"`cudaq::dem_from_kernel`: kernel '{kernel_name}' branches on "
        "a measurement result. DEM analysis not supported.")


@trace.traced
def dem_from_kernel(kernel, *args, noise_model=None):
    """Generate a detector error model (DEM) from a CUDA-Q kernel.

    Runs `kernel` under the internal `"dem"` execution context, captures
    the recorded circuit from the backend, and returns Stim's standard
    `.dem` text via `stim::DetectorErrorModel::str()`. The active CUDA-Q
    target is unaffected; the analysis simulator is an internal,
    thread-local override.

    Args:
      kernel (:class:`Kernel`): The :class:`Kernel` to analyze.
      *arguments: Concrete argument values forwarded to the kernel invocation.
      noise_model (:class:`NoiseModel`, optional): Noise model layered on
          top of any `apply_noise` ops already present in the kernel.

    Returns:
      UTF-8 string in Stim's standard `.dem` file format. Consumers
      that need a structured DEM can parse it with
      `stim.DetectorErrorModel(text)`.
    """
    _detail_check_conditionals_on_measure(kernel)

    if isa_kernel_decorator(kernel):
        decorator = kernel
    else:
        decorator = mk_decorator(kernel)
    processedArgs, module = decorator.prepare_call(*args)
    return cudaq_runtime.dem_from_kernel_impl(decorator.uniqName, module,
                                              noise_model, *processedArgs)
