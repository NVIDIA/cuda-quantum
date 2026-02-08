# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.    #
# ============================================================================ #

from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime
from cudaq.kernel.kernel_decorator import (mk_decorator, isa_kernel_decorator)

from cudaq.mlir._mlir_libs._quakeDialects.cudaq_runtime.ptsbe import *


def sample(kernel,
           *args,
           shots_count=1000,
           noise_model=None,
           max_trajectories=None,
           sampling_strategy=None):
    """
    Sample using Pre-Trajectory Sampling with Batch Execution (`PTSBE`).

    Pre-samples noise realizations (trajectories) and batches circuit
    executions by unique noise configuration, enabling efficient noisy
    sampling of many shots.

    Args:
      kernel: The quantum kernel to execute.
      shots_count (int): Number of measurement shots. Defaults to 1000.
      noise_model: The noise model to apply (required).
      max_trajectories (int or ``None``): Maximum unique trajectories to
          generate. ``None`` means use the number of shots. Note for large
          shot counts setting a maximum is recommended to get the benefits
          of PTS.
      sampling_strategy (``PTSSamplingStrategy`` or ``None``): Strategy for
          trajectory generation. ``None`` uses the default probabilistic
          sampling strategy.

    Returns:
      ``SampleResult``: Measurement results.

    Raises:
      RuntimeError: If ``noise_model`` is not provided or kernel is invalid.
    """
    if noise_model is None:
        raise RuntimeError(
            "PTSBE requires a noise_model. Pass noise_model=... to "
            "cudaq.ptsbe.sample().")

    decorator = kernel
    if not isa_kernel_decorator(decorator):
        decorator = mk_decorator(decorator)

    if isa_kernel_decorator(decorator):
        if decorator.qkeModule is None:
            raise RuntimeError(
                "Unsupported target / Invalid kernel for `ptsbe.sample`: "
                "missing module")

    if decorator.formal_arity() != len(args):
        raise RuntimeError(
            "Invalid number of arguments passed to ptsbe.sample. " +
            str(len(args)) + " given and " + str(decorator.formal_arity()) +
            " expected.")

    if (not isinstance(shots_count, int)) or (shots_count < 0):
        raise RuntimeError(
            "Invalid `shots_count`. Must be a non-negative integer.")

    if max_trajectories is not None:
        if (not isinstance(max_trajectories, int)) or (max_trajectories < 1):
            raise RuntimeError(
                "Invalid `max_trajectories`. Must be a positive integer.")

    specMod, processedArgs = decorator.handle_call_arguments(*args)
    retTy = decorator.get_none_type()

    return cudaq_runtime.ptsbe.sample_impl(decorator.uniqName, specMod, retTy,
                                           shots_count, noise_model,
                                           max_trajectories, sampling_strategy,
                                           *processedArgs)
