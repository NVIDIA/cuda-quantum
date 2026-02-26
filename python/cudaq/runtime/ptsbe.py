# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime
from cudaq.kernel.kernel_decorator import (mk_decorator, isa_kernel_decorator)
from cudaq.runtime.sample import (_detail_check_conditionals_on_measure,
                                  AsyncSampleResult)
from .utils import __isBroadcast, __createArgumentSet

from cudaq.mlir._mlir_libs._quakeDialects.cudaq_runtime.ptsbe import *


class AsyncPTSBESampleResult(AsyncSampleResult):
    """Async result for PTSBE sampling.

    Extends AsyncSampleResult with a reference to the noise_model so the
    Python object is not garbage-collected while the async C++ work is
    in flight. The reference is released when .get() is called.
    """

    def __init__(self, impl, mod, noise_model):
        super().__init__(impl, mod)
        self._noise_model = noise_model

    def get(self):
        result = super().get()
        self._noise_model = None
        return result


def _validate_ptsbe_args(kernel, args, shots_count, noise_model,
                         max_trajectories):
    """Validate arguments common to `sample` and `sample_async`."""
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

    _detail_check_conditionals_on_measure(decorator)

    return decorator


def sample(kernel,
           *args,
           shots_count=1000,
           noise_model=None,
           max_trajectories=None,
           sampling_strategy=None,
           shot_allocation=None,
           return_execution_data=False):
    """
    Sample using Pre-Trajectory Sampling with Batch Execution (`PTSBE`).

    Pre-samples noise realizations (trajectories) and batches circuit
    executions by unique noise configuration, enabling efficient noisy
    sampling of many shots.

    When called with list arguments (broadcast mode), executes the kernel
    for each set of arguments and returns a list of results.

    Args:
      kernel: The quantum kernel to execute.
      shots_count (int): Number of measurement shots. Defaults to 1000.
      noise_model: Optional noise model for gate-based noise. Noise can also
          be specified inside the kernel via ``cudaq.apply_noise()``; both
          can be used together.
      max_trajectories (int or ``None``): Maximum unique trajectories to
          generate. ``None`` means use the number of shots. Note for large
          shot counts setting a maximum is recommended to get the benefits
          of PTS.
      sampling_strategy (``PTSSamplingStrategy`` or ``None``): Strategy for
          trajectory generation. ``None`` uses the default probabilistic
          sampling strategy.
      shot_allocation (``ShotAllocationStrategy`` or ``None``): Strategy for
          allocating shots across trajectories. ``None`` uses the default
          proportional (weight-based) allocation.
      return_execution_data (bool): Include circuit structure, trajectory
          specifications, and per-trajectory measurement outcomes in the
          returned result. Defaults to ``False``.

    Returns:
      ``SampleResult``: Measurement results. Returns a list of results
          in broadcast mode.

    Raises:
      RuntimeError: If the kernel is invalid or arguments are invalid.
    """
    decorator = _validate_ptsbe_args(kernel, args, shots_count, noise_model,
                                     max_trajectories)

    if noise_model is None:
        noise_model = cudaq_runtime.NoiseModel()

    if __isBroadcast(decorator, *args):
        argSets = __createArgumentSet(*args)
        results = []
        for argSet in argSets:
            processedArgs, module = decorator.prepare_call(*argSet)
            retTy = decorator.get_none_type()
            result = cudaq_runtime.ptsbe.sample_impl(
                decorator.uniqName, module, retTy, shots_count, noise_model,
                max_trajectories, sampling_strategy, shot_allocation,
                return_execution_data, *processedArgs)
            results.append(result)
        return results

    processedArgs, module = decorator.prepare_call(*args)
    retTy = decorator.get_none_type()

    return cudaq_runtime.ptsbe.sample_impl(decorator.uniqName, module, retTy,
                                           shots_count, noise_model,
                                           max_trajectories, sampling_strategy,
                                           shot_allocation,
                                           return_execution_data,
                                           *processedArgs)


def sample_async(kernel,
                 *args,
                 shots_count=1000,
                 noise_model=None,
                 max_trajectories=None,
                 sampling_strategy=None,
                 shot_allocation=None,
                 return_execution_data=False):
    """
    Asynchronously sample using PTSBE. Returns a future whose result
    can be retrieved via ``.get()``.

    Args:
      kernel: The quantum kernel to execute.
      shots_count (int): Number of measurement shots. Defaults to 1000.
      noise_model: Optional noise model for gate-based noise; noise can also
          be specified in the kernel via ``cudaq.apply_noise()``.
      max_trajectories (int or ``None``): Maximum unique trajectories.
      sampling_strategy (``PTSSamplingStrategy`` or ``None``): Strategy for
          trajectory generation.
      shot_allocation (``ShotAllocationStrategy`` or ``None``): Strategy for
          allocating shots across trajectories.
      return_execution_data (bool): Include execution data in the result.

    Returns:
      ``AsyncSampleResult``: A future whose ``.get()`` returns the
          ``SampleResult``.

    Raises:
      RuntimeError: If the kernel is invalid or arguments are invalid.
    """
    decorator = _validate_ptsbe_args(kernel, args, shots_count, noise_model,
                                     max_trajectories)

    if noise_model is None:
        noise_model = cudaq_runtime.NoiseModel()

    processedArgs, module = decorator.prepare_call(*args)
    retTy = decorator.get_none_type()

    impl = cudaq_runtime.ptsbe.sample_async_impl(
        decorator.uniqName, module, retTy, shots_count, noise_model,
        max_trajectories, sampling_strategy, shot_allocation,
        return_execution_data, *processedArgs)

    return AsyncPTSBESampleResult(impl, module, noise_model)
