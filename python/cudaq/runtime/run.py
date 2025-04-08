# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime
from cudaq.mlir.ir import UnitAttr
import numpy as np


def run(kernel, *args, shots_count=1, noise_model=None):
    """Run the provided `kernel` at the given kernel 
`arguments` over the specified number of circuit executions (`shots_count`). 

Args:
  kernel (:class:`Kernel`): The :class:`Kernel` to execute `shots_count`
    times on the QPU.
  *arguments (Optional[Any]): The concrete values to evaluate the kernel
    function at. Leave empty if the kernel doesn't accept any arguments. For 
    example, if the kernel takes two `float` values as input, the `sample` call 
    should be structured as `cudaq.run(kernel, firstFloat, secondFloat)`.
  shots_count (Optional[int]): The number of kernel executions on the QPU.
    Defaults to 1. Key-word only.
  noise_model (Optional[`NoiseModel`]): The optional :class:`NoiseModel`
    to add noise to the kernel execution on the simulator. Defaults to
    an empty noise model.

Returns:
  `numpy.array[Any]`: 
  An array of `kernel` return values. The length of the list is equal to `shots_count`."""
    kernel.enable_return_to_log()
    if kernel.returnType is None:
        raise ValueError("cudaq.run only supports kernels that return values.")

    if shots_count < 0:
        raise ValueError("Invalid shots_count. Must be non-negative.")

    if shots_count == 0:
        return np.array([])

    # Default construct the result array (allocate memory buffer)
    results = np.array([kernel.returnType() for _ in range(shots_count)])

    target = cudaq_runtime.get_target()

    if noise_model != None:
        if target.is_remote_simulator() or target.is_remote():
            raise ValueError(
                "Noise model is not supported on remote simulator or hardware QPU"
            )

        cudaq_runtime.set_noise(noise_model)

    if target.is_remote_simulator() or target.is_remote() or target.is_emulated(
    ):
        ctx = cudaq_runtime.ExecutionContext("run", shots_count)
        cudaq_runtime.setExecutionContext(ctx)
        kernel(*args)
        cudaq_runtime.resetExecutionContext()
        cudaq_runtime.decodeQirOutputLog(''.join(ctx.invocationResultBuffer),
                                         results)
    else:
        ctx = cudaq_runtime.ExecutionContext("run", 1)
        for i in range(shots_count):
            cudaq_runtime.setExecutionContext(ctx)
            kernel(*args)
            cudaq_runtime.resetExecutionContext()

        cudaq_runtime.decodeQirOutputLog(cudaq_runtime.getQirOutputLog(),
                                         results)
        cudaq_runtime.clearQirOutputLog()

    cudaq_runtime.unset_noise()
    return results


def run_async(kernel, *args, shots_count=1, noise_model=None, qpu_id=0):
    """Run the provided `kernel` at the given kernel 
`arguments` over the specified number of circuit executions (`shots_count`) asynchronously on the specified `qpu_id`. 

Args:
  kernel (:class:`Kernel`): The :class:`Kernel` to execute `shots_count`
    times on the QPU.
  *arguments (Optional[Any]): The concrete values to evaluate the kernel
    function at. Leave empty if the kernel doesn't accept any arguments. For 
    example, if the kernel takes two `float` values as input, the `sample` call 
    should be structured as `cudaq.run(kernel, firstFloat, secondFloat)`.
  shots_count (Optional[int]): The number of kernel executions on the QPU.
    Defaults to 1. Key-word only.
  noise_model (Optional[`NoiseModel`]): The optional :class:`NoiseModel`
    to add noise to the kernel execution on the simulator. Defaults to
    an empty noise model.
  qpu_id (Optional[int]): The id of the QPU.
    Defaults to 0. Key-word only.

Returns:
  `AsyncRunResult`: 
  An async. handle, which can be waited on via a `get()` method, which returns an array of `kernel` return values. The length of the list is equal to `shots_count`.
  """
    kernel.enable_return_to_log()
    if kernel.returnType is None:
        raise ValueError("cudaq.run only supports kernels that return values.")

    if shots_count < 0:
        raise ValueError("Invalid shots_count. Must be non-negative.")

    if shots_count == 0:
        return np.array([])

    # Default construct the result array (allocate memory buffer)
    results = np.array([kernel.returnType() for _ in range(shots_count)])

    async_results = cudaq_runtime.run_async_internal(results,
                                                     kernel,
                                                     *args,
                                                     shots_count=shots_count,
                                                     qpu_id=qpu_id)

    return async_results
