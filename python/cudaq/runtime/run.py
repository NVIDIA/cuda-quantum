# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime
from cudaq.mlir.ir import UnitAttr
import numpy as np


def run_async(kernel, *args, shots_count=100, noise_model=None, qpu_id=0):
    """Run the provided `kernel` with the given kernel
`arguments` over the specified number of circuit executions
(`shots_count`) asynchronously on the specified `qpu_id`.

Args:
  kernel (:class:`Kernel`): The :class:`Kernel` to execute `shots_count`
    times on the QPU.
  *arguments (Optional[Any]): The concrete values to evaluate the kernel
    function at. Leave empty if the kernel doesn't accept any arguments. For
    example, if the kernel takes two `float` values as input, the `run` call
    should be structured as `cudaq.run(kernel, firstFloat, secondFloat)`.
  shots_count (Optional[int]): The number of kernel executions on the QPU.
    Defaults to 100. Key-word only.
  noise_model (Optional[`NoiseModel`]): The optional :class:`NoiseModel`
    to add noise to the kernel execution on the simulator. Defaults to
    an empty noise model.
  `qpu_id` (Optional[int]): The id of the QPU.
    Defaults to 0. Key-word only.

Returns:
  `AsyncRunResult`:
  A handle, which can be waited on via a `get()` method, which returns an array
  of `kernel` return values. The length of the list is equal to `shots_count`.
  """
    if kernel.returnType is None:
        raise RuntimeError(
            "`cudaq.run` only supports kernels that return a value.")

    if shots_count < 0:
        raise RuntimeError(
            "Invalid `shots_count`. Must be a non-negative number.")

    target = cudaq_runtime.get_target()
    num_qpus = target.num_qpus()
    if qpu_id >= num_qpus:
        raise ValueError(
            f"qpu_id ({qpu_id}) exceeds the number of available QPUs ({num_qpus})."
        )

    if noise_model != None:
        if target.is_remote_simulator() or target.is_remote():
            raise ValueError(
                "Noise model is not supported on remote simulator or hardware QPU"
            )

    async_results = cudaq_runtime.run_async_internal(kernel,
                                                     *args,
                                                     shots_count=shots_count,
                                                     noise_model=noise_model,
                                                     qpu_id=qpu_id)
    return async_results
