# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime
from cudaq.mlir.ir import UnitAttr
from cudaq.kernel.kernel_decorator import (mk_decorator, isa_kernel_decorator)
import numpy as np

# Maintain a dictionary of queued `async` run kernels. This dictionary is used
# to keep the `mlir::ModuleOp` alive so the interpreter doesn't garbage collect
# them before they can be launched properly.
cudaq_async_run_module_cache = {}
cudaq_async_run_cache_counter = 0


class AsyncRunResult:

    def __init__(self, impl, mod):
        global cudaq_async_run_module_cache
        global cudaq_async_run_cache_counter
        self.impl = impl
        self.getCalled = False
        self.counter = cudaq_async_run_cache_counter
        cudaq_async_run_cache_counter = self.counter + 1
        cudaq_async_run_module_cache[self.counter] = mod

    def get(self):
        result = self.impl.get()
        self.getCalled = True
        return result

    def __del__(self):
        # FIXME: This potentially leaks memory intentionally. It is possible
        # that the AsyncRunResult object gets deleted *before* the `async` run
        # call occurs or finishes. In that case, we leave the module in the
        # dictionary to prevent the interpreter from crashing.
        # We ought to have a way to inform the C++ code that the result is no
        # longer being sought and the module and `py::handle` should be freed.
        if self.getCalled:
            del (cudaq_async_run_module_cache[self.counter])


def run(decorator, *args, shots_count=100, noise_model=None, qpu_id=0):
    if isa_kernel_decorator(decorator):
        if decorator.qkeModule is None:
            raise RuntimeError(
                "Unsupported target / Invalid kernel for `run`: missing module")

    if decorator.formal_arity() != len(args):
        raise RuntimeError("Invalid number of arguments passed to run. " +
                           str(len(args)) + " given and " +
                           str(decorator.formal_arity()) + " expected.")
    if (not isinstance(shots_count, int)) or (shots_count < 0):
        raise RuntimeError(
            "Invalid `shots_count`. Must be a non-negative number.")

    specMod, processedArgs = decorator.handle_call_arguments(*args)
    retTy = decorator.get_none_type()
    return cudaq_runtime.run_impl(decorator.uniqName + ".run", specMod, retTy,
                                  shots_count, noise_model, qpu_id,
                                  *processedArgs)


def run_async(decorator, *args, shots_count=100, noise_model=None, qpu_id=0):
    """
Run the provided `kernel` with the given kernel `arguments` over the specified
number of circuit executions (`shots_count`) asynchronously on the specified
`qpu_id`.

Args:
  kernel (:class:`Kernel`): The :class:`Kernel` to execute `shots_count` times
    on the QPU.
  *arguments (Optional[Any]): The concrete values to evaluate the kernel
    function at. Leave empty if the kernel doesn't accept any arguments. For
    example, if the kernel takes two `float` values as input, the `run` call
    should be structured as `cudaq.run(kernel, firstFloat, secondFloat)`.
  shots_count (Optional[int]): The number of kernel executions on the QPU.
    Defaults to 100. Key-word only.
  noise_model (Optional[`NoiseModel`]): The optional :class:`NoiseModel` to add
    noise to the kernel execution on the simulator. Defaults to an empty noise
    model.
  `qpu_id` (Optional[int]): The id of the QPU. Defaults to 0. Key-word only.

Returns:
  `AsyncRunResult`:
  A handle, which can be waited on via a `get()` method, which returns an array
  of `kernel` return values. The length of the list is equal to `shots_count`.
  """

    if isa_kernel_decorator(decorator):
        if decorator.qkeModule is None:
            raise RuntimeError(
                "Unsupported target / Invalid kernel for `run`: missing module")

    if decorator.formal_arity() != len(args):
        raise RuntimeError("Invalid number of arguments passed to run_async. " +
                           str(len(args)) + " given and " +
                           str(decorator.formal_arity()) + " expected.")
    if (not isinstance(shots_count, int)) or (shots_count < 0):
        raise RuntimeError(
            "Invalid `shots_count`. Must be a non-negative number.")

    target = cudaq_runtime.get_target()
    num_qpus = target.num_qpus()
    if qpu_id >= num_qpus:
        raise ValueError(f"qpu_id ({qpu_id}) exceeds the number of available "
                         f"QPUs ({num_qpus}).")

    if noise_model != None:
        if target.is_remote_simulator() or target.is_remote():
            raise ValueError("Noise model is not supported on remote simulator"
                             " or hardware QPU.")

    specMod, processedArgs = decorator.handle_call_arguments(*args)
    retTy = decorator.get_none_type()
    async_results = cudaq_runtime.run_async_impl(decorator.uniqName + ".run",
                                                 specMod, retTy, shots_count,
                                                 noise_model, qpu_id,
                                                 *processedArgs)
    return AsyncRunResult(async_results, specMod)
