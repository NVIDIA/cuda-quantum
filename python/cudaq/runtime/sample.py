# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime
from cudaq.kernel.kernel_builder import PyKernel
from cudaq.kernel.kernel_decorator import (mk_decorator, isa_kernel_decorator)
from cudaq.kernel.utils import nvqppPrefix
from .utils import __isBroadcast, __createArgumentSet

# Maintain a dictionary of queued `async` sample kernels.This dictionary is used
# to keep the `mlir::ModuleOp` alive so the interpreter doesn't garbage collect
# them before they can be launched properly.
cudaq_async_sample_module_cache = {}
cudaq_async_sample_cache_counter = 0


class AsyncSampleResult:

    def __init__(self, *args, **kwargs):
        if len(args) == 2 and isinstance(args[0],
                                         cudaq_runtime.AsyncSampleResultImpl):
            impl = args[0]
            mod = args[1]
            global cudaq_async_sample_module_cache
            global cudaq_async_sample_cache_counter
            self.impl = impl
            self.getCalled = False
            self.counter = cudaq_async_sample_cache_counter
            cudaq_async_sample_cache_counter = self.counter + 1
            cudaq_async_sample_module_cache[self.counter] = mod
        elif len(args) == 1 and isinstance(args[0], str):
            # String-based constructor from JSON
            self.impl = cudaq_runtime.AsyncSampleResultImpl(args[0])
            self.counter = None
        else:
            raise RuntimeError(
                "Invalid arguments passed to AsyncSampleResult constructor.")

    def get(self):
        result = self.impl.get()
        self.getCalled = True
        return result

    def __del__(self):
        # FIXME : This potentially leaks memory intentionally. It is possible
        # that the `AsyncSampleResult` object gets deleted *before* the `async`
        # sample call occurs or finishes. In that case, we leave the module in
        # the dictionary to prevent the interpreter from crashing. We ought to
        # have a way to inform the C++ code that the result is no longer being
        # sought and the module and `py::handle` should be freed.
        if self.getCalled and self.counter is not None:
            del (cudaq_async_sample_module_cache[self.counter])

    def __str__(self):
        # Serialize to JSON string
        return str(self.impl)


def __broadcastSample(kernel,
                      *args,
                      shots_count=0,
                      explicit_measurements=False):
    """Implement broadcasting of a single sample call over an argument set."""
    argSet = __createArgumentSet(*args)
    N = len(argSet)
    results = []
    for i, a in enumerate(argSet):
        ctx = cudaq_runtime.ExecutionContext('sample', shots_count)
        ctx.totalIterations = N
        ctx.batchIteration = i
        ctx.explicitMeasurements = explicit_measurements
        cudaq_runtime.setExecutionContext(ctx)
        try:
            kernel(*a)
        except BaseException:
            # silence any further exceptions
            try:
                cudaq_runtime.resetExecutionContext()
            except BaseException:
                pass
            raise
        else:
            cudaq_runtime.resetExecutionContext()
        res = ctx.result
        results.append(res)

    return results


def _detail_has_conditionals_on_measure(kernel):
    if isa_kernel_decorator(kernel):
        if kernel.returnType is not None:
            raise RuntimeError(
                f"The `sample` API only supports kernels that return None "
                f"(void). Kernel '{kernel.name}' has return type "
                f"'{kernel.returnType}'. Consider using `run` for kernels "
                f"that return values.")
        # Only check for kernels that are compiled, not library-mode kernels (e.g., photonics)
        if kernel.qkeModule is not None:
            for operation in kernel.qkeModule.body.operations:
                if (hasattr(operation, 'name') and nvqppPrefix + kernel.uniqName
                        == operation.name.value and
                        'qubitMeasurementFeedback' in operation.attributes):
                    return True
    elif isinstance(kernel, PyKernel) and kernel.conditionalOnMeasure:
        return True
    return False


def _detail_check_explicit_measurements(explicit_measurements,
                                        has_conditionals_on_measure_result):
    if explicit_measurements:
        if not cudaq_runtime.supportsExplicitMeasurements():
            raise RuntimeError(
                "The sampling option `explicit_measurements` is not supported "
                "on this target.")
        if has_conditionals_on_measure_result:
            raise RuntimeError(
                "The sampling option `explicit_measurements` is not supported "
                "on kernel with conditional logic on a measurement result.")


def sample(kernel,
           *args,
           shots_count=1000,
           noise_model=None,
           explicit_measurements=False):
    """
    Sample the state generated by the provided `kernel` at the given kernel
    `arguments` over the specified number of circuit executions (`shots_count`).
    Each argument in `arguments` provided can be a list or `ndarray` of
    arguments of the specified kernel argument type, and in this case, the
    `sample` functionality will be broadcasted over all argument sets and a list
    of `sample_result` instances will be returned.

    Args:
      kernel (:class:`Kernel`): The :class:`Kernel` to execute `shots_count`
          times on the QPU.
      *arguments (Optional[Any]): The concrete values to evaluate the kernel
          function at. Leave empty if the kernel doesn't accept any arguments.
          For example, if the kernel takes two `float` values as input, the
          `sample` call should be structured as
          `cudaq.sample(kernel, firstFloat, secondFloat)`. For broadcasting of
          the `sample` function, the arguments should be structured as a `list`
          or `ndarray` of argument values of the specified kernel argument type.
      shots_count (Optional[int]): The number of kernel executions on the QPU.
          Defaults to 1000. Key-word only.
      noise_model (Optional[`NoiseModel`]): The optional :class:`NoiseModel`
          to add noise to the kernel execution on the simulator. Defaults to
          an empty noise model.
      explicit_measurements (Optional[bool]): Whether or not to concatenate
          measurements in execution order for the returned sample result.

    Returns:
      :class:`SampleResult` or `list[SampleResult]`: A dictionary containing
          the measurement count results for the :class:`Kernel`, or a list of
          such results in the case of `sample` function broadcasting.
    """

    has_conditionals_on_measure_result = _detail_has_conditionals_on_measure(
        kernel)

    _detail_check_explicit_measurements(explicit_measurements,
                                        has_conditionals_on_measure_result)

    if noise_model:
        cudaq_runtime.set_noise(noise_model)

    if __isBroadcast(kernel, *args):
        res = __broadcastSample(kernel,
                                *args,
                                shots_count=shots_count,
                                explicit_measurements=explicit_measurements)
        cudaq_runtime.unset_noise()
        return res

    ctx = cudaq_runtime.ExecutionContext("sample", shots_count)
    ctx.hasConditionalsOnMeasureResults = has_conditionals_on_measure_result
    ctx.explicitMeasurements = explicit_measurements
    ctx.allowJitEngineCaching = True
    cudaq_runtime.setExecutionContext(ctx)

    counts = cudaq_runtime.SampleResult()
    while counts.get_total_shots() < shots_count:
        try:
            kernel(*args)
        except BaseException:
            # silence any further exceptions
            try:
                cudaq_runtime.resetExecutionContext()
            except BaseException:
                pass
            raise
        else:
            cudaq_runtime.resetExecutionContext()
        # If the platform is a hardware QPU, launch only once
        countsTotalIsZero = counts.get_total_shots() == 0
        resultTotalWasReached = ctx.result.get_total_shots() == shots_count
        if (countsTotalIsZero and
                resultTotalWasReached) or cudaq_runtime.isQuantumDevice():
            # Early return for case where all shots were gathered the first time
            # through this loop.This avoids an additional copy.
            cudaq_runtime.unset_noise()
            return ctx.result
        counts += ctx.result
        if counts.get_total_shots() == 0:
            if explicit_measurements:
                raise RuntimeError(
                    "The sampling option `explicit_measurements` is not "
                    "supported on a kernel without any measurement operation.")
            print("WARNING: this kernel invocation produced 0 shots worth of "
                  "results when executed. Exiting shot loop to avoid infinite "
                  "loop.")
            break
        ctx.result.clear()
        if counts.get_total_shots() < shots_count:
            cudaq_runtime.setExecutionContext(ctx)

    cudaq_runtime.unset_noise()
    ctx.unset_jit_engine()
    return counts


def sample_async(decorator,
                 *args,
                 shots_count=1000,
                 explicit_measurements=False,
                 noise_model=None,
                 qpu_id=0):
    """
    Asynchronously sample the state of the provided kernel `decorator` at the
    specified number of circuit executions (`shots_count`). When targeting a
    quantum platform with more than one QPU, the optional `qpu_id` allows for
    control over which QPU to enable. Will return a future whose results can be
    retrieved via `future.get()`.

    Args:
      kernel (:class:`Kernel`): The :class:`Kernel` to execute `shots_count`
          times on the QPU.
      *arguments (Optional[Any]): The concrete values to evaluate the kernel
          function at. Leave empty if the kernel doesn't accept any arguments.
      shots_count (Optional[int]): The number of kernel executions on the
          QPU. Defaults to 1000. Key-word only.
      explicit_measurements (Optional[bool]): A flag to indicate whether or not
          to concatenate measurements in execution order for the returned
          sample result.
      `qpu_id` (Optional[int]): The optional identification for which QPU
          on the platform to target. Defaults to zero. Key-word only.

    Returns:
      :class:`AsyncSampleResult`: A dictionary containing the measurement count
          results for the :class:`Kernel`.
    """
    kernel = decorator
    if not isa_kernel_decorator(decorator):
        decorator = mk_decorator(decorator)
    if decorator.formal_arity() != len(args):
        raise RuntimeError(
            "Invalid number of arguments passed to sample_async. " +
            str(len(args)) + " given and " + str(decorator.formal_arity()) +
            " expected.")
    if (not isinstance(shots_count, int)) or (shots_count < 0):
        raise RuntimeError(
            "Invalid `shots_count`. Must be a non-negative number.")
    if (decorator.returnType and
            decorator.returnType != decorator.get_none_type()):
        raise RuntimeError("The `sample_async` API only supports kernels that "
                           "return None (void). Consider using `run_async` for "
                           "kernels that return values.")
    target = cudaq_runtime.get_target()
    num_qpus = target.num_qpus()
    if qpu_id >= num_qpus:
        raise ValueError(f"qpu_id ({qpu_id}) exceeds the number of available "
                         f"QPUs ({num_qpus}).")

    if noise_model:
        if target.is_remote_simulator() or target.is_remote():
            raise ValueError("Noise model is not supported on remote simulator"
                             " or hardware QPU.")

    specMod, processedArgs = decorator.handle_call_arguments(*args)
    has_conditionals_on_measure_result = _detail_has_conditionals_on_measure(
        kernel)
    _detail_check_explicit_measurements(explicit_measurements,
                                        has_conditionals_on_measure_result)

    retTy = decorator.get_none_type()
    sample_results = cudaq_runtime.sample_async_impl(decorator.uniqName,
                                                     specMod, retTy,
                                                     shots_count, noise_model,
                                                     explicit_measurements,
                                                     qpu_id, *processedArgs)
    return AsyncSampleResult(sample_results, specMod)
