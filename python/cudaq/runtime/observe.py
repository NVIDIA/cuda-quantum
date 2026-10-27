# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime
from cudaq.util import trace
from .utils import __isBroadcast, __createArgumentSet
from cudaq.kernel.kernel_decorator import (mk_decorator, isa_kernel_decorator)
from cudaq.kernel.kernel_builder import isa_dynamic_kernel


def isValidObserveKernel(kernel):
    if isa_kernel_decorator(kernel):
        decorator = kernel
    else:
        decorator = mk_decorator(kernel)
    if not decorator.supports_compilation():
        return [
            False,
            "Unsupported target / Invalid kernel for `observe`: cannot compile"
        ]
    return cudaq_runtime.isValidObserveKernel_impl(decorator.uniqName,
                                                   decorator.qkeModule)


def __broadcastObserve(kernel, spin_operator, *args, shots_count=-1, qpu_id=0):
    argSet = __createArgumentSet(*args)
    N = len(argSet)
    results = []
    kernel_name = kernel.name if hasattr(kernel, 'name') else ''
    ctx_shots = shots_count if shots_count > 0 else 0
    ctx = cudaq_runtime.ExecutionContext('observe', ctx_shots, qpu_id)
    ctx.totalIterations = N
    ctx.setSpinOperator(spin_operator)
    ctx.kernelName = kernel_name
    has_vector_args = isa_kernel_decorator(kernel) and any(
        hasattr(a, 'shape') and len(a.shape) == 2 for a in args)
    policy = cudaq_runtime.ObservePolicy(ctx, kernel_name, spin_operator)
    for i, a in enumerate(argSet):
        ctx.batchIteration = i
        results.append(
            cudaq_runtime.launch_observe(policy, ctx, lambda a=a: kernel(*a)))
    return results


@trace.traced
def observe(kernel,
            spin_operator,
            *args,
            shots_count=-1,
            noise_model=None,
            num_trajectories=None,
            execution=None,
            qpu_id=0):
    """
    Compute the expected value of the `spin_operator` with respect to the
    `kernel`. If the input `spin_operator` is a list of `SpinOperator` then
    compute the expected value of every operator in the list and return a list
    of results.  If the kernel accepts arguments, it will be evaluated with
    respect to `kernel(*arguments)`. Each argument in `arguments` provided can
    be a list or `ndarray` of arguments of the specified kernel argument type,
    and in this case, the `observe` functionality will be broadcasted over all
    argument sets and a list of `observe_result` instances will be returned.
    If both the input `spin_operator` and `arguments` are broadcast lists, a
    nested list of results over `arguments` then `spin_operator` will be
    returned.

    If the `spin_operator` is a list, each element is reported separately.
    An element with more than one term reports the coefficient weighted sum
    of the expectation values of its terms, and hence agrees with an
    equivalent per-operator `observe` call for the same operator. An element
    with a single term reports the raw Pauli expectation of that term,
    matching the term-list overload of the C++ `observe`, so its coefficient
    has to be applied by the caller.

    Args:
      kernel (:class:`Kernel`): The :class:`Kernel` to evaluate the 
          expectation value with respect to.
      spin_operator (`SpinOperator` or `list[SpinOperator]`): The Hermitian
          spin operator to calculate the expectation of, or a list of such
          operators.
      *arguments (Optional[Any]): The concrete values to evaluate the kernel
          function at. Leave empty if the kernel doesn't accept any arguments.
      shots_count (Optional[int]): The number of shots to use for QPU execution.
          Defaults to -1 implying no shots-based sampling. Key-word only.
      noise_model (Optional[`NoiseModel`]): The optional :class:`NoiseModel` to
        add noise to the kernel execution on the simulator. Defaults to an empty
        noise model.
      `num_trajectories` (Optional[int]): The optional number of trajectories
        for noisy simulation. Only valid if a noise model is provided.
        `Keyword` only.
      `qpu_id` (Optional[int]): The id of the QPU. Defaults to 0. Key-word only.

    Returns:
      :class:`ObserveResult`: A data-type containing the expectation value of
      the `spin_operator` with respect to the `kernel(*arguments)`, or a list of
      such results in the case of `observe` function broadcasting. If
      `shots_count` was provided, the :class:`ObserveResult` will also contain
      a :class:`SampleResult` dictionary.
    """

    validityCheck = isValidObserveKernel(kernel)
    if not validityCheck[0]:
        raise RuntimeError('observe specification violated for \'' +
                           kernel.name + '\': ' + validityCheck[1])

    spin_operator = spin_operator.copy()
    if isinstance(spin_operator, list):
        for idx, op in enumerate(spin_operator):
            spin_operator[idx] = op.canonicalize()
    else:
        spin_operator.canonicalize()

    # Handle parallel execution use cases
    if execution != None:
        return observe_parallel(kernel,
                                spin_operator,
                                execution,
                                *args,
                                shots_count=shots_count,
                                noise_model=noise_model)

    def __termOf(op):
        """Return the single term of `op`.

        `op` is either a `SpinOperatorTerm` or a `SpinOperator` with
        exactly one term.
        """
        if isinstance(op, cudaq_runtime.SpinOperatorTerm):
            return op
        return next(iter(op))

    def __computeTermExpectation(term, observe_result, weighted=False):
        """Return the contribution of `term` to an expectation value.

        The identity term is never measured, so it contributes its
        coefficient either way. Any other `term` contributes the raw Pauli
        expectation, scaled by the term's coefficient if `weighted`.
        """
        if term.is_identity():
            return term.evaluate_coefficient().real
        exp_val = observe_result.expectation(term)
        if weighted:
            exp_val *= term.evaluate_coefficient().real
        return exp_val

    def __computeListExpectation(op, observe_result):
        """Return the expectation value of `op`, an element of a
        `spin_operator` list.

        A single-term `op` yields the raw Pauli expectation, matching the
        C++ term-list overload in `runtime/cudaq/algorithms/observe.h`. A
        multi-term `op` yields the coefficient weighted sum of its terms, so
        that the list form agrees with a per-operator `observe` call for the
        same operator.
        """
        if op.term_count == 1:
            return __computeTermExpectation(__termOf(op), observe_result)

        exp_val = 0.0
        for term in op:
            exp_val += __computeTermExpectation(term, observe_result, True)
        return exp_val

    def __listCountsData(observe_result):
        """Return the counts data shared by the elements of a
        `spin_operator` list.

        All the operators of a list are measured together, so every element
        shares the same counts data. It holds one register per measured
        term, and the identity term, which is never measured, has none.
        """
        return observe_result.counts()

    if noise_model != None:
        cudaq_runtime.set_noise(noise_model)

    # Process spin_operator if its a list
    if isinstance(spin_operator, cudaq_runtime.SpinOperatorTerm):
        localOp = cudaq_runtime.SpinOperator(spin_operator)
    elif isinstance(spin_operator, list):
        localOp = cudaq_runtime.SpinOperator.empty()
        for o in spin_operator:
            localOp += o
    else:
        localOp = spin_operator

    results = None
    if isa_kernel_decorator(kernel):
        decorator = kernel
    elif isa_dynamic_kernel(kernel):
        decorator = mk_decorator(kernel)
    else:
        raise RuntimeError(
            "unrecognized kernel - did you forget the @kernel attribute?")
    if (decorator.launch_args_required() != 0) and (decorator.formal_arity()
                                                    != len(args)):
        raise RuntimeError("Invalid number of arguments passed to observe. " +
                           str(len(args)) + " given and " +
                           str(decorator.formal_arity()) + " expected.")
    if __isBroadcast(kernel, *args):
        results = __broadcastObserve(kernel,
                                     localOp,
                                     *args,
                                     shots_count=shots_count,
                                     qpu_id=qpu_id)

        if isinstance(spin_operator, list):
            results = [[
                cudaq_runtime.ObserveResult(__computeListExpectation(o, p), o,
                                            __listCountsData(p))
                for o in spin_operator
            ]
                       for p in results]
    else:
        if shots_count > 0:
            ctx = cudaq_runtime.ExecutionContext('observe', shots_count, qpu_id)
        else:
            ctx = cudaq_runtime.ExecutionContext('observe', 0, qpu_id)
        ctx.setSpinOperator(localOp)
        if num_trajectories is not None:
            if noise_model is None:
                raise RuntimeError(
                    "num_trajectories is provided without a noise_model.")
            ctx.numberTrajectories = num_trajectories
        kernel_name = kernel.name if hasattr(kernel, 'name') else ''
        policy = cudaq_runtime.ObservePolicy(ctx, kernel_name, localOp)
        observeResult = cudaq_runtime.launch_observe(
            policy, ctx, lambda args=args: kernel(*args))
        if not isinstance(spin_operator, list):
            if noise_model != None:
                cudaq_runtime.unset_noise()

            return observeResult

        results = [
            cudaq_runtime.ObserveResult(
                __computeListExpectation(op, observeResult), op,
                __listCountsData(observeResult)) for op in spin_operator
        ]

    if noise_model != None:
        cudaq_runtime.unset_noise()

    return results


@trace.traced
def observe_async(kernel, spin_operator, *args, qpu_id=0, shots_count=-1):
    """
    Compute the expected value of the `spin_operator` with respect to the
    `kernel` asynchronously. If the kernel accepts arguments, it will be
    evaluated with respect to `kernel(*arguments)`. When targeting a quantum
    platform with more than one QPU, the optional `qpu_id` allows for control
    over which QPU to enable. Will return a future whose results can be
    retrieved via `future.get()`.

    Args:
      kernel (:class:`Kernel`): The :class:`Kernel` to evaluate the expectation
          value with respect to.
      spin_operator (`SpinOperator`): The Hermitian spin operator to calculate
          the expectation of.
      *arguments (Optional[Any]): The concrete values to evaluate the kernel
          function at. Leave empty if the kernel doesn't accept any arguments.
      `qpu_id` (Optional[int]): The optional identification for which QPU on the
          platform to target. Defaults to zero. Key-word only.
      shots_count (Optional[int]): The number of shots to use for QPU execution.
          Defaults to -1 implying no shots-based sampling. Key-word only.

    Returns:
      :class:`AsyncObserveResult`: A future containing the result of the call
          to observe.
    """
    validityCheck = isValidObserveKernel(kernel)
    if not validityCheck[0]:
        raise RuntimeError('observe specification violated for \'' +
                           kernel.name + '\': ' + validityCheck[1])

    if isa_kernel_decorator(kernel):
        decorator = kernel
    elif isa_dynamic_kernel(kernel):
        decorator = mk_decorator(kernel)
    else:
        raise RuntimeError(
            "unrecognized kernel - did you forget the @kernel attribute?")
    if (decorator.launch_args_required() != 0) and (decorator.formal_arity()
                                                    != len(args)):
        raise RuntimeError(
            "Invalid number of arguments passed to observe_async. " +
            str(len(args)) + " given and " + str(decorator.formal_arity()) +
            " expected.")
    shortName = decorator.uniqName
    processedArgs, module = decorator.prepare_call(*args)
    return cudaq_runtime.observe_async_impl(shortName, module,
                                            decorator.compiledModuleCache(),
                                            spin_operator, qpu_id, shots_count,
                                            *processedArgs)


def observe_parallel(kernel,
                     spin_operator,
                     execution,
                     *args,
                     shots_count=-1,
                     noise_model=None):
    """
    Compute the expected value of the `spin_operator` with respect to the
    `kernel`. If the input `spin_operator` is a list of `SpinOperator` then
    compute the expected value of every operator in the list and return a list
    of results. If the kernel accepts arguments, it will be evaluated with
    respect to `kernel(*arguments)`. Each argument in `arguments` provided can
    be a list or `ndarray` of arguments of the specified kernel argument type, &
    in this case, the `observe` functionality will be broadcasted over all
    argument sets and a list of `observe_result` instances will be returned. If
    both the input `spin_operator` and `arguments` are broadcast lists, a nested
    list of results over `arguments` then `spin_operator` will be returned.

    Args:
      kernel (:class:`Kernel`): The :class:`Kernel` to evaluate the expectation
          value with respect to.
      spin_operator (:class:`SpinOperator` or `list[SpinOperator]`): The
          Hermitian spin operator to calculate the expectation of, or a list of
          such operators.
      *arguments (Optional[Any]): The concrete values to evaluate the kernel
           function at. Leave empty if the kernel doesn't accept any arguments.
      shots_count (Optional[int]): The number of shots to use for QPU execution.
           Defaults to -1 implying no shots-based sampling. Key-word only.
      noise_model (Optional[`NoiseModel`]): The optional :class:`NoiseModel` to
          add noise to the kernel execution on the simulator. Defaults to an
          empty noise model.

    Returns:
      :class:`ObserveResult`: A data-type containing the expectation value of
          the `spin_operator` with respect to the `kernel(*arguments)`, or a
          list of such results in the case of `observe` function broadcasting.
          If `shots_count` was provided, the :class:`ObserveResult` will also
          contain a :class:`SampleResult` dictionary.
    """
    validityCheck = isValidObserveKernel(kernel)
    if not validityCheck[0]:
        raise RuntimeError('observe specification violated for \'' +
                           kernel.name + '\': ' + validityCheck[1])

    if isa_kernel_decorator(kernel):
        decorator = kernel
        if (decorator.launch_args_required()
                != 0) and (decorator.formal_arity() != len(args)):
            raise RuntimeError(
                "Invalid number of arguments passed to observe_parallel. " +
                str(len(args)) + " given and " + str(decorator.formal_arity()) +
                " expected.")
    elif isa_dynamic_kernel(kernel):
        decorator = mk_decorator(kernel)
    else:
        raise RuntimeError(
            "unrecognized kernel - did you forget the @kernel attribute?")
    shortName = decorator.uniqName
    processedArgs, module = decorator.prepare_call(*args)
    return cudaq_runtime.observe_parallel_impl(shortName, module,
                                               decorator.compiledModuleCache(),
                                               execution, spin_operator,
                                               shots_count, noise_model,
                                               *processedArgs)
