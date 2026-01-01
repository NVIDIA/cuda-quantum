# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime
from .utils import __isBroadcast, __createArgumentSet


def __broadcastObserve(kernel, spin_operator, *args, shots_count=0):
    argSet = __createArgumentSet(*args)
    N = len(argSet)
    results = []
    for i, a in enumerate(argSet):
        ctx = cudaq_runtime.ExecutionContext('observe', shots_count)
        ctx.totalIterations = N
        ctx.batchIteration = i
        ctx.setSpinOperator(spin_operator)
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
        results.append(
            cudaq_runtime.ObserveResult(ctx.getExpectationValue(),
                                        spin_operator, res))

    return results


def observe(kernel,
            spin_operator,
            *args,
            shots_count=0,
            noise_model=None,
            num_trajectories=None,
            execution=None):
    """Compute the expected value of the `spin_operator` with respect to 
the `kernel`. If the input `spin_operator` is a list of `SpinOperator` then compute 
the expected value of every operator in the list and return a list of results.
If the kernel accepts arguments, it will be evaluated 
with respect to `kernel(*arguments)`. Each argument in `arguments` provided
can be a list or `ndarray` of arguments of the specified kernel argument
type, and in this case, the `observe` functionality will be broadcasted over
all argument sets and a list of `observe_result` instances will be returned.
If both the input `spin_operator` and `arguments` are broadcast lists, 
a nested list of results over `arguments` then `spin_operator` will be returned.

Args:
  kernel (:class:`Kernel`): The :class:`Kernel` to evaluate the 
    expectation value with respect to.
  spin_operator (`SpinOperator` or `list[SpinOperator]`): The Hermitian spin operator to 
    calculate the expectation of, or a list of such operators.
  *arguments (Optional[Any]): The concrete values to evaluate the 
    kernel function at. Leave empty if the kernel doesn't accept any arguments.
  shots_count (Optional[int]): The number of shots to use for QPU 
    execution. Defaults to -1 implying no shots-based sampling. Key-word only.
  noise_model (Optional[`NoiseModel`]): The optional :class:`NoiseModel` to add 
    noise to the kernel execution on the simulator. Defaults to an empty 
    noise model.
  `num_trajectories` (Optional[int]): The optional number of trajectories for noisy simulation. Only valid if a noise model is provided. Key-word only.

Returns:
  :class:`ObserveResult`: 
    A data-type containing the expectation value of the `spin_operator` with 
    respect to the `kernel(*arguments)`, or a list of such results in the case 
    of `observe` function broadcasting. If `shots_count` was provided, the 
    :class:`ObserveResult` will also contain a :class:`SampleResult` dictionary.
    """

    validityCheck = cudaq_runtime.isValidObserveKernel(kernel)
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
        return cudaq_runtime.observe_parallel(kernel,
                                              spin_operator,
                                              *args,
                                              execution=execution,
                                              shots_count=shots_count,
                                              noise_model=noise_model)

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
    if __isBroadcast(kernel, *args):
        results = __broadcastObserve(kernel,
                                     localOp,
                                     *args,
                                     shots_count=shots_count)

        if isinstance(spin_operator, list):
            results = [[
                cudaq_runtime.ObserveResult(p.expectation(o), o, p.counts(o))
                for o in spin_operator
            ]
                       for p in results]
    else:
        if shots_count > 0:
            ctx = cudaq_runtime.ExecutionContext('observe', shots_count)
        else:
            ctx = cudaq_runtime.ExecutionContext('observe')
        ctx.setSpinOperator(localOp)
        if num_trajectories is not None:
            if noise_model is None:
                raise RuntimeError(
                    "num_trajectories is provided without a noise_model.")
            ctx.numberTrajectories = num_trajectories
        cudaq_runtime.setExecutionContext(ctx)
        try:
            kernel(*args)
        finally:
            cudaq_runtime.resetExecutionContext()
        res = ctx.result

        expVal = ctx.getExpectationValue()
        if expVal == None:
            sum = 0.0

            def computeExpVal(term):
                nonlocal sum
                if term.is_identity():
                    sum += term.evaluate_coefficient().real
                else:
                    sum += res.expectation(
                        term.term_id) * term.evaluate_coefficient().real

            for term in localOp:
                computeExpVal(term)
            expVal = sum

        observeResult = cudaq_runtime.ObserveResult(expVal, localOp, res)
        if not isinstance(spin_operator, list):
            if noise_model != None:
                cudaq_runtime.unset_noise()

            return observeResult

        results = []
        for op in spin_operator:
            results.append(
                cudaq_runtime.ObserveResult(observeResult.expectation(op), op,
                                            observeResult.counts(op)))

    if noise_model != None:
        cudaq_runtime.unset_noise()

    return results
