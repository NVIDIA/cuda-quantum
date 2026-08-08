# ============================================================================ #
# Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime
from cudaq.mlir.ir import UnitAttr
from cudaq.kernel.kernel_decorator import (mk_decorator, isa_kernel_decorator)
from cudaq.util import trace


@trace.traced
def get_unitary(kernel, *args):
    """
    Return the unitary matrix of the execution path of the provided kernel.

    Args:
      kernel (:class:`Kernel`): The :class:`Kernel` to analyze.
      *arguments (Optional[Any]): The concrete values to evaluate the kernel at.

    Returns:
      `numpy.ndarray`: The unitary matrix as a complex-valued NumPy array.

    .. code-block:: python

      import cudaq
      @cudaq.kernel
      def bell():
        `q = cudaq.qvector(2)`
        h(q[0])
        `cx(q[0], q[1])`
      U = cudaq.get_unitary(bell)
     print(U)
    """
    if isa_kernel_decorator(kernel):
        decorator = kernel
    else:
        decorator = mk_decorator(kernel)
    processedArgs, module = decorator.prepare_call(*args)
    # The unitary is assembled from the trace of the executed kernel, so the
    # trace has to see every qubit the kernel allocates, in allocation order.
    # Value-semantics lowering runs dead quantum elimination, which drops
    # qubits no gate acts on and renumbers the ones that remain. The resulting
    # matrix would then have the wrong dimension and the wrong qubit ordering.
    # Opt this launch out of those passes. `prepare_call` handed back a clone,
    # so the kernel's own module is untouched.
    module.operation.attributes.__setitem__(
        'quake.noOptimization', UnitAttr.get(context=module.context))
    return cudaq_runtime.get_unitary_impl(decorator.uniqName, module,
                                          decorator.unoptimizedModuleCache(),
                                          *processedArgs)
