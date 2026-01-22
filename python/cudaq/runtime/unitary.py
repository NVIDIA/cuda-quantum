# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime
from cudaq.kernel.kernel_decorator import (mk_decorator, isa_kernel_decorator)


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
    specMod, processedArgs = decorator.handle_call_arguments(*args)
    returnTy = (decorator.returnType
                if decorator.returnType else decorator.get_none_type())
    return cudaq_runtime.get_unitary_impl(decorator.uniqName, specMod, returnTy,
                                          *processedArgs)
