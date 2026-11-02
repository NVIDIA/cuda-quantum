# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime
from cudaq.kernel.kernel_decorator import (mk_decorator, isa_kernel_decorator)
from cudaq.util import trace


def _detail_draw(format, decorator, *args, optimization_level=None):
    if not isa_kernel_decorator(decorator):
        decorator = mk_decorator(decorator)
    if decorator.formal_arity() != len(args):
        raise RuntimeError("Invalid number of arguments passed to run. " +
                           str(len(args)) + " given and " +
                           str(decorator.formal_arity()) + " expected.")
    # Must handle arguments exactly like this is a `callsite` to the decorator.
    processedArgs, module = decorator.prepare_call(
        *args, optimization_level=optimization_level)
    return cudaq_runtime.draw_impl(format, decorator.uniqName, module,
                                   decorator.compiledModuleCache(),
                                   *processedArgs)


@trace.traced
def draw(decoratorOrFormat, *args, optimization_level=None):
    """
    The CUDA-Q specification overloads draw. To meet that, this function uses
    parameter type checking. The two overloads for `cudaq.draw` are:
    ```
    python
    cudaq.draw("<format>", kernel, opt_args...)
    cudaq.draw(kernel, opt_args...)
    ```
    The second overload is equivalent to using a format string of `"ascii"`.
    The optional key-word only `optimization_level` (an integer between 0 and
    3) selects the quantum optimization level to compile the kernel with.
    """
    if isinstance(decoratorOrFormat, str):
        # User specified output format.
        assert (len(args) == 1) and "must have a kernel"
        vargs = args[1:]
        return _detail_draw(decoratorOrFormat,
                            args[0],
                            *vargs,
                            optimization_level=optimization_level)
    # Default to the UTF-8 code points (confusingly named `"ascii"`).
    return _detail_draw("ascii",
                        decoratorOrFormat,
                        *args,
                        optimization_level=optimization_level)


__all__ = ['draw']
