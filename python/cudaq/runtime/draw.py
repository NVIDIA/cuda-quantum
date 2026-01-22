# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime
from cudaq.kernel.kernel_decorator import (mk_decorator, isa_kernel_decorator)


def _detail_draw(format, decorator, *args):
    if not isa_kernel_decorator(decorator):
        decorator = mk_decorator(decorator)
    if decorator.formal_arity() != len(args):
        raise RuntimeError("Invalid number of arguments passed to run. " +
                           str(len(args)) + " given and " +
                           str(decorator.formal_arity()) + " expected.")
    # Must handle arguments exactly like this is a `callsite` to the decorator.
    specMod, processedArgs = decorator.handle_call_arguments(*args)
    retTy = decorator.returnType
    if not retTy:
        retTy = decorator.get_none_type()
    # Arguments are resolved, so go ahead and do the draw functionality, which
    # performs a kernel launch.
    return cudaq_runtime.draw_impl(format, decorator.uniqName, specMod, retTy,
                                   *processedArgs)


def draw(decoratorOrFormat, *args):
    """
    The CUDA-Q specification overloads draw. To meet that, this function uses
    parameter type checking. The two overloads for `cudaq.draw` are:
    ```
    python
    cudaq.draw("<format>", kernel, opt_args...)
    cudaq.draw(kernel, opt_args...)
    ```
    The second overload is equivalent to using a format string of `"ascii"`.
    """
    if isinstance(decoratorOrFormat, str):
        # User specified output format.
        assert (len(args) == 1) and "must have a kernel"
        vargs = args[1:]
        return _detail_draw(decoratorOrFormat, args[0], *vargs)
    # Default to the UTF-8 code points (confusingly named `"ascii"`).
    return _detail_draw("ascii", decoratorOrFormat, *args)


__all__ = ['draw']
