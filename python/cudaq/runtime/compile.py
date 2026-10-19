# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime
from cudaq.kernel.kernel_decorator import (mk_decorator, isa_kernel_decorator)
from cudaq.util import trace

CompiledModule = cudaq_runtime.CompiledModule


@trace.traced
def compile(kernel, *args, target=None):
    """
    Compiles the given quantum kernel for a target without launching it, and
    returns the result as a :class:`CompiledModule`.

    The kernel is compiled with the compile target and for the runtime endpoint
    of ``target``, exactly as a launch (e.g. :func:`cudaq.sample`) would
    compile it. In particular, a JIT-compiled entry point is built if, and
    only if, such a launch would build one. The MLIR the target's pipeline
    produced is available as ``CompiledModule.mlir_module``.

    Launch policies are not taken into account: the module is compiled with
    the default options, not those a specific algorithm such as
    :func:`cudaq.observe` would request.

    Args:
      kernel (:class:`Kernel`): The :class:`Kernel` to compile.
      *args (Optional[Any]): The concrete values to compile the kernel for.
          Leave empty if the kernel doesn't accept any arguments.
      target (Optional[:class:`CustomTarget`]): The target to compile for.
          Defaults to the target currently installed with
          :func:`cudaq.set_target`.

    Returns:
      :class:`CompiledModule`: The compiled kernel.

    # Example:
    import cudaq

    @cudaq.kernel
    def bell_pair():
        q = cudaq.qvector(2)
        h(q[0])
        x.ctrl(q[0], q[1])
        mz(q)

    module = cudaq.compile(bell_pair)
    print(module.mlir_module)
    """
    if isa_kernel_decorator(kernel):
        decorator = kernel
    else:
        decorator = mk_decorator(kernel)

    compile_target = None
    endpoint_traits = {}
    if target is not None:
        from cudaq.core.backends import CustomTarget
        if not isinstance(target, CustomTarget):
            raise TypeError(
                "target must be a CustomTarget, which pairs a compile target "
                f"with the runtime endpoint it compiles for, not {type(target).__name__}"
            )
        compile_target = target.compile_target
        endpoint = target.runtime_endpoint
        # These match the defaults of `RuntimeEndpoint`.
        endpoint_traits = dict(
            is_remote=getattr(endpoint, "is_remote", False),
            is_emulated=getattr(endpoint, "is_emulated", False),
            supports_jit=getattr(endpoint, "supports_jit", True),
        )

    processed_args, module = decorator.prepare_call(*args)
    return cudaq_runtime.compile_module(decorator.uniqName,
                                        module,
                                        True,
                                        compile_target=compile_target,
                                        args=tuple(processed_args),
                                        **endpoint_traits)
