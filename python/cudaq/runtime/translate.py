# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime
from cudaq.kernel.kernel_decorator import isa_kernel_decorator


def translate(kernel, *args, format="qir:0.1"):
    """
    Return a `UTF-8` encoded string representing drawing of the execution path,
    i.e., the trace, of the provided `kernel`.

    Args:
      format (`str`): format to translate to, <name[:version]>.
          Available format names: `qir`, `qir-full`, `qir-base`, `qir-adaptive`,
          `openqasm2`. QIR versions: `0.1` and `1.0`.
      kernel (:class:`Kernel`): The :class:`Kernel` to translate.
      *arguments (Optional[Any]): The concrete values to evaluate the kernel
          function at. Leave empty if the kernel doesn't accept any arguments.

    Note: Translating functions with arguments to OpenQASM 2.0 is not supported.

    Returns:
      The `UTF-8` encoded string of the circuit, without measurement operations.

    # Example:
    import cudaq
    @cudaq.kernel
    def bell_pair():
    `q = cudaq.qvector(2)`
    h(q[0])
    `cx(q[0], q[1])`
    `mz(q)`
    print(cudaq.translate(bell_pair, `format="qir"`))

    # Output
    `; ModuleID = 'LLVMDialectModule'`
    `source_filename = 'LLVMDialectModule'`

    %Array = type opaque
    %Result = type opaque
    %Qubit = type opaque

    ...
    ...

    define void `@__nvqpp__mlirgen__function_variable_qreg._Z13variable_qregv`() `local_unnamed_addr` {
    %1 = tail call %Array* @__quantum__rt__qubit_allocate_array(i64 2)
    ...
    %8 = tail call %Result* @`__quantum__qis__mz`(%Qubit* %4)
    %9 = tail call %Result* @`__quantum__qis__mz`(%Qubit* %7)
    tail call void @`__quantum__rt__qubit_release_array`(%Array* %1)
    `ret void`
    }
    """
    decorator = None
    if isa_kernel_decorator(kernel):
        decorator = kernel
    elif isa_dynamic_kernel(kernel):
        decorator = mk_decorator(kernel)
    if not decorator:
        raise RuntimeError("kernel is invalid type")
    launchArgsReq = decorator.launch_args_required()
    formals = decorator.formal_arity()
    suppliedArgs = len(args)
    if (launchArgsReq != 0) and "openqasm" in format:
        raise RuntimeError("Use synthesize before translate to openqasm to "
                           "specialize the kernel with the argument values")
    else:
        if suppliedArgs > formals:
            raise RuntimeError(
                f"Invalid number of arguments passed to translate. "
                f"{suppliedArgs} given, {formals} formally declared, and "
                f"{launchArgsReq} required.")
    specMod, processedArgs = decorator.handle_call_arguments(*args)
    if launchArgsReq != len(processedArgs):
        deducedArgs = len(processedArgs) - suppliedArgs
        raise RuntimeError(f"Invalid number of arguments passed to translate. "
                           f"{suppliedArgs} given, {deducedArgs} deduced, and "
                           f"{launchArgsReq} required.")
    retTy = (decorator.returnType
             if decorator.returnType else decorator.get_none_type())
    # Arguments are resolved. Specialize this kernel and translate to the
    # selected transport layer.
    return cudaq_runtime.translate_impl(decorator.uniqName, specMod, retTy,
                                        format, *processedArgs)
