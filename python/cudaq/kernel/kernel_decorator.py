# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import ast
import inspect
import json
from functools import wraps
from cudaq.kernel.utils import emitWarning
import numpy as np
import sys

from cudaq.handlers import get_target_handler
from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime
from cudaq.mlir.dialects import cc, func
from cudaq.mlir.ir import (ComplexType, F32Type, F64Type, FunctionType,
                           IntegerType, NoneType, TypeAttr, UnitAttr, Module,
                           Type)
from .analysis import FunctionDefVisitor
from .kernel_signature import CapturedLinkedKernel, CapturedVariable, KernelSignature
from .ast_bridge import compile_to_mlir
from .utils import (emitFatalError, emitErrorIfInvalidPauli, get_module_name,
                    globalRegisteredTypes, mlirTypeFromPyType, mlirTypeToPyType,
                    nvqppPrefix, getMLIRContext, recover_func_op,
                    recover_value_of)

# This file implements the decorator mechanism needed to JIT compile CUDA-Q
# kernels. It exposes the cudaq.kernel() decorator which hooks us into the JIT
# compilation infrastructure which maps the AST representation to an MLIR
# representation and ultimately executable code.


def ensure_compiled(method):
    """
    Decorator for `PyKernelDecorator` methods that ensures the kernel
    is compiled before the method body executes.
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        self._ensure_compiled()
        return method(self, *args, **kwargs)

    return wrapper


def ensure_not_recursive(method):
    """
    Decorator for `PyKernelDecorator.resolve_captured_arguments` method that
    ensures the method is not called recursively.
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if self._resolving_arguments:
            self._resolving_arguments = False
            emitFatalError(
                f"Not supported: recursive kernel call detected in {self.name}."
            )

        self._resolving_arguments = True
        ret = method(self, *args, **kwargs)
        self._resolving_arguments = False
        return ret

    return wrapper


class DecoratorCapture:

    def __init__(self, decorator, values):
        self.decorator = decorator
        self.resolved = values

    def __str__(self):
        self.decorator.name + " -> " + str(self.resolved)

    def __repr__(self):
        "name: " + self.decorator.name + ", resolved: " + str(self.resolved)


class LinkedKernelCapture:
    '''
    Captures a linked C++ kernel. Includes the name of the
    linked kernel and its quake code.
    '''

    def __init__(self, linkedKernel, qkeModule):
        self.linkedKernel = linkedKernel
        self.qkeModule = qkeModule

    def __str__(self):
        self.linkedKernel

    def __repr__(self):
        "name: " + self.linkedKernel


class PyKernelDecorator(object):
    """
    The `PyKernelDecorator` serves as a standard Python decorator that takes 
    the decorated function as input. The function AST is parsed and converted to
    a Quake MLIR representation. This is passed on to the CUDAQ runtime for
    execution at kernel call time.

    By default, MLIR compilation is deferred until the first call to the kernel.
    If `defer_compilation` is set to `False`, the kernel will be compiled at
    declaration time instead.
    """

    def __init__(self,
                 function,
                 verbose=False,
                 defer_compilation=True,
                 module=None,
                 kernelName=None,
                 signature=None,
                 location=None,
                 overrideGlobalScopedVars=None,
                 decorator=None):

        self.location = location
        self.signature = signature
        self.kernelModuleName = None
        self.name = kernelName
        self.verbose = verbose
        # Caches the `qkeModule` property once compiled
        self._cached_qkeModule = None
        self.defFrame = _recover_defining_frame()
        # Whether we are currently resolving arguments to self. Used to detect
        # (and prevent) recursive kernel calls.
        self._resolving_arguments = False

        if isinstance(function, str):
            self.kernelFunction = None
            self.funcSrc = function
        else:
            self.kernelFunction = function
            (src, loc) = _get_source(self.kernelFunction)
            self.funcSrc = src
            self.location = loc

        if self.kernelFunction:
            self.kernelModuleName = self.kernelFunction.__module__
            if self.name is None:
                self.name = self.kernelFunction.__name__

        if module is not None:
            if self.kernelFunction is not None or self.funcSrc is not None:
                raise RuntimeError(
                    "constructor arguments `module` and `function` cannot be provided together."
                )

            if decorator is not None:
                # shallow copy attributes from `decorator`
                self.uniqueId = decorator.uniqueId
                self.uniqName = decorator.uniqName
            else:
                self.uniqueId = int(kernelName.split("..0x")[1], 16)
                self.uniqName = kernelName

            self._cached_qkeModule = module
            self.astModule = None
            self.signature = KernelSignature.parse_from_mlir(
                self.qkeModule, self.uniqName)
        else:
            # Get any global variables from parent scope. Note here we assume
            # that the parent scope is 2 stack frames up
            self.parentFrame = inspect.stack()[2].frame
            self.globalScopedVars = {}
            if overrideGlobalScopedVars:
                parentVars = overrideGlobalScopedVars
            else:
                parentVars = self.parentFrame.f_locals
            for name, var in parentVars.items():
                self._add_global_scoped_var(name, var)

            self.astModule = _parse_ast(self.funcSrc, self.verbose)
            self.signature = KernelSignature.parse_from_ast(
                self.astModule, self.name)
            self.uniqueId = id(self)
            self.uniqName = self.name + ".." + hex(self.uniqueId)

            if not defer_compilation:
                self.compile()

    def __del__(self):
        # explicitly call `del` on the MLIR `ModuleOp` wrappers.
        if self._cached_qkeModule:
            del self._cached_qkeModule

    @property
    @ensure_compiled
    def qkeModule(self):
        """
        A target independent Quake MLIR representation of the kernel.
        """
        return self._cached_qkeModule

    def signatureWithCallables(self):
        """
        returns True if and only if the entry-point contains callable arguments
        and/or return values.
        """
        for ty in self.signature.get_all_types():
            if cc.CallableType.isinstance(ty) or FunctionType.isinstance(ty):
                return True
        return False

    @property
    def return_type(self):
        return self.signature.return_type

    def arg_types(self, include_captured: bool = False) -> list[Type]:
        arg_types = self.signature.arg_types
        if include_captured:
            arg_types = arg_types + self.signature.captured_types()
        return arg_types

    def captured_variables(self):
        """The list of variables captured by the kernel."""
        return self.signature.captured_variables()

    def _ensure_compiled(self):
        """
        Ensure that the kernel is compiled.
        """
        if self._cached_qkeModule is None:
            self.compile()

    def is_compiled(self):
        """Whether the kernel has already been compiled."""
        return self._cached_qkeModule is not None

    def supports_compilation(self):
        """Whether the kernel can be compiled for the current target."""
        handler = get_target_handler()
        return not handler.skip_compilation()

    def compile(self):
        """
        Compile the Python AST to portable Quake.
        """
        if not self.astModule:
            emitFatalError(
                f"Cannot compile kernel {self.name}: no AST module available")

        if not self.supports_compilation():
            emitFatalError(
                f"Cannot compile kernel '{self.name}': target handler "
                f"'{cudaq_runtime.get_target().name}' does not support compilation"
            )

        self._cached_qkeModule = compile_to_mlir(
            id(self),
            self.astModule,
            self.signature,
            self.defFrame,
            verbose=self.verbose,
            location=self.location,
            kernelName=self.name,
            kernelModuleName=self.kernelModuleName)

        # recursively compile any captured kernels if required
        for captured_arg in self.signature.captured_args:
            if isinstance(captured_arg, CapturedVariable
                         ) and captured_arg.name in self.globalScopedVars:
                var = self.globalScopedVars[captured_arg.name]
                if isa_kernel_decorator(var):
                    var._ensure_compiled()

    def merge_kernel(self, otherMod):
        """
        Merge the kernel in this PyKernelDecorator (the ModuleOp) with the
        provided ModuleOp.
        """
        # NB: this method used by tests.
        if isinstance(otherMod, str):
            raise RuntimeError("otherMod must be an MlirModule")
        newMod = cudaq_runtime.mergeExternalMLIR(self.qkeModule,
                                                 otherMod.qkeModule)
        # Get the name of the kernel entry point
        name = self.uniqName
        for op in newMod.body:
            if isinstance(op, func.FuncOp):
                for attr in op.attributes:
                    if 'cudaq-entrypoint' == attr.name:
                        name = op.name.value.removeprefix(nvqppPrefix)
                        break

        return PyKernelDecorator(None,
                                 kernelName=name,
                                 module=newMod,
                                 decorator=self)

    def merge_quake_source(self, quakeText):
        """
        Merge a module of quake code from source text form into this decorator's
        `qkeModule` attribute.
        """
        if not isinstance(quakeText, str):
            raise RuntimeError("argument must be a string")
        newMod = cudaq_runtime.mergeMLIRString(self.qkeModule, quakeText)
        # Get the name of the kernel entry point
        name = self.uniqName
        for op in newMod.body:
            if isinstance(op, func.FuncOp):
                for attr in op.attributes:
                    if 'cudaq-entrypoint' == attr.name:
                        name = op.name.value.removeprefix(nvqppPrefix)
                        break

        return PyKernelDecorator(None,
                                 kernelName=name,
                                 module=newMod,
                                 decorator=self)

    def __str__(self):
        """
        Return a string representation for this kernel, either as MLIR if
        available, or as the source code if not.

        To ensure an MLIR representation is returned, call `compile` beforehand.
        """
        if self._cached_qkeModule:
            return f"Compiled kernel {self.name}\n: {self._cached_qkeModule}"
        elif self.funcSrc:
            return f"Uncompiled kernel {self.name}\n: " + self.funcSrc
        else:
            return f"Uncompiled kernel {self.name}"

    def enable_return_to_log(self):
        """
        Enable translation from `return` statements to QIR output log
        """
        if self._cached_qkeModule is None:
            emitFatalError(
                f"kernel decorator {self.name} has not been compiled")
        self._cached_qkeModule.operation.attributes.__setitem__(
            'quake.cudaq_run', UnitAttr.get(context=self.qkeModule.context))

    @ensure_compiled
    def _repr_svg_(self):
        """
        Return the SVG representation of `self` (:class:`PyKernelDecorator`).
        This assumes no arguments are required to execute the kernel, and
        `latex` (with `quantikz` package) and `dvisvgm` are installed, and the
        temporary directory is writable.  If any of these assumptions fail,
        returns None.
        """
        if len(self.arg_types(include_captured=True)) != 0:
            return None
        from cudaq import getSVGstring

        try:
            from subprocess import CalledProcessError

            try:
                return getSVGstring(self)
            except CalledProcessError:
                return None
        except ImportError:
            return None

    def isCastablePyType(self, fromTy, toTy):
        if IntegerType.isinstance(toTy) and IntegerType(toTy).width != 1:
            return IntegerType.isinstance(fromTy) and IntegerType(
                fromTy).width != 1

        if F64Type.isinstance(toTy):
            return F32Type.isinstance(fromTy) or IntegerType.isinstance(fromTy)

        if F32Type.isinstance(toTy):
            return F64Type.isinstance(fromTy) or IntegerType.isinstance(fromTy)

        if F64Type.isinstance(toTy):
            return F32Type.isinstance(fromTy) or IntegerType.isinstance(fromTy)

        if ComplexType.isinstance(toTy):
            floatToType = ComplexType(toTy).element_type
            if ComplexType.isinstance(fromTy):
                floatFromType = ComplexType(fromTy).element_type
                return self.isCastablePyType(floatFromType, floatToType)

            return fromTy == floatToType or self.isCastablePyType(
                fromTy, floatToType)

        # Support passing `list[int]` to a `list[float]` argument and
        # passing `list[int]` or `list[float]` to a `list[complex]` argument.
        if cc.StdvecType.isinstance(fromTy):
            if cc.StdvecType.isinstance(toTy):
                fromEleTy = cc.StdvecType.getElementType(fromTy)
                toEleTy = cc.StdvecType.getElementType(toTy)

                return self.isCastablePyType(fromEleTy, toEleTy)

        return False

    def castPyType(self, fromTy, toTy, value):
        if self.isCastablePyType(fromTy, toTy):
            if IntegerType.isinstance(toTy):
                intToTy = IntegerType(toTy)
                if intToTy.width == 1:
                    return bool(value)
                if intToTy.width == 8:
                    return np.int8(value)
                if intToTy.width == 16:
                    return np.int16(value)
                if intToTy.width == 32:
                    return np.int32(value)
                if intToTy.width == 64:
                    return int(value)

            if F64Type.isinstance(toTy):
                return float(value)

            if F32Type.isinstance(toTy):
                return np.float32(value)

            if ComplexType.isinstance(toTy):
                floatToType = ComplexType(toTy).element_type

                if F64Type.isinstance(floatToType):
                    return complex(value)

                return np.complex64(value)

            # Support passing `list[int]` to a `list[float]` argument and
            # passing `list[int]` or `list[float]` to a `list[complex]` argument
            if cc.StdvecType.isinstance(fromTy):
                if cc.StdvecType.isinstance(toTy):
                    fromEleTy = cc.StdvecType.getElementType(fromTy)
                    toEleTy = cc.StdvecType.getElementType(toTy)

                    if self.isCastablePyType(fromEleTy, toEleTy):
                        return [
                            self.castPyType(fromEleTy, toEleTy, element)
                            for element in value
                        ]
        return value

    @staticmethod
    def type_to_str(t):
        """
        This converts types to strings in a clean JSON-compatible way.
        int -> 'int'
        list[float] -> 'list[float]'
        List[float] -> 'list[float]'
        """
        if hasattr(t, '__origin__') and t.__origin__ is not None:
            # Handle generic types from typing
            origin = t.__origin__
            args = t.__args__
            args_str = ', '.join(
                PyKernelDecorator.type_to_str(arg) for arg in args)
            return f'{origin.__name__}[{args_str}]'
        elif hasattr(t, '__name__'):
            return t.__name__
        else:
            return str(t)

    def to_json(self):
        """
        Convert `self` to a JSON-serialized version of the kernel such that
        `from_json` can reconstruct it elsewhere.
        """
        obj = dict()
        obj['name'] = self.name
        obj['location'] = self.location
        obj['funcSrc'] = self.funcSrc
        return json.dumps(obj)

    @staticmethod
    def from_json(jStr, overrideDict=None):
        """
        Convert a JSON string into a new PyKernelDecorator object.
        """
        j = json.loads(jStr)
        return PyKernelDecorator(function=j['funcSrc'],
                                 verbose=False,
                                 kernelName=j['name'],
                                 location=j['location'],
                                 overrideGlobalScopedVars=overrideDict)

    def convertStringsToPauli(self, arg):
        if isinstance(arg, str):
            # Only allow `pauli_word` as string input
            emitErrorIfInvalidPauli(arg)
            return cudaq_runtime.pauli_word(arg)

        if issubclass(type(arg), list):
            return [self.convertStringsToPauli(a) for a in arg]

        return arg

    def formal_arity(self):
        return len(self.arg_types())

    @ensure_compiled
    @ensure_not_recursive
    def resolve_captured_arguments(self):
        """
        Resolve the captured arguments of the decorator.

        These arguments get resolved in the scope of the kernel definition
        (lexical scoping).
        """
        captured_args = []

        for arg in self.signature.captured_args:
            if isinstance(arg, CapturedLinkedKernel):
                # Lifted argument is a registered C++ kernel, load and capture it
                [linkedKernel,
                 maybeCode] = cudaq_runtime.checkRegisteredCppDeviceKernel(
                     self.qkeModule, arg.kernel_name)
                qkeModule = Module.parse(maybeCode,
                                         context=self.qkeModule.context)
                captured_args.append(
                    LinkedKernelCapture(linkedKernel, qkeModule))
            else:
                arg_value = recover_value_of(arg.name, self.defFrame)
                arg = self.process_argument(arg_value, arg.type)
                captured_args.append(arg)

        return captured_args

    @ensure_compiled
    def process_call_arguments(self, *args):
        """
        Resolve the arguments passed to the decorator at call site.
        """
        return [
            self.process_argument(arg, arg_type)
            for arg, arg_type in zip(args, self.arg_types())
        ]

    def prepare_call(self, *args):
        """
        Process call site arguments, capture lifted arguments and retrieve
        compiled module for kernel execution.
        """
        processed_args = self.process_call_arguments(*args)
        # append captured arguments
        processed_args.extend(self.resolve_captured_arguments())

        module = cudaq_runtime.cloneModule(self.qkeModule)

        return processed_args, module

    def get_none_type(self):
        if self._cached_qkeModule:
            context = self._cached_qkeModule.context
        else:
            context = getMLIRContext()
        return NoneType.get(context)

    def handle_call_results(self):
        if not self.return_type:
            return self.get_none_type()
        return self.return_type

    @ensure_compiled
    def launch_args_required(self):
        """
        This is a deeper query on the quake module. The quake module may have
        been specialized such that none of the arguments are, in fact, required
        to be provided in order to run the kernel. (Argument synthesis.)
        
        This will analyze the designated entry-point kernel for the quake module
        and determine if any arguments are used and return the number used.
        """
        if len(self.arg_types(include_captured=True)) == 0:
            return 0
        shortName = self.uniqName
        return cudaq_runtime.get_launch_args_required(self.qkeModule, shortName)

    def __call__(self, *args):
        """
        Invoke the CUDA-Q kernel. JIT compilation of the kernel AOT Quake module
        to machine code will occur here.
        """

        # If this target requires library mode (no MLIR compilation), just
        # launch it.
        handler = get_target_handler()
        if handler.call_processed(self.kernelFunction, args) is True:
            return

        if args and self.formal_arity() != len(args):
            emitFatalError("wrong number of arguments provided")

        processed_args, module = self.prepare_call(*args)

        mlirTy = self.handle_call_results()
        result = cudaq_runtime.marshal_and_launch_module(
            self.uniqName, module, mlirTy, *processed_args)
        return result

    def beta_reduction(self, isEntryPoint, *args):
        """
        Perform beta reduction on this kernel decorator in the current calling
        context. We are primary concerned with resolving the lambda lifted
        arguments, but the formal arguments may be supplied as well.

        This beta reduction may happen in a context that is earlier than the
        actual call to the decorator. While this loses some of Python's
        intrinsic dynamism, it allows Python kernels to be specialized and
        passed to algorithms written in C++ that call back to these Python
        kernels in a functional composition.
        """
        processed_args, module = self.prepare_call(*args)
        mlirTy = self.handle_call_results()
        return cudaq_runtime.marshal_and_retain_module(self.uniqName, module,
                                                       mlirTy, isEntryPoint,
                                                       *processed_args)

    def delete_cache_execution_engine(self, key):
        """
        Delete the `ExecutionEngine` cache given by a cache key.
        """
        cudaq_runtime.delete_cache_execution_engine(key)

    def process_argument(self, arg, arg_type):
        if isa_kernel_decorator(arg):
            captured_args = arg.resolve_captured_arguments()
            return DecoratorCapture(arg, captured_args)

        arg = self.convertStringsToPauli(arg)
        mlirType = mlirTypeFromPyType(type(arg),
                                      getMLIRContext(),
                                      argInstance=arg,
                                      argTypeToCompareTo=arg_type)

        # Check error conditions before proceeding.
        if cc.CallableType.isinstance(mlirType):
            emitFatalError(
                f"Argument has callable type but the argument ({arg}) is not "
                f"a kernel decorator.")

        if self.isCastablePyType(mlirType, arg_type):
            return self.castPyType(mlirType, arg_type, arg)

        if mlirType != arg_type:
            emitFatalError(f"Invalid runtime argument type. Argument of type "
                           f"{mlirTypeToPyType(mlirType)} was provided, but "
                           f"{mlirTypeToPyType(arg_type)} was expected.")

        # Convert `numpy` arrays to lists
        if cc.StdvecType.isinstance(mlirType) and hasattr(arg, "tolist"):
            if arg.ndim != 1:
                emitFatalError(
                    f"CUDA-Q kernels only support array arguments from NumPy "
                    f"that are one dimensional (found shape = {arg.shape}).")
            return arg.tolist()
        else:
            return arg

    def _add_global_scoped_var(self, name, var):
        self.globalScopedVars[name] = var

        # Register any external class types that may be used in the kernel
        # definition
        if isinstance(var, type) and hasattr(var, '__annotations__'):
            globalRegisteredTypes.registerClass(name, var)


def mk_decorator(builder):
    """
    Make a kernel decorator object from a kernel builder object to make any code
    that handles both CUDA-Q kernel object classes more unified.
    """
    return PyKernelDecorator(None,
                             module=builder.module,
                             kernelName=builder.uniqName)


def kernel(function=None, **kwargs):
    """
    The `cudaq.kernel` represents the CUDA-Q language function attribute that
    programmers leverage to indicate the following function is a CUDA-Q kernel
    and should be compile and executed on an available quantum coprocessor.

    Verbose logging can be enabled via `verbose=True`. 
    """
    if function:
        return PyKernelDecorator(function)
    else:

        def wrapper(function):
            return PyKernelDecorator(function, **kwargs)

        return wrapper


def isa_kernel_decorator(object):
    """
    Return True if and only if object is an instance of PyKernelDecorator.
    """
    return isinstance(object, PyKernelDecorator)


def _get_source(function):
    if function is None:
        return None, None
    # Get the function source location
    location = (inspect.getfile(function), inspect.getsourcelines(function)[1])
    # Get the function source
    src = inspect.getsource(function)
    # Strip off the extra tabs
    leadingSpaces = len(src) - len(src.lstrip())
    src = '\n'.join([line[leadingSpaces:] for line in src.split('\n')])
    return src, location


def _recover_defining_frame():
    """
    Walk back from the current frame until we find the first frame that is not
    in the current module. We take that as the "defining scope" of the
    decorator.
    """
    stack = inspect.stack()
    current_module = get_module_name(stack[0].frame)

    assert current_module == "cudaq.kernel.kernel_decorator"

    try:
        for frameinfo in stack:
            # Walk back until we leave the decorator module
            if get_module_name(frameinfo.frame) != current_module:
                return frameinfo.frame
    finally:
        del stack
        del frameinfo
    return None


def _parse_ast(funcSrc: str, verbose: bool = False):
    astModule = ast.parse(funcSrc)
    if verbose:
        try:
            from astpretty import pprint
            pprint(astModule.body[0])
        except ImportError:
            pass
    return astModule
