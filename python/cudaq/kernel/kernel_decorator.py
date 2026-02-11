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
from cudaq.kernel.utils import emitWarning
import numpy as np
import sys

from cudaq.handlers import get_target_handler
from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime
from cudaq.mlir.dialects import cc, func
from cudaq.mlir.ir import (ComplexType, F32Type, F64Type, FunctionType,
                           IntegerType, NoneType, TypeAttr, UnitAttr, Module)
from .analysis import HasReturnNodeVisitor
from .ast_bridge import compile_to_mlir
from .utils import (emitFatalError, emitErrorIfInvalidPauli,
                    globalRegisteredTypes, mlirTypeFromPyType, mlirTypeToPyType,
                    nvqppPrefix, getMLIRContext, recover_func_op,
                    recover_value_of, recover_calling_module)

# This file implements the decorator mechanism needed to JIT compile CUDA-Q
# kernels. It exposes the cudaq.kernel() decorator which hooks us into the JIT
# compilation infrastructure which maps the AST representation to an MLIR
# representation and ultimately executable code.


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
    the decorated function as input and optionally lowers its  AST 
    representation to executable code via MLIR. This decorator enables full JIT
    compilation mode, where the function is lowered to an MLIR representation.

    This decorator exposes a call overload that executes the code via the 
    MLIR `ExecutionEngine` for the MLIR mode. 
    """

    def __init__(self,
                 function,
                 verbose=False,
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
        # The `qkeModule` will be the quake target independent ModuleOp
        self.qkeModule = None
        # The `nvqModule` will be (if present) the default simulation ModuleOp
        self.nvqModule = None
        self.defModule = _recover_module('cudaq.kernel.kernel_decorator')

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
                self.__dict__.update(vars(decorator))
            else:
                self.uniqueId = int(kernelName.split("..0x")[1], 16)
                self.uniqName = kernelName
                self.liftedArgs = []
                self.firstLiftedPos = None

            self.qkeModule = module
            self._parse_signature_from_mlir()
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

            self._parse_ast()
            self._parse_signature_from_python()
            self.pre_compile()

    def __del__(self):
        # explicitly call `del` on the MLIR `ModuleOp` wrappers.
        if self.qkeModule:
            del self.qkeModule
        if self.nvqModule:
            del self.nvqModule

    def signatureWithCallables(self):
        """
        returns True if and only if the entry-point contains callable arguments
        and/or return values.
        """
        name = nvqppPrefix + self.uniqName
        funcOp = recover_func_op(self.qkeModule, name)
        attr = TypeAttr(funcOp.attributes['function_type'])
        funcTy = FunctionType(attr.value)
        for ty in funcTy.inputs + funcTy.results:
            if cc.CallableType.isinstance(ty) or FunctionType.isinstance(ty):
                return True
        return False

    def pre_compile(self):
        """
        Compile the Python AST to portable Quake.
        """

        # If this target requires library mode, do not compile it to MLIR.
        # TODO: this should always compile to MLIR.
        handler = get_target_handler()
        if handler.skip_compilation():
            return

        # Otherwise, `precompile` the kernel to portable MLIR.
        if self.qkeModule:
            raise RuntimeError(self.name + " was already compiled")
        self.uniqueId = id(self)
        self.uniqName = self.name + ".." + hex(self.uniqueId)
        self.qkeModule, self.argTypes, self.liftedArgs, self.firstLiftedPos = compile_to_mlir(
            id(self),
            self.astModule,
            verbose=self.verbose,
            returnType=self.return_type,
            location=self.location,
            parentVariables=self.globalScopedVars,
            preCompile=True,
            kernelName=self.name,
            kernelModuleName=self.kernelModuleName)

        if (cudaq_runtime.is_current_target_full_qir() and
                not self.signatureWithCallables()):
            resMod = self.convert_to_full_qir([])
            if not self.nvqModule:
                self.nvqModule = resMod

    def compile(self):
        return

    def convert_to_full_qir(self, vals):
        return self.lower_quake_to_codegen(vals)

    def lower_quake_to_codegen(self, argValues):
        """
        Take the quake code as input and lower it to be ready for final code
        generation. If argument values are provided, we run argument synthesis
        and specialize this instance of the kernel.
        """
        if not self.qkeModule:
            emitFatalError(f"no module in kernel decorator {self.name}")
        result = cudaq_runtime.cloneModule(self.qkeModule)

        if argValues:
            if len(self.argTypes) != len(argValues):
                emitFatalError("wrong number of arguments provided")

        return result

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
        Return the MLIR Module string representation for this kernel.
        """
        if self.qkeModule:
            return str(self.qkeModule)
        return "The decorator " + hex(id(self)) + " is malformed"

    def enable_return_to_log(self):
        """
        Enable translation from `return` statements to QIR output log
        """
        self.qkeModule.operation.attributes.__setitem__(
            'quake.cudaq_run', UnitAttr.get(context=self.qkeModule.context))

    def _repr_svg_(self):
        """
        Return the SVG representation of `self` (:class:`PyKernelDecorator`).
        This assumes no arguments are required to execute the kernel, and
        `latex` (with `quantikz` package) and `dvisvgm` are installed, and the
        temporary directory is writable.  If any of these assumptions fail,
        returns None.
        """
        if self.argTypes is None or len(self.argTypes) != 0:
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
        obj['signature'] = {
            k: PyKernelDecorator.type_to_str(v)
            for k, v in self.signature.items()
        }
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
                                 signature=j['signature'],
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
        if self.liftedArgs:
            return self.firstLiftedPos
        return len(self.argTypes)

    def handle_call_arguments(self, *args):
        """
        Resolve all the arguments at the call site for this decorator.
        """
        # Process all the normal arguments
        processedArgs = []
        callingModule = recover_calling_module()
        self.process_arguments_to_call(processedArgs, callingModule, args)

        # Process any lifted arguments
        if self.liftedArgs:
            for j, a in enumerate(self.liftedArgs):
                i = self.firstLiftedPos + j
                # get the value associated with the variable named "a" in the
                # current context.
                if isinstance(a, dict) and a.get('linkedKernel'):
                    # Lifted argument is a registered C++ kernel, load and capture it
                    [linkedKernel,
                     maybeCode] = cudaq_runtime.checkRegisteredCppDeviceKernel(
                         self.qkeModule, a['linkedKernel'])
                    qkeModule = Module.parse(maybeCode,
                                             context=self.qkeModule.context)
                    processedArgs.append(
                        LinkedKernelCapture(linkedKernel, qkeModule))
                else:
                    a_value = recover_value_of(a, None)
                    self.process_argument(processedArgs, i, a_value,
                                          callingModule)

        # Specialize quake code via argument synthesis, lower to full QIR.
        specialized_module = self.convert_to_full_qir(processedArgs)
        return specialized_module, processedArgs

    def get_none_type(self):
        return NoneType.get(self.qkeModule.context)

    def handle_call_results(self):
        if not self.return_type:
            return self.get_none_type()
        return mlirTypeFromPyType(self.return_type, self.qkeModule.context)

    def launch_args_required(self):
        """
        This is a deeper query on the quake module. The quake module may have
        been specialized such that none of the arguments are, in fact, required
        to be provided in order to run the kernel. (Argument synthesis.)
        
        This will analyze the designated entry-point kernel for the quake module
        and determine if any arguments are used and return the number used.
        """
        if len(self.argTypes) == 0:
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

        specialized_module, processedArgs = self.handle_call_arguments(*args)
        mlirTy = self.handle_call_results()
        result = cudaq_runtime.marshal_and_launch_module(
            self.uniqName, specialized_module, mlirTy, *processedArgs)
        return result

    def beta_reduction(self, *args):
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
        specialized_module, processedArgs = self.handle_call_arguments(*args)
        mlirTy = self.handle_call_results()
        return cudaq_runtime.marshal_and_retain_module(self.uniqName,
                                                       specialized_module,
                                                       mlirTy, *processedArgs)

    def delete_cache_execution_engine(self, key):
        """
        Delete the `ExecutionEngine` cache given by a cache key.
        """
        cudaq_runtime.delete_cache_execution_engine(key)

    def resolve_decorator_at_callsite(self, callingMod):
        # Resolve all lifted arguments for `self`.
        processedArgs = []
        for j, la in enumerate(self.liftedArgs):
            i = self.firstLiftedPos + j
            resMod = None
            if callingMod != self.defModule:
                resMod = self.defModule
            if isinstance(la, dict) and la.get('linkedKernel'):
                # Lifted argument is a registered C++ kernel, load and capture it
                [linkedKernel,
                 maybeCode] = cudaq_runtime.checkRegisteredCppDeviceKernel(
                     self.qkeModule, la['linkedKernel'])
                qkeModule = Module.parse(maybeCode,
                                         context=self.qkeModule.context)
                processedArgs.append(
                    LinkedKernelCapture(linkedKernel, qkeModule))
            else:
                la_value = recover_value_of(la, resMod)
                self.process_argument(processedArgs, i, la_value, callingMod)
        return DecoratorCapture(self, processedArgs)

    def process_argument(self, processedArgs, i, arg, callingMod):
        if isa_kernel_decorator(arg):
            rdr = arg.resolve_decorator_at_callsite(callingMod)
            processedArgs.append(rdr)
            return

        arg = self.convertStringsToPauli(arg)
        mlirType = mlirTypeFromPyType(type(arg),
                                      getMLIRContext(),
                                      argInstance=arg,
                                      argTypeToCompareTo=self.argTypes[i])

        # Check error conditions before proceeding.
        if cc.CallableType.isinstance(mlirType):
            emitFatalError(
                f"Argument has callable type but the argument ({arg}) is not "
                f"a kernel decorator.")

        if self.isCastablePyType(mlirType, self.argTypes[i]):
            processedArgs.append(
                self.castPyType(mlirType, self.argTypes[i], arg))
            mlirType = self.argTypes[i]
            return

        if mlirType != self.argTypes[i]:
            emitFatalError(
                f"Invalid runtime argument type. Argument of type "
                f"{mlirTypeToPyType(mlirType)} was provided, but "
                f"{mlirTypeToPyType(self.argTypes[i])} was expected.")

        # Convert `numpy` arrays to lists
        if cc.StdvecType.isinstance(mlirType) and hasattr(arg, "tolist"):
            if arg.ndim != 1:
                emitFatalError(
                    f"CUDA-Q kernels only support array arguments from NumPy "
                    f"that are one dimensional (input argument {i} has shape ="
                    f" {arg.shape}).")
            processedArgs.append(arg.tolist())
        else:
            processedArgs.append(arg)

    def process_arguments_to_call(self, processedArgs, resMod, args):
        for i, arg in enumerate(args):
            self.process_argument(processedArgs, i, arg, resMod)

    def _add_global_scoped_var(self, name, var):
        self.globalScopedVars[name] = var

        # Register any external class types that may be used in the kernel
        # definition
        if isinstance(var, type) and hasattr(var, '__annotations__'):
            globalRegisteredTypes.registerClass(name, var)

    def _parse_signature_from_python(self):
        if not self.kernelFunction:
            return

        self.signature = inspect.getfullargspec(self.kernelFunction).annotations
        self.arguments = [
            (k, v) for k, v in self.signature.items() if k != 'return'
        ]

        # Validate that we have a return type annotation if necessary
        hasRetNodeVis = HasReturnNodeVisitor()
        hasRetNodeVis.visit(self.astModule)
        if hasRetNodeVis.hasReturnNode and 'return' not in self.signature:
            emitFatalError('CUDA-Q kernel has return statement '
                           'but no return type annotation.')

        self.return_type = self.signature.get('return', None)

    def _parse_signature_from_mlir(self):
        funcOp = recover_func_op(self.qkeModule, nvqppPrefix + self.uniqName)
        fnTy = FunctionType(TypeAttr(funcOp.attributes['function_type']).value)
        self.argTypes = fnTy.inputs
        self.return_type = fnTy.results[0] if fnTy.results else None

    def _parse_ast(self):
        self.astModule = ast.parse(self.funcSrc)
        if self.verbose:
            try:
                from astpretty import pprint
                pprint(self.astModule.body[0])
            except ImportError:
                pass


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


def _recover_module(name):
    """
    All decorators are defined in this (`kernel_decorator.py`) module.
    Strip all the frames from this module until we find the next
    enclosing frame.
    """

    def frame_and_mod(fr):
        if fr is None:
            return None
        mod = inspect.getmodule(fr)
        if mod is not None and getattr(mod, "__name__", None):
            return mod.__name__
        # Fallback to search module in `globals`
        return fr.f_globals.get("__name__")

    frame = inspect.currentframe()
    try:
        # Walk back until we enter the decorator module
        while frame is not None and frame_and_mod(frame) != name:
            frame = frame.f_back

        if frame is None:
            return None

        # Walk back until we leave the decorator module
        while frame is not None and frame_and_mod(frame) == name:
            frame = frame.f_back

        if frame is None:
            return None

        mod = inspect.getmodule(frame)
        if mod is not None:
            return mod

        # Resolve by `globals` name
        return sys.modules.get(frame.f_globals.get("__name__"))
    finally:
        del frame
