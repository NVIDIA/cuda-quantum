# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations
import ast
import inspect
import re
import sys
import traceback
import importlib
import numpy as np
from typing import get_origin, get_args, Callable, List
import types
from cudaq.mlir.execution_engine import ExecutionEngine
from cudaq.mlir.dialects import func
from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime
from cudaq.mlir.dialects import quake, cc
from cudaq.mlir.ir import (ComplexType, F32Type, F64Type, IntegerType, Context,
                           Module)
from cudaq.mlir._mlir_libs._quakeDialects import register_all_dialects

State = cudaq_runtime.State
qvector = cudaq_runtime.qvector
qview = cudaq_runtime.qview
qubit = cudaq_runtime.qubit
pauli_word = cudaq_runtime.pauli_word
qreg = qvector

nvqppPrefix = '__nvqpp__mlirgen__'

ahkPrefix = '__analog_hamiltonian_kernel__'

# Keep a global registry of all kernel Python AST modules keyed on their name
# (without `__nvqpp__mlirgen__` prefix). The values in this dictionary are a
# tuple of the AST module and the source code location for the kernel.
globalAstRegistry = {}

# Keep a global registry of all registered custom operations.
globalRegisteredOperations = {}

# Keep a global registry of any custom data types
globalRegisteredTypes = cudaq_runtime.DataClassRegistry


def getMLIRContext():
    """
    This code creates an MLIRContext singleton for this python process. We do
    not want to have a brand new context every time Python does something with a
    kernel.
    """
    global cudaq__global_mlir_context
    try:
        cudaq__global_mlir_context
    except NameError:
        cudaq__global_mlir_context = Context()
        register_all_dialects(cudaq__global_mlir_context)
        quake.register_dialect(context=cudaq__global_mlir_context)
        cc.register_dialect(context=cudaq__global_mlir_context)
        cudaq_runtime.registerLLVMDialectTranslation(cudaq__global_mlir_context)
    return cudaq__global_mlir_context


class Initializer:
    # We need static initializers to run in the CAPI `ExecutionEngine`, so here
    # we run a simple JIT compile at global scope.
    def initialize(self):
        self.context = getMLIRContext()
        self.module = Module.parse("llvm.func @none() { llvm.return }",
                                   context=self.context)
        ExecutionEngine(self.module)


try:
    globalExecutionEngineInitialized
except NameError:
    globalExecutionEngineInitialized = True
    try:
        Initializer().initialize()
    except Exception as e:
        print("python failed to load the execution engine", file=sys.stderr)
        sys.exit()


class Color:
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


# Name of module attribute to recover the name of the entry-point for the python
# kernel decorator.  The associated StringAttr is *without* the `nvqppPrefix`.
cudaq__unique_attr_name = "cc.python_uniqued"


def recover_func_op(module, name):
    for op in module.body:
        if isinstance(op, func.FuncOp):
            if op.sym_name.value == name:
                return op
    return None


def recover_calling_module():

    def frame_and_mod(fr):
        if fr is None:
            return None
        mod = inspect.getmodule(fr)
        if mod is not None and getattr(mod, "__name__", None):
            return mod.__name__
        # Fallback for notebooks to search module in `globals`
        return fr.f_globals.get("__name__")

    frame = inspect.currentframe()
    try:
        frame = frame.f_back
        name = frame_and_mod(frame)

        while frame is not None and name is not None and (
                name.startswith("cudaq.kernel") or
                name.startswith("cudaq.runtime")):
            frame = frame.f_back
            name = frame_and_mod(frame)

        if frame is None:
            return None

        # A real module object if available
        mod = inspect.getmodule(frame)
        if mod is not None:
            return mod

        # Resolve by `globals` name
        return sys.modules.get(frame.f_globals.get("__name__"))
    finally:
        del frame


def resolve_qualified_symbol(y):
    """
    If `y` is a qualified symbol (containing a '.' in the name), then resolve
    the symbol to the kernel decorator object. Returns `None` if the qualified
    name cannot be resolved to a kernel decorator object.

    For legacy reasons, this supports improper use of qualified names. For
    example, in the module `cudaq.kernels.uccsd` there is a kernel named
    `uccsd`. However, legacy tests just use the module name and omit the kernel
    decorator name.
    """
    parts = y.split('.')
    # Walk the path right to left to resolve the longest path name as soon as
    # possible. (See the python documentation on `importlib`. This is the
    # algorithm.)
    for i in range(len(parts), 0, -1):
        modName = ".".join(parts[:i])
        try:
            mod = importlib.import_module(modName)
        except ModuleNotFoundError:
            continue
        obj = mod
        try:
            for attr in parts[i:]:
                obj = getattr(obj, attr)
        except AttributeError:
            return None
        from .kernel_decorator import isa_kernel_decorator
        if not isa_kernel_decorator(obj):
            # FIXME: Legacy hack to support incorrect Python spellings of kernel
            # names.
            try:
                obj = getattr(obj, parts[-1])
            except AttributeError:
                pass
        return obj if isa_kernel_decorator(obj) else None
    return None


def recover_value_of_or_none(name, resMod):
    """
    Recover the Python value of the symbol `name` from the enclosing context.
    The enclosing context is the context in which the `PyKernelDecorator`
    object's `__init__` or `__call__` method were invoked.

    If `name` is qualified, then lookup the symbol in the module that is
    specified in the name itself.

    If there is a resolve-in module, `resMod`, then resolve the symbol in the
    given module.

    Otherwise, the symbol is neither qualified nor is there another module to
    resolve the name in.  So perform a normal LEGB resolution of the symbol in
    the current set of stack frames. (Actually, EGB since the symbol cannot be
    local.)

    Note that this need not be used with a `PyKernel` object as the semantics of
    the kernel builder presumes immediate lookup and resolution of all symbols
    during construction.
    """
    from .kernel_decorator import isa_kernel_decorator

    if '.' in name:
        return resolve_qualified_symbol(name)

    if resMod:
        return resMod.__dict__.get(name, None)

    def drop_front():
        drop = 0
        for frameinfo in inspect.stack():
            frame = frameinfo.frame
            if 'self' in frame.f_locals:
                if isa_kernel_decorator(frame.f_locals['self']):
                    return drop
            drop = drop + 1
        return drop

    drop = drop_front()
    for frameinfo in inspect.stack()[drop:]:
        frame = frameinfo.frame
        if name in frame.f_locals:
            return frame.f_locals[name]
        if name in frame.f_globals:
            return frame.f_globals[name]
    return None


def is_recovered_value_ok(result):
    try:
        if result != None:
            return True
    except ValueError:
        # `nd.array` values raise `ValueError` with the above `if result` but
        # are otherwise legit here.
        return True
    return False


def recover_value_of(name, resMod):
    result = recover_value_of_or_none(name, resMod)
    if is_recovered_value_ok(result):
        return result
    raise RuntimeError("'" + name + "' is not available in this scope.")


def emitFatalError(msg):
    """
    Emit a fatal error diagnostic. The goal here is to 
    retain the input message, but also provide the offending 
    source location by inspecting the stack trace. 
    """
    print(Color.BOLD, end='')
    try:
        # Raise the exception so we can get the stack trace to inspect
        raise RuntimeError(msg)
    except RuntimeError:
        # Immediately grab the exception and analyze the stack trace, getting
        # the source location and construct a new error diagnostic.
        cached = sys.tracebacklimit
        sys.tracebacklimit = None
        offendingSrc = traceback.format_stack()
        sys.tracebacklimit = cached
        if len(offendingSrc):
            msg = (Color.RED + "error: " + Color.END + Color.BOLD + msg +
                   Color.END + '\n\nOffending code:\n' + offendingSrc[0])
    raise RuntimeError(msg)


def emitWarning(msg):
    """
    Emit a warning, providing the user with source file information and
    the offending code.
    """
    print(Color.BOLD, end='')
    try:
        # Raise the exception so we can get the stack trace to inspect
        raise RuntimeError(msg)
    except RuntimeError:
        # Immediately grab the exception and analyze the stack trace, getting
        # the source location and construct a new error diagnostic
        cached = sys.tracebacklimit
        sys.tracebacklimit = None
        offendingSrc = traceback.format_stack()
        sys.tracebacklimit = cached
        if len(offendingSrc):
            msg = (Color.YELLOW + "error: " + Color.END + Color.BOLD + msg +
                   Color.END + '\n\nOffending code:\n' + offendingSrc[0])


def mlirTryCreateStructType(mlirEleTypes, name="tuple", context=None):
    """
    Creates either a `quake.StruqType` or a `cc.StructType` used to represent 
    tuples and `dataclass` structs of quantum and classical types. Returns
    None if the given element types don't satisfy the restrictions imposed
    on these types.
    """

    def isQuantumType(ty):
        return quake.RefType.isinstance(ty) or quake.VeqType.isinstance(
            ty) or quake.StruqType.isinstance(ty)

    numQuantumMembers = sum((isQuantumType(t) for t in mlirEleTypes))
    if numQuantumMembers == 0:
        if any((cc.PointerType.isinstance(t) for t in mlirEleTypes)):
            return None
        return cc.StructType.getNamed(name, mlirEleTypes, context=context)
    if numQuantumMembers != len(mlirEleTypes) or \
        any((quake.StruqType.isinstance(t) for t in mlirEleTypes)):
        return None
    return quake.StruqType.getNamed(name, mlirEleTypes, context=context)


def mlirTypeFromAnnotation(annotation, ctx, raiseError=False):
    """
    Return the MLIR Type corresponding to the given kernel function argument
    type annotation.  Throws an exception if the programmer did not annotate
    function argument types.
    """

    localEmitFatalError = emitFatalError
    if raiseError:
        # Client calling this will handle errors
        def emitFatalErrorOverride(msg):
            raise RuntimeError(msg)

        localEmitFatalError = emitFatalErrorOverride

    if annotation == None:
        localEmitFatalError(
            'cudaq.kernel functions must have argument type annotations.')

    with ctx:

        if hasattr(annotation, 'attr') and hasattr(annotation.value, 'id'):
            if annotation.value.id == 'cudaq':
                if annotation.attr in ['qview', 'qvector']:
                    return quake.VeqType.get()
                if annotation.attr in ['State']:
                    return cc.PointerType.get(cc.StateType.get())
                if annotation.attr == 'qubit':
                    return quake.RefType.get()
                if annotation.attr == 'pauli_word':
                    return cc.CharspanType.get()

            if annotation.value.id in ['numpy', 'np']:
                if annotation.attr in ['array', 'ndarray']:
                    return cc.StdvecType.get(F64Type.get())
                if annotation.attr == 'complex128':
                    return ComplexType.get(F64Type.get())
                if annotation.attr == 'complex64':
                    return ComplexType.get(F32Type.get())
                if annotation.attr == 'float64':
                    return F64Type.get()
                if annotation.attr == 'float32':
                    return F32Type.get()
                if annotation.attr == 'int64':
                    return IntegerType.get_signless(64)
                if annotation.attr == 'int32':
                    return IntegerType.get_signless(32)
                if annotation.attr == 'int16':
                    return IntegerType.get_signless(16)
                if annotation.attr == 'int8':
                    return IntegerType.get_signless(8)

        if isinstance(annotation,
                      ast.Subscript) and annotation.value.id == 'Callable':
            if not hasattr(annotation, 'slice'):
                localEmitFatalError(
                    f"Callable type must have signature specified ("
                    f"{ast.unparse(annotation) if hasattr(ast, 'unparse') else annotation})."
                )

            if hasattr(annotation.slice, 'elts') and len(
                    annotation.slice.elts) == 2:
                args = annotation.slice.elts[0]
                ret = annotation.slice.elts[1]
            elif hasattr(annotation.slice, 'value') and hasattr(
                    annotation.slice.value, 'elts') and len(
                        annotation.slice.value.elts) == 2:
                args = annotation.slice.value.elts[0]
                ret = annotation.slice.value.elts[1]
            else:
                localEmitFatalError(
                    f"Unable to get list elements when inferring type from annotation ("
                    f"{ast.unparse(annotation) if hasattr(ast, 'unparse') else annotation})."
                )
            argTypes = [mlirTypeFromAnnotation(a, ctx) for a in args.elts]
            if not isinstance(ret, ast.Constant) or ret.value:
                localEmitFatalError("passing kernels as arguments that return"
                                    " a value is not currently supported")
            return cc.CallableType.get(ctx, argTypes, [])

        if isinstance(annotation,
                      ast.Subscript) and (annotation.value.id == 'list' or
                                          annotation.value.id == 'List'):
            if not hasattr(annotation, 'slice'):
                localEmitFatalError(
                    f"list subscript missing slice node ("
                    f"{ast.unparse(annotation) if hasattr(ast, 'unparse') else annotation})."
                )

            eleTypeNode = annotation.slice
            # expected that slice is a Name node
            listEleTy = mlirTypeFromAnnotation(eleTypeNode, ctx)
            return cc.StdvecType.get(listEleTy)

        if isinstance(annotation,
                      ast.Subscript) and (annotation.value.id == 'tuple' or
                                          annotation.value.id == 'Tuple'):

            if not hasattr(annotation, 'slice'):
                localEmitFatalError(
                    f"tuple subscript missing slice node ("
                    f"{ast.unparse(annotation) if hasattr(ast, 'unparse') else annotation})."
                )

            # slice is an `ast.Tuple` of type annotations
            elements = None
            if hasattr(annotation.slice, 'elts'):
                elements = annotation.slice.elts
            else:
                localEmitFatalError(
                    f"Unable to get tuple elements when inferring type from "
                    f"annotation ({ast.unparse(annotation) if hasattr(ast, 'unparse') else annotation})."
                )

            eleTypes = [mlirTypeFromAnnotation(v, ctx) for v in elements]
            tupleTy = mlirTryCreateStructType(eleTypes)
            if tupleTy is None:
                localEmitFatalError("Hybrid quantum-classical data types and "
                                    "nested quantum structs are not allowed.")
            return tupleTy

        if hasattr(annotation, 'id'):
            id = annotation.id
        elif hasattr(annotation, 'value'):
            if hasattr(annotation.value, 'id'):
                id = annotation.value.id
            elif hasattr(annotation.value, 'value') and hasattr(
                    annotation.value.value, 'id'):
                id = annotation.value.value.id
            else:
                localEmitFatalError(
                    f"{ast.unparse(annotation) if hasattr(ast, 'unparse') else annotation}"
                    f" is not yet a supported type (could not infer type name)."
                )
        else:
            localEmitFatalError(
                f"{ast.unparse(annotation) if hasattr(ast, 'unparse') else annotation}"
                f" is not a supported type yet (could not infer type name).")

        if id == 'list' or id == 'List':
            localEmitFatalError(
                'list argument annotation must provide element type, e.g. list[float] instead of list.'
            )

        if id == 'int':
            return IntegerType.get_signless(64)

        if id == 'float':
            return F64Type.get()

        if id == 'bool':
            return IntegerType.get_signless(1)

        if id == 'complex':
            return ComplexType.get(F64Type.get())

        if isinstance(annotation, ast.Attribute):
            # in this case we might have `mod1.mod2...mod3.UserType`
            # slurp up the path to the type
            id = annotation.attr

        # One final check to see if this is a custom data type.
        if id in globalRegisteredTypes.classes:
            pyType, memberTys = globalRegisteredTypes.getClassAttributes(id)
            structTys = [
                mlirTypeFromPyType(v, ctx) for _, v in memberTys.items()
            ]

            if '__slots__' not in pyType.__dict__:
                emitWarning(
                    "Adding new fields in data classes is not yet supported. "
                    "The dataclass must be declared with @dataclass(slots=True)"
                    " or @dataclasses.dataclass(slots=True).")

            if len({
                    k: v
                    for k, v in pyType.__dict__.items()
                    if not (k.startswith('__') and k.endswith('__')) and
                    isinstance(v, types.FunctionType)
            }) != 0:
                localEmitFatalError(
                    'struct types with user specified methods are not allowed.')

            tupleTy = mlirTryCreateStructType(structTys, name=id)
            if tupleTy is None:
                localEmitFatalError(
                    "Hybrid quantum-classical data types and nested "
                    "quantum structs are not allowed.")
            return tupleTy

    localEmitFatalError(
        f"{ast.unparse(annotation) if hasattr(ast, 'unparse') else annotation}"
        f" is not a supported type.")


def pyInstanceFromName(name: str):
    if name == 'bool':
        return bool(False)
    if name == 'int':
        return int(0)
    if name in ['numpy.int8', 'np.int8']:
        return np.int8(0)
    if name in ['numpy.int16', 'np.int16']:
        return np.int16(0)
    if name in ['numpy.int32', 'np.int32']:
        return np.int32(0)
    if name in ['numpy.int64', 'np.int64']:
        return np.int64(0)
    if name == 'int':
        return int(0)
    if name == 'float':
        return float(0.0)
    if name in ['numpy.float32', 'np.float32']:
        return np.float32(0.0)
    if name in ['numpy.float64', 'np.float64']:
        return np.float64(0.0)
    if name == 'complex':
        return 0j
    if name == 'pauli_word':
        return pauli_word('')
    if name in ['numpy.complex128', 'np.complex128']:
        return np.complex128(0.0)
    if name in ['numpy.complex64', 'np.complex64']:
        return np.complex64(0.0)


def mlirTypeFromPyType(argType, ctx, **kwargs):
    if argType in [int, np.int64]:
        return IntegerType.get_signless(64, ctx)
    if argType == np.int32:
        return IntegerType.get_signless(32, ctx)
    if argType == np.int16:
        return IntegerType.get_signless(16, ctx)
    if argType == np.int8:
        return IntegerType.get_signless(8, ctx)
    if argType in [float, np.float64]:
        return F64Type.get(ctx)
    if argType == np.float32:
        return F32Type.get(ctx)
    if argType == bool:
        return IntegerType.get_signless(1, ctx)
    if argType == complex:
        return ComplexType.get(mlirTypeFromPyType(float, ctx))
    if argType == np.complex128:
        return ComplexType.get(mlirTypeFromPyType(np.float64, ctx))
    if argType == np.complex64:
        return ComplexType.get(mlirTypeFromPyType(np.float32, ctx))
    if argType == pauli_word:
        return cc.CharspanType.get(ctx)
    if argType == State:
        return cc.PointerType.get(cc.StateType.get(ctx), ctx)

    if get_origin(argType) == list:
        pyEleTy = get_args(argType)
        if len(pyEleTy) == 1:
            eleTy = mlirTypeFromPyType(pyEleTy[0], ctx)
            return cc.StdvecType.get(eleTy, ctx)
        argType = list

    if argType in [list, np.ndarray, List]:
        if 'argInstance' not in kwargs:
            return cc.StdvecType.get(mlirTypeFromPyType(float, ctx), ctx)
        if argType != np.ndarray:
            if kwargs['argInstance'] == None:
                return cc.StdvecType.get(mlirTypeFromPyType(float, ctx), ctx)

        argInstance = kwargs['argInstance']
        argTypeToCompareTo = kwargs[
            'argTypeToCompareTo'] if 'argTypeToCompareTo' in kwargs else None

        if len(argInstance) == 0:
            if argTypeToCompareTo == None:
                emitFatalError('Cannot infer runtime argument type')

            eleTy = cc.StdvecType.getElementType(argTypeToCompareTo)
            return cc.StdvecType.get(eleTy, ctx)

        if isinstance(argInstance[0], list):
            return cc.StdvecType.get(
                mlirTypeFromPyType(
                    type(argInstance[0]),
                    ctx,
                    argInstance=argInstance[0],
                    argTypeToCompareTo=cc.StdvecType.getElementType(
                        argTypeToCompareTo)), ctx)

        return cc.StdvecType.get(mlirTypeFromPyType(type(argInstance[0]), ctx),
                                 ctx)

    if get_origin(argType) == tuple:
        eleTypes = []
        for pyEleTy in get_args(argType):
            eleTypes.append(mlirTypeFromPyType(pyEleTy, ctx))
        tupleTy = mlirTryCreateStructType(eleTypes, context=ctx)
        if tupleTy is None:
            emitFatalError("Hybrid quantum-classical data types and nested "
                           "quantum structs are not allowed.")
        return tupleTy

    if (argType == tuple):
        argInstance = kwargs['argInstance']
        if argInstance == None or (len(argInstance) == 0):
            emitFatalError(f'Cannot infer runtime argument type for {argType}')
        argTypeToCompareTo = (kwargs['argTypeToCompareTo']
                              if 'argTypeToCompareTo' in kwargs else None)
        if argTypeToCompareTo is None:
            eleTypes = [
                mlirTypeFromPyType(type(ele), ctx) for ele in argInstance
            ]
            tupleTy = mlirTryCreateStructType(eleTypes, context=ctx)
        else:
            tupleTy = argTypeToCompareTo
        if tupleTy is None:
            emitFatalError("Hybrid quantum-classical data types and nested "
                           "quantum structs are not allowed.")
        return tupleTy

    if argType == qvector or argType == qreg or argType == qview:
        return quake.VeqType.get(context=ctx)
    if argType == qubit:
        return quake.RefType.get(ctx)
    if argType == pauli_word:
        return cc.CharspanType.get(ctx)

    if 'argInstance' in kwargs:
        argInstance = kwargs['argInstance']
        if isinstance(argInstance, Callable):
            return cc.CallableType.get(ctx, argInstance.argTypes, [])

    for name in globalRegisteredTypes.classes:
        customTy, memberTys = globalRegisteredTypes.getClassAttributes(name)
        if argType == customTy:
            structTys = [
                mlirTypeFromPyType(v, ctx) for _, v in memberTys.items()
            ]
            numQuantumMemberTys = sum([
                1 if
                (quake.RefType.isinstance(ty) or quake.VeqType.isinstance(ty) or
                 quake.StruqType.isinstance(ty)) else 0 for ty in structTys
            ])
            numStruqMemberTys = sum([
                1 if (quake.StruqType.isinstance(ty)) else 0 for ty in structTys
            ])
            if numQuantumMemberTys != 0:  # we have quantum member types
                if numQuantumMemberTys != len(structTys):
                    emitFatalError(
                        f'hybrid quantum-classical data types not allowed')
                if numStruqMemberTys != 0:
                    emitFatalError(
                        f'recursive quantum struct types not allowed.')
                return quake.StruqType.getNamed(name, structTys, ctx)

            return cc.StructType.getNamed(name, structTys, ctx)

    if 'argInstance' not in kwargs:
        if argType == list[int]:
            return cc.StdvecType.get(mlirTypeFromPyType(int, ctx), ctx)
        if argType == list[float]:
            return cc.StdvecType.get(mlirTypeFromPyType(float, ctx), ctx)

    emitFatalError(
        f"Cannot handle conversion of python type {argType} to MLIR type.")


def mlirTypeToPyType(argType):

    if IntegerType.isinstance(argType):
        width = IntegerType(argType).width
        if width == 1:
            return bool
        if width == 8:
            return np.int8
        if width == 16:
            return np.int16
        if width == 32:
            return np.int32
        if width == 64:
            return int

    if F64Type.isinstance(argType):
        return float

    if F32Type.isinstance(argType):
        return np.float32

    if quake.VeqType.isinstance(argType):
        return qvector

    if quake.RefType.isinstance(argType):
        return qubit

    if cc.CallableType.isinstance(argType):
        return Callable

    if ComplexType.isinstance(argType):
        if F64Type.isinstance(ComplexType(argType).element_type):
            return complex
        return np.complex64

    if cc.CharspanType.isinstance(argType):
        return pauli_word

    if cc.StdvecType.isinstance(argType):
        eleTy = cc.StdvecType.getElementType(argType)
        if cc.CharspanType.isinstance(eleTy):
            return list[pauli_word]

        pyEleTy = mlirTypeToPyType(eleTy)
        return list[pyEleTy]

    if cc.PointerType.isinstance(argType):
        valueTy = cc.PointerType.getElementType(argType)
        if cc.StateType.isinstance(valueTy):
            return State

    if cc.StructType.isinstance(argType):
        if (cc.StructType.getName(argType) == "tuple"):
            elements = [
                mlirTypeToPyType(v) for v in cc.StructType.getTypes(argType)
            ]
            return types.GenericAlias(tuple, tuple(elements))

        clsName = cc.StructType.getName(argType)
        if globalRegisteredTypes.isRegisteredClass(clsName):
            pyType, _ = globalRegisteredTypes.getClassAttributes(clsName)
            return pyType

    if quake.StruqType.isinstance(argType):
        if (quake.StruqType.getName(argType) == "tuple"):
            elements = [
                mlirTypeToPyType(v) for v in quake.StruqType.getTypes(argType)
            ]
            return types.GenericAlias(tuple, tuple(elements))

        clsName = quake.StruqType.getName(argType)
        if globalRegisteredTypes.isRegisteredClass(clsName):
            pyType, _ = globalRegisteredTypes.getClassAttributes(clsName)
            return pyType

    emitFatalError(
        f"Cannot infer python type from provided CUDA-Q type ({argType})")


def emitErrorIfInvalidPauli(pauliArg):
    """
    Verify that the input string is a valid Pauli string. 
    Throw an exception if not.
    """
    if any(c not in 'XYZI' for c in pauliArg):
        emitFatalError(
            f"Invalid pauli_word string provided as runtime argument ("
            f"{pauliArg}) - can only contain X, Y, Z, or I.")
