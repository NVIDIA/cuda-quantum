# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations
import ast
import re
import sys
import traceback
import numpy as np
from typing import get_origin, Callable, List
import types

from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime
from cudaq.mlir.dialects import quake, cc
from cudaq.mlir.ir import ComplexType, F32Type, F64Type, IntegerType

State = cudaq_runtime.State
qvector = cudaq_runtime.qvector
qview = cudaq_runtime.qview
qubit = cudaq_runtime.qubit
pauli_word = cudaq_runtime.pauli_word
qreg = qvector

nvqppPrefix = '__nvqpp__mlirgen__'

ahkPrefix = '__analog_hamiltonian_kernel__'

# Keep a global registry of all kernel FuncOps
# keyed on their name (without `__nvqpp__mlirgen__` prefix)
globalKernelRegistry = {}

# Keep a global registry of all kernel Python AST modules
# keyed on their name (without `__nvqpp__mlirgen__` prefix).
# The values in this dictionary are a tuple of the AST module
# and the source code location for the kernel.
globalAstRegistry = {}

# Keep a global registry of all registered custom operations.
globalRegisteredOperations = {}

# Keep a global registry of any custom data types
globalRegisteredTypes = cudaq_runtime.DataClassRegistry


class Color:
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


def emitFatalError(msg):
    """
    Emit a fatal error diagnostic. The goal here is to 
    retain the input message, but also provide the offending 
    source location by inspecting the stack trace. 
    """
    print(Color.BOLD, end='')
    try:
        # Raise the exception so we can get the
        # stack trace to inspect
        raise RuntimeError(msg)
    except RuntimeError:
        # Immediately grab the exception and
        # analyze the stack trace, get the source location
        # and construct a new error diagnostic
        cached = sys.tracebacklimit
        sys.tracebacklimit = None
        offendingSrc = traceback.format_stack()
        sys.tracebacklimit = cached
        if len(offendingSrc):
            msg = Color.RED + "error: " + Color.END + Color.BOLD + msg + Color.END + '\n\nOffending code:\n' + offendingSrc[
                0]
    raise RuntimeError(msg)


def emitWarning(msg):
    """
    Emit a warning, providing the user with source file information and
    the offending code.
    """
    print(Color.BOLD, end='')
    try:
        # Raise the exception so we can get the
        # stack trace to inspect
        raise RuntimeError(msg)
    except RuntimeError:
        # Immediately grab the exception and
        # analyze the stack trace, get the source location
        # and construct a new error diagnostic
        cached = sys.tracebacklimit
        sys.tracebacklimit = None
        offendingSrc = traceback.format_stack()
        sys.tracebacklimit = cached
        if len(offendingSrc):
            msg = Color.YELLOW + "error: " + Color.END + Color.BOLD + msg + Color.END + '\n\nOffending code:\n' + offendingSrc[
                0]


def mlirTypeFromAnnotation(annotation, ctx, raiseError=False):
    """
    Return the MLIR Type corresponding to the given kernel function argument type annotation.
    Throws an exception if the programmer did not annotate function argument types. 
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

    if hasattr(annotation, 'attr') and hasattr(annotation.value, 'id'):
        if annotation.value.id == 'cudaq':
            if annotation.attr in ['qview', 'qvector']:
                return quake.VeqType.get(ctx)
            if annotation.attr in ['State']:
                return cc.PointerType.get(ctx, cc.StateType.get(ctx))
            if annotation.attr == 'qubit':
                return quake.RefType.get(ctx)
            if annotation.attr == 'pauli_word':
                return cc.CharspanType.get(ctx)

        if annotation.value.id in ['numpy', 'np']:
            if annotation.attr in ['array', 'ndarray']:
                return cc.StdvecType.get(ctx, F64Type.get())
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
                f"Callable type must have signature specified ({ast.unparse(annotation) if hasattr(ast, 'unparse') else annotation})."
            )

        if hasattr(annotation.slice, 'elts'):
            firstElement = annotation.slice.elts[0]
        elif hasattr(annotation.slice, 'value') and hasattr(
                annotation.slice.value, 'elts'):
            firstElement = annotation.slice.value.elts[0]
        else:
            localEmitFatalError(
                f"Unable to get list elements when inferring type from annotation ({ast.unparse(annotation) if hasattr(ast, 'unparse') else annotation})."
            )
        argTypes = [mlirTypeFromAnnotation(a, ctx) for a in firstElement.elts]
        return cc.CallableType.get(ctx, argTypes)

    if isinstance(annotation,
                  ast.Subscript) and (annotation.value.id == 'list' or
                                      annotation.value.id == 'List'):
        if not hasattr(annotation, 'slice'):
            localEmitFatalError(
                f"list subscript missing slice node ({ast.unparse(annotation) if hasattr(ast, 'unparse') else annotation})."
            )

        eleTypeNode = annotation.slice
        # expected that slice is a Name node
        listEleTy = mlirTypeFromAnnotation(eleTypeNode, ctx)
        return cc.StdvecType.get(ctx, listEleTy)

    if isinstance(annotation,
                  ast.Subscript) and (annotation.value.id == 'tuple' or
                                      annotation.value.id == 'Tuple'):
        localEmitFatalError(
            f"Use of tuples is not supported in kernels ({ast.unparse(annotation) if hasattr(ast, 'unparse') else annotation})."
        )

        #FIXME: re-enable tuple support after we have the spec.
        # https://github.com/NVIDIA/cuda-quantum/issues/3031
        if not hasattr(annotation, 'slice'):
            localEmitFatalError(
                f"tuple subscript missing slice node ({ast.unparse(annotation) if hasattr(ast, 'unparse') else annotation})."
            )

        # slice is an `ast.Tuple` of type annotations
        elements = None
        if hasattr(annotation.slice, 'elts'):
            elements = annotation.slice.elts
        else:
            localEmitFatalError(
                f"Unable to get tuple elements when inferring type from annotation ({ast.unparse(annotation) if hasattr(ast, 'unparse') else annotation})."
            )

        eleTypes = [mlirTypeFromAnnotation(v, ctx) for v in elements]
        return cc.StructType.getNamed(ctx, "tuple", eleTypes)

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
                f"{ast.unparse(annotation) if hasattr(ast, 'unparse') else annotation} is not yet a supported type (could not infer type name)."
            )
    else:
        localEmitFatalError(
            f"{ast.unparse(annotation) if hasattr(ast, 'unparse') else annotation} is not a supported type yet (could not infer type name)."
        )

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
        structTys = [mlirTypeFromPyType(v, ctx) for _, v in memberTys.items()]
        for ty in structTys:
            if cc.StructType.isinstance(ty):
                localEmitFatalError(
                    'recursive struct types are not allowed in kernels.')

        if len({
                k: v
                for k, v in pyType.__dict__.items()
                if not (k.startswith('__') and k.endswith('__'))
        }) != 0:
            localEmitFatalError(
                'struct types with user specified methods are not allowed.')

        numQuantumMemberTys = sum([
            1 if
            (quake.RefType.isinstance(ty) or quake.VeqType.isinstance(ty) or
             quake.StruqType.isinstance(ty)) else 0 for ty in structTys
        ])
        numStruqMemberTys = sum(
            [1 if (quake.StruqType.isinstance(ty)) else 0 for ty in structTys])
        if numQuantumMemberTys != 0:  # we have quantum member types
            if numQuantumMemberTys != len(structTys):
                emitFatalError(
                    f'hybrid quantum-classical data types not allowed in kernel code.'
                )
            if numStruqMemberTys != 0:
                emitFatalError(f'recursive quantum struct types not allowed.')
            return quake.StruqType.getNamed(ctx, id, structTys)

        return cc.StructType.getNamed(ctx, id, structTys)

    localEmitFatalError(
        f"{ast.unparse(annotation) if hasattr(ast, 'unparse') else annotation} is not a supported type."
    )


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
        return cc.PointerType.get(ctx, cc.StateType.get(ctx))

    if get_origin(argType) == list:
        result = re.search(r'ist\[(.*)\]', str(argType))
        eleTyName = result.group(1)
        argType = list
        inst = pyInstanceFromName(eleTyName)
        if (inst != None):
            kwargs['argInstance'] = [inst]

    if argType in [list, np.ndarray, List]:
        if 'argInstance' not in kwargs:
            return cc.StdvecType.get(ctx, mlirTypeFromPyType(float, ctx))
        if argType != np.ndarray:
            if kwargs['argInstance'] == None:
                return cc.StdvecType.get(ctx, mlirTypeFromPyType(float, ctx))

        argInstance = kwargs['argInstance']
        argTypeToCompareTo = kwargs[
            'argTypeToCompareTo'] if 'argTypeToCompareTo' in kwargs else None

        if len(argInstance) == 0:
            if argTypeToCompareTo == None:
                emitFatalError('Cannot infer runtime argument type')

            eleTy = cc.StdvecType.getElementType(argTypeToCompareTo)
            return cc.StdvecType.get(ctx, eleTy)

        if isinstance(argInstance[0], list):
            return cc.StdvecType.get(
                ctx,
                mlirTypeFromPyType(
                    type(argInstance[0]),
                    ctx,
                    argInstance=argInstance[0],
                    argTypeToCompareTo=cc.StdvecType.getElementType(
                        argTypeToCompareTo)))

        return cc.StdvecType.get(ctx,
                                 mlirTypeFromPyType(type(argInstance[0]), ctx))

    if get_origin(argType) == tuple:
        #FIXME: re-enable tuple support after we have the spec.
        # https://github.com/NVIDIA/cuda-quantum/issues/3031
        emitFatalError(f'Use of tuples is not supported in kernels ({argType})')
        result = re.search(r'uple\[(?P<names>.*)\]', str(argType))
        eleTyNames = result.group('names')
        eleTypes = []
        while eleTyNames != None:
            result = re.search(r'(?P<names>.*),\s*(?P<name>.*)', eleTyNames)
            eleTyName = result.group('name') if result != None else eleTyNames
            eleTyNames = result.group('names') if result != None else None
            pyInstance = pyInstanceFromName(eleTyName)
            if pyInstance == None:
                emitFatalError(f'Invalid tuple element type ({eleTyName})')
            eleTypes.append(mlirTypeFromPyType(type(pyInstance), ctx))
        eleTypes.reverse()
        return cc.StructType.getNamed(ctx, "tuple", eleTypes)

    if (argType == tuple):
        #FIXME: re-enable tuple support after we have the spec.
        # https://github.com/NVIDIA/cuda-quantum/issues/3031
        emitFatalError(f'Use of tuples is not supported in kernels ({argType})')
        argInstance = kwargs['argInstance']
        if argInstance == None or (len(argInstance) == 0):
            emitFatalError(f'Cannot infer runtime argument type for {argType}')
        eleTypes = [mlirTypeFromPyType(type(ele), ctx) for ele in argInstance]
        return cc.StructType.getNamed(ctx, "tuple", eleTypes)

    if argType == qvector or argType == qreg or argType == qview:
        return quake.VeqType.get(ctx)
    if argType == qubit:
        return quake.RefType.get(ctx)
    if argType == pauli_word:
        return cc.CharspanType.get(ctx)

    if 'argInstance' in kwargs:
        argInstance = kwargs['argInstance']
        if isinstance(argInstance, Callable):
            return cc.CallableType.get(ctx, argInstance.argTypes)

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
                return quake.StruqType.getNamed(ctx, name, structTys)

            return cc.StructType.getNamed(ctx, name, structTys)

    if 'argInstance' not in kwargs:
        if argType == list[int]:
            return cc.StdvecType.get(ctx, mlirTypeFromPyType(int, ctx))
        if argType == list[float]:
            return cc.StdvecType.get(ctx, mlirTypeFromPyType(float, ctx))

    emitFatalError(
        f"Can not handle conversion of python type {argType} to MLIR type.")


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

    emitFatalError(
        f"Cannot infer python type from provided CUDA-Q type ({argType})")


def emitErrorIfInvalidPauli(pauliArg):
    """
    Verify that the input string is a valid Pauli string. 
    Throw an exception if not.
    """
    if any(c not in 'XYZI' for c in pauliArg):
        emitFatalError(
            f"Invalid pauli_word string provided as runtime argument ({pauliArg}) - can only contain X, Y, Z, or I."
        )
