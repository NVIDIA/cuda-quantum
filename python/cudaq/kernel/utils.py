# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from ..mlir._mlir_libs._quakeDialects import cudaq_runtime
from ..mlir.dialects import quake, cc
from ..mlir.ir import *
from ..mlir.passmanager import *
import numpy as np
from typing import Callable, List
import ast, sys, traceback
import re
from typing import get_origin

State = cudaq_runtime.State
qvector = cudaq_runtime.qvector
qview = cudaq_runtime.qview
qubit = cudaq_runtime.qubit
pauli_word = cudaq_runtime.pauli_word
qreg = qvector

nvqppPrefix = '__nvqpp__mlirgen__'

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
globalRegisteredTypes = {}


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

        # The tree differs here between Python 3.8 and 3.9+
        eleTypeNode = annotation.slice
        ## [PYTHON_VERSION_FIX]
        if sys.version_info < (3, 9):
            eleTypeNode = eleTypeNode.value

        # expected that slice is a Name node
        listEleTy = mlirTypeFromAnnotation(eleTypeNode, ctx)
        return cc.StdvecType.get(ctx, listEleTy)

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
    if id in globalRegisteredTypes:
        _, memberTys = globalRegisteredTypes[id]
        structTys = [mlirTypeFromPyType(v, ctx) for _, v in memberTys.items()]
        return cc.StructType.getNamed(ctx, id, structTys)

    localEmitFatalError(
        f"{ast.unparse(annotation) if hasattr(ast, 'unparse') else annotation} is not a supported type."
    )


def mlirTypeFromPyType(argType, ctx, **kwargs):
    if argType == int:
        return IntegerType.get_signless(64, ctx)
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
        if eleTyName == 'int':
            kwargs['argInstance'] = [int(0)]
        elif eleTyName == 'float':
            kwargs['argInstance'] = [float(0.0)]
        elif eleTyName == 'bool':
            kwargs['argInstance'] = [bool(False)]
        elif eleTyName == 'complex':
            kwargs['argInstance'] = [0j]
        elif eleTyName == 'pauli_word':
            kwargs['argInstance'] = [pauli_word('')]
        elif eleTyName == 'numpy.complex128':
            kwargs['argInstance'] = [np.complex128(0.0)]
        elif eleTyName == 'numpy.complex64':
            kwargs['argInstance'] = [np.complex64(0.0)]

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

        if isinstance(argInstance[0], bool):
            return cc.StdvecType.get(ctx, mlirTypeFromPyType(bool, ctx))
        if isinstance(argInstance[0], int):
            return cc.StdvecType.get(ctx, mlirTypeFromPyType(int, ctx))
        if isinstance(argInstance[0], (float, np.float64)):
            return cc.StdvecType.get(ctx, mlirTypeFromPyType(float, ctx))
        if isinstance(argInstance[0], np.float32):
            return cc.StdvecType.get(ctx, mlirTypeFromPyType(np.float32, ctx))

        if isinstance(argInstance[0], (complex, np.complex128)):
            return cc.StdvecType.get(ctx, mlirTypeFromPyType(complex, ctx))

        if isinstance(argInstance[0], np.complex64):
            return cc.StdvecType.get(ctx, mlirTypeFromPyType(np.complex64, ctx))

        if isinstance(argInstance[0], pauli_word):
            return cc.StdvecType.get(ctx, cc.CharspanType.get(ctx))

        if isinstance(argInstance[0], list):
            return cc.StdvecType.get(
                ctx,
                mlirTypeFromPyType(
                    type(argInstance[0]),
                    ctx,
                    argInstance=argInstance[0],
                    argTypeToCompareTo=cc.StdvecType.getElementType(
                        argTypeToCompareTo)))

        emitFatalError(f'Invalid list element type ({argType})')

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
    else:
        if argType == list[int]:
            return cc.StdvecType.get(ctx, mlirTypeFromPyType(int, ctx))
        if argType == list[float]:
            return cc.StdvecType.get(ctx, mlirTypeFromPyType(float, ctx))

    for name, (customTys, memberTys) in globalRegisteredTypes.items():
        if argType == customTys:
            structTys = [
                mlirTypeFromPyType(v, ctx) for _, v in memberTys.items()
            ]
            return cc.StructType.getNamed(ctx, name, structTys)

    emitFatalError(
        f"Can not handle conversion of python type {argType} to MLIR type.")


def mlirTypeToPyType(argType):

    if IntegerType.isinstance(argType):
        if IntegerType(argType).width == 1:
            return bool
        return int

    if F64Type.isinstance(argType):
        return float

    if F32Type.isinstance(argType):
        return np.float32

    if ComplexType.isinstance(argType):
        if F64Type.isinstance(ComplexType(argType).element_type):
            return complex
        return np.complex64

    if cc.CharspanType.isinstance(argType):
        return pauli_word

    def getListType(eleType: type):
        ## [PYTHON_VERSION_FIX]
        if sys.version_info < (3, 9):
            return List[eleType]
        else:
            return list[eleType]

    if cc.StdvecType.isinstance(argType):
        eleTy = cc.StdvecType.getElementType(argType)
        if cc.CharspanType.isinstance(eleTy):
            return getListType(pauli_word)

        if IntegerType.isinstance(eleTy):
            if IntegerType(eleTy).width == 1:
                return getListType(bool)
            return getListType(int)
        if F64Type.isinstance(eleTy):
            return getListType(float)
        if F32Type.isinstance(eleTy):
            return getListType(np.float32)
        if ComplexType.isinstance(eleTy):
            ty = complex if F64Type.isinstance(
                ComplexType(eleTy).element_type) else np.complex64
            return getListType(ty)

    if cc.PointerType.isinstance(argType):
        valueTy = cc.PointerType.getElementType(argType)
        if cc.StateType.isinstance(valueTy):
            return State

    emitFatalError(
        f"Cannot infer CUDA-Q type from provided Python type ({argType})")


def emitErrorIfInvalidPauli(pauliArg):
    """
    Verify that the input string is a valid Pauli string. 
    Throw an exception if not.
    """
    if any(c not in 'XYZI' for c in pauliArg):
        emitFatalError(
            f"Invalid pauli_word string provided as runtime argument ({pauliArg}) - can only contain X, Y, Z, or I."
        )
