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

qvector = cudaq_runtime.qvector
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


class Color:
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
    except RuntimeError as e:
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

    if hasattr(annotation, 'attr'):
        if annotation.value.id == 'cudaq':
            if annotation.attr in ['qview', 'qvector']:
                return quake.VeqType.get(ctx)
            if annotation.attr == 'qubit':
                return quake.RefType.get(ctx)
            if annotation.attr == 'pauli_word':
                return cc.CharspanType.get(ctx)

        if annotation.value.id in ['numpy', 'np']:
            if annotation.attr == 'ndarray':
                return cc.StdvecType.get(ctx, F64Type.get())

    if isinstance(annotation,
                  ast.Subscript) and annotation.value.id == 'Callable':
        if not hasattr(annotation, 'slice'):
            localEmitFatalError(
                f'Callable type must have signature specified ({ast.unparse(annotation)}).'
            )

        if hasattr(annotation.slice, 'elts'):
            firstElement = annotation.slice.elts[0]
        elif hasattr(annotation.slice, 'value') and hasattr(
                annotation.slice.value, 'elts'):
            firstElement = annotation.slice.value.elts[0]
        else:
            localEmitFatalError(
                f'Unable to get list elements when inferring type from annotation ({ast.unparse(annotation)}).'
            )
        argTypes = [mlirTypeFromAnnotation(a, ctx) for a in firstElement.elts]
        return cc.CallableType.get(ctx, argTypes)

    if isinstance(annotation,
                  ast.Subscript) and (annotation.value.id == 'list' or
                                      annotation.value.id == 'List'):
        if not hasattr(annotation, 'slice'):
            localEmitFatalError(
                f'list subscript missing slice node ({ast.unparse(annotation)}).'
            )

        # The tree differs here between Python 3.8 and 3.9+
        eleTypeNode = annotation.slice
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
            f'{ast.unparse(annotation)} is not a supported type yet (could not infer type name).'
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

    localEmitFatalError(f'{id} is not a supported type.')


def mlirTypeFromPyType(argType, ctx, **kwargs):

    if argType == int:
        return IntegerType.get_signless(64, ctx)
    if argType in [float, np.float64]:
        return F64Type.get(ctx)
    if argType == bool:
        return IntegerType.get_signless(1, ctx)
    if argType == complex:
        return ComplexType.get(mlirTypeFromPyType(float, ctx))
    if argType == pauli_word:
        return cc.CharspanType.get(ctx)

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
        if isinstance(argInstance[0], float):
            if argTypeToCompareTo != None:
                # check if we are comparing to a complex...
                eleTy = cc.StdvecType.getElementType(argTypeToCompareTo)
                if ComplexType.isinstance(eleTy):
                    emitFatalError(
                        "Invalid runtime argument to kernel. list[complex] required, but list[float] provided."
                    )
            return cc.StdvecType.get(ctx, mlirTypeFromPyType(float, ctx))

        if isinstance(argInstance[0], (complex, np.complex128)):
            return cc.StdvecType.get(ctx, mlirTypeFromPyType(complex, ctx))

        if isinstance(argInstance[0], np.complex64):
            return cc.StdvecType.get(ctx, ComplexType.get(F32Type.get(ctx)))

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

    if argType == qvector or argType == qreg:
        return quake.VeqType.get(ctx)
    if argType == qubit:
        return quake.RefType.get(ctx)
    if argType == pauli_word:
        return cc.CharspanType.get(ctx)

    if 'argInstance' in kwargs:
        argInstance = kwargs['argInstance']
        if isinstance(argInstance, Callable):
            return cc.CallableType.get(ctx, argInstance.argTypes)

    emitFatalError(
        f"Can not handle conversion of python type {argType} to MLIR type.")


def mlirTypeToPyType(argType):

    if IntegerType.isinstance(argType):
        if IntegerType(argType).width == 1:
            return bool
        return int

    if F64Type.isinstance(argType):
        return float

    if ComplexType.isinstance(argType):
        return complex

    if cc.CharspanType.isinstance(argType):
        return pauli_word

    def getListType(eleType: type):
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
        if ComplexType.isinstance(eleTy):
            return getListType(complex)

    emitFatalError(
        f"Cannot infer CUDA Quantum type from provided Python type ({argType})")


def emitErrorIfInvalidPauli(pauliArg):
    """
    Verify that the input string is a valid Pauli string. 
    Throw an exception if not.
    """
    if any(c not in 'XYZI' for c in pauliArg):
        emitFatalError(
            f"Invalid pauli_word string provided as runtime argument ({pauliArg}) - can only contain X, Y, Z, or I."
        )
