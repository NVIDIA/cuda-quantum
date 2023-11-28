# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from mlir_cudaq._mlir_libs._quakeDialects import cudaq_runtime
from mlir_cudaq.dialects import quake, cc
from mlir_cudaq.ir import *
from mlir_cudaq.passmanager import *
import numpy as np
from typing import Callable
import ast

qvector = cudaq_runtime.qvector
qubit = cudaq_runtime.qubit
qreg = qvector

nvqppPrefix = '__nvqpp__mlirgen__'

# Keep a global registry of all kernel FuncOps
# keyed on their name (without `__nvqpp__mlirgen__` prefix)
globalKernelRegistry = {}

# Keep a global registry of all kernel Python AST modules
# keyed on their name (without `__nvqpp__mlirgen__` prefix)
globalAstRegistry = {}


def mlirTypeFromAnnotation(annotation, ctx):
    """
        Return the MLIR Type corresponding to the given kernel function argument type annotation.
        Throws an exception if the programmer did not annotate function argument types. 
        """
    if annotation == None:
        raise RuntimeError(
            'cudaq.kernel functions must have argument type annotations.')

    if hasattr(annotation, 'attr'):
        if annotation.value.id == 'cudaq':
            if annotation.attr in ['qlist', 'qview', 'qvector']:
                return quake.VeqType.get(ctx)
            if annotation.attr == 'qubit':
                return quake.RefType.get(ctx)

        if annotation.value.id in ['numpy', 'np']:
            if annotation.attr == 'ndarray':
                return cc.StdvecType.get(ctx, F64Type.get())

    if isinstance(annotation,
                  ast.Subscript) and annotation.value.id == 'Callable':
        if not hasattr(annotation, 'slice'):
            raise RuntimeError('Callable type must have signature specified.')

        argTypes = [
            mlirTypeFromAnnotation(a, ctx)
            for a in annotation.slice.elts[0].elts
        ]
        return cc.CallableType.get(ctx, argTypes)

    if isinstance(annotation, ast.Subscript) and annotation.value.id == 'list':
        if not hasattr(annotation, 'slice'):
            raise RuntimeError('list subscript missing slice node.')

        # expected that slice is a Name node
        listEleTy = mlirTypeFromAnnotation(annotation.slice, ctx)
        return cc.StdvecType.get(ctx, listEleTy)

    if annotation.id == 'int':
        return IntegerType.get_signless(64)
    elif annotation.id == 'float':
        return F64Type.get()
    elif annotation.id == 'list':
        return cc.StdvecType.get(ctx, F64Type.get())
    elif annotation.id == 'bool':
        return IntegerType.get_signless(1)
    elif annotation.id == 'complex':
        return ComplexType.get(F64Type.get())
    else:
        raise RuntimeError('{} is not a supported type yet.'.format(
            annotation.id))


def mlirTypeFromPyType(argType, ctx,
                       **kwargs):

    if argType == int:
        return IntegerType.get_signless(64, ctx)
    if argType in [float, np.float64]:
        return F64Type.get(ctx)
    if argType == bool:
        return IntegerType.get_signless(1, ctx)
    if argType == complex:
        return ComplexType.get(mlirTypeFromPyType(float, ctx))

    if argType in [list, np.ndarray]:
        if 'argInstance' not in kwargs:
            return cc.StdvecType.get(ctx, mlirTypeFromPyType(float, ctx))

        argInstance = kwargs['argInstance']
        argTypeToCompareTo = kwargs['argTypeToCompareTo']

        if isinstance(argInstance[0], int):
            return cc.StdvecType.get(ctx, mlirTypeFromPyType(int, ctx))
        if isinstance(argInstance[0], float):
            # check if we are comparing to a complex...
            eleTy = cc.StdvecType.getElementType(argTypeToCompareTo)
            if ComplexType.isinstance(eleTy):
                raise RuntimeError(
                    "invalid runtime argument to kernel. list[complex] required, but list[float] provided."
                )
            return cc.StdvecType.get(ctx, mlirTypeFromPyType(float, ctx))
        if isinstance(argInstance[0], complex):
            return cc.StdvecType.get(ctx, mlirTypeFromPyType(complex, ctx))

    if argType == qvector or argType == qreg:
        return quake.VeqType.get(ctx)
    if argType == qubit:
        return quake.RefType.get(ctx)

    if 'argInstance' in kwargs:
        argInstance = kwargs['argInstance']
        if isinstance(argInstance, Callable):
            return cc.CallableType.get(ctx, argInstance.argTypes)

    raise RuntimeError(
        "can not handle conversion of python type {} to mlir type.".format(
            argType))


def mlirTypeToPyType(argType):

    if IntegerType.isinstance(argType):
        if IntegerType(argType).width == 1:
            return bool
        return int

    if F64Type.isinstance(argType):
        return float

    if ComplexType.isinstance(argType):
        return complex

    if cc.StdvecType.isinstance(argType):
        eleTy = cc.StdvecType.getElementType(argType)
        if IntegerType.isinstance(argType):
            if IntegerType(argType).width == 1:
                return list[bool]
            return list[int]
        if F64Type.isinstance(argType):
            return list[float]
        if ComplexType.isinstance(argType):
            return list[complex]

    raise RuntimeError("unhandled mlir-to-pytype {}".format(argType))
