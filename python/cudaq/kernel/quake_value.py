# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from functools import partialmethod
import inspect
import sys
import random
import string
import numpy as np
import ctypes

from ..mlir.ir import *
from ..mlir.passmanager import *
from ..mlir.dialects import quake, cc
from ..mlir.dialects import builtin, func, arith
from ..mlir._mlir_libs._quakeDialects import cudaq_runtime
from .utils import mlirTypeFromPyType

qvector = cudaq_runtime.qvector


class QuakeValue(object):

    def __init__(self, mlirValue, pyKernel, size=None):
        self.mlirValue = mlirValue
        self.pyKernel = pyKernel
        self.ctx = self.pyKernel.ctx
        self.floatType = mlirTypeFromPyType(float, self.ctx)
        self.intType = mlirTypeFromPyType(int, self.ctx)
        self.knownUniqueExtractions = set()

    def __str__(self):
        return str(self.mlirValue)

    def size(self):
        with self.ctx, Location.unknown(), self.pyKernel.insertPoint:
            # assert this is a `stdvec` type or a `veq` type
            # See if we know the size of the `veq`
            # return `stdvecsizeop` or `veqsizeop`
            type = self.mlirValue.type
            if not quake.VeqType.isinstance(
                    type) and not cc.StdvecType.isinstance(type):
                raise RuntimeError(
                    "QuakeValue.size only valid for veq and stdvec types.")

            if quake.VeqType.isinstance(type):
                size = quake.VeqType.getSize(type)
                if size:
                    return size
                return QuakeValue(
                    quake.VeqSizeOp(self.intType, self.mlirValue).result,
                    self.pyKernel)

            # Must be a `stdvec` type
            return QuakeValue(
                cc.StdvecSizeOp(self.intType, self.mlirValue).result,
                self.pyKernel)

    def __intToFloat(self, intVal):
        return arith.SIToFPOp(self.floatType, intVal).result

    def __floatToVal(self, concreteFloat):
        return arith.ConstantOp(self.floatType,
                                FloatAttr.get(self.floatType,
                                              concreteFloat)).result

    def __intToVal(self, concreteInt):
        return arith.ConstantOp(self.intType,
                                IntegerAttr.get(self.intType,
                                                concreteInt)).result

    def __checkTypesAndCreateQuakeValue(self, other, opStr):
        thisVal = self.mlirValue
        otherVal = None
        mulOpStr = None
        if isinstance(other, float):
            otherVal = self.__floatToVal(other)
            mulOpStr = '{}FOp'.format(opStr)

            # Could be that this value is an integer, in which case
            # we have to cast to a float
            if IntegerType.isinstance(thisVal.type):
                thisVal = self.__intToFloat(thisVal)

        elif isinstance(other, int):
            otherVal = self.__intToVal(other)
            mulOpStr = '{}IOp'.format(opStr)
            # Could be that this value is a float, in which
            # case we should cast the other to an int
            if F64Type.isinstance(thisVal.type):
                otherVal = self.__intToFloat(otherVal)
                mulOpStr = '{}FOp'.format(opStr)
        else:
            # Here we know that the other value is a QuakeValue
            otherVal = other.mlirValue
            mulOpStr = '{}FOp'.format(opStr) if F64Type.isinstance(
                thisVal.type) else '{}IOp'.format(opStr)
            if mulOpStr == '{}FOp'.format(opStr) and IntegerType.isinstance(
                    otherVal.type):
                otherVal = arith.SIToFPOp(self.floatType, otherVal).result

        return thisVal, otherVal, mulOpStr

    def slice(self, startIdx, count):
        raise RuntimeError("QuakeValue.slice not implemented")

    def __neg__(self):
        with self.ctx, Location.unknown(), self.pyKernel.insertPoint:
            thisVal = self.mlirValue
            if IntegerType.isinstance(thisVal.type):
                thisVal = self.__intToFloat(thisVal)
            return QuakeValue(arith.NegFOp(thisVal).result, self.pyKernel)

    def __mul__(self, other):
        with self.ctx, Location.unknown(), self.pyKernel.insertPoint:
            thisVal, otherVal, opStr = self.__checkTypesAndCreateQuakeValue(
                other, 'Mul')
            return QuakeValue(
                getattr(arith, opStr)(thisVal, otherVal).result, self.pyKernel)

    def __rmul__(self, other):
        with self.ctx, Location.unknown(), self.pyKernel.insertPoint:
            thisVal, otherVal, opStr = self.__checkTypesAndCreateQuakeValue(
                other, 'Mul')
            return QuakeValue(
                getattr(arith, opStr)(otherVal, thisVal).result, self.pyKernel)

    def __truediv__(self, other):
        with self.ctx, Location.unknown(), self.pyKernel.insertPoint:
            thisVal, otherVal, opStr = self.__checkTypesAndCreateQuakeValue(
                other, 'Div')
            if opStr == 'DivIOp':
                opStr = 'DivSIOp'

            return QuakeValue(
                getattr(arith, opStr)(thisVal, otherVal).result, self.pyKernel)

    def __rtruediv__(self, other):
        with self.ctx, Location.unknown(), self.pyKernel.insertPoint:
            thisVal, otherVal, opStr = self.__checkTypesAndCreateQuakeValue(
                other, 'Div')
            if opStr == 'DivIOp':
                opStr = 'DivSIOp'

            return QuakeValue(
                getattr(arith, opStr)(otherVal, thisVal).result, self.pyKernel)

    def __add__(self, other):
        with self.ctx, Location.unknown(), self.pyKernel.insertPoint:
            thisVal, otherVal, opStr = self.__checkTypesAndCreateQuakeValue(
                other, 'Add')
            return QuakeValue(
                getattr(arith, opStr)(thisVal, otherVal).result, self.pyKernel)

    def __radd__(self, other):
        with self.ctx, Location.unknown(), self.pyKernel.insertPoint:
            thisVal, otherVal, opStr = self.__checkTypesAndCreateQuakeValue(
                other, 'Add')
            return QuakeValue(
                getattr(arith, opStr)(otherVal, thisVal).result, self.pyKernel)

    def __sub__(self, other):
        with self.ctx, Location.unknown(), self.pyKernel.insertPoint:
            thisVal, otherVal, opStr = self.__checkTypesAndCreateQuakeValue(
                other, 'Sub')
            return QuakeValue(
                getattr(arith, opStr)(thisVal, otherVal).result, self.pyKernel)

    def __rsub__(self, other):
        with self.ctx, Location.unknown(), self.pyKernel.insertPoint:
            thisVal, otherVal, opStr = self.__checkTypesAndCreateQuakeValue(
                other, 'Sub')
            return QuakeValue(
                getattr(arith, opStr)(otherVal, thisVal).result, self.pyKernel)

    def __getitem__(self, idx):
        with self.ctx, Location.unknown(), self.pyKernel.insertPoint:
            if cc.StdvecType.isinstance(self.mlirValue.type):
                eleTy = mlirTypeFromPyType(float, self.ctx)
                arrPtrTy = cc.PointerType.get(self.ctx,
                                              cc.ArrayType.get(self.ctx, eleTy))
                vecPtr = cc.StdvecDataOp(arrPtrTy, self.mlirValue).result
                elePtrTy = cc.PointerType.get(self.ctx, eleTy)
                eleAddr = None
                i64Ty = IntegerType.get_signless(64)
                if isinstance(idx, QuakeValue):
                    eleAddr = cc.ComputePtrOp(
                        elePtrTy, vecPtr, [idx.mlirValue],
                        DenseI32ArrayAttr.get([-2147483648], context=self.ctx))
                elif isinstance(idx, int):
                    self.knownUniqueExtractions.add(idx)
                    eleAddr = cc.ComputePtrOp(
                        elePtrTy, vecPtr, [],
                        DenseI32ArrayAttr.get([idx], context=self.ctx))
                loaded = cc.LoadOp(eleAddr.result)
                return QuakeValue(loaded.result, self.pyKernel)

            if quake.VeqType.isinstance(self.mlirValue.type):
                processedIdx = None
                if isinstance(idx, QuakeValue):
                    processedIdx = idx.mlirValue
                elif isinstance(idx, int):
                    i64Ty = IntegerType.get_signless(64)
                    processedIdx = arith.ConstantOp(i64Ty,
                                                    IntegerAttr.get(i64Ty,
                                                                    idx)).result
                else:
                    raise RuntimeError("invalid idx passed to QuakeValue.")
                op = quake.ExtractRefOp(quake.RefType.get(self.ctx),
                                        self.mlirValue,
                                        -1,
                                        index=processedIdx)
                return QuakeValue(op.result, self.pyKernel)

        raise RuntimeError("invalid getitem: ", idx)
