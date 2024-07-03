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
    """
    A :class:`QuakeValue` represents a handle to an individual function 
    argument of a :class:`Kernel`, or a return value from an operation within
    it. As documented in :func:`make_kernel`, a :class:`QuakeValue` can hold
    values of the following types: int, float, list/List, :class:`qubit`, or 
    :class:`qvector`. The :class:`QuakeValue` can also hold kernel operations 
    such as qubit allocations and measurements.
    """

    def __init__(self, mlirValue, pyKernel, size=None):
        self.mlirValue = mlirValue
        self.pyKernel = pyKernel
        self.ctx = self.pyKernel.ctx
        self.floatType = mlirTypeFromPyType(float, self.ctx)
        self.intType = mlirTypeFromPyType(int, self.ctx)
        self.knownUniqueExtractions = set()

    def __str__(self):
        """
        Return a string representation of the value of `self` (:class:`QuakeValue`).
        """
        return str(self.mlirValue)

    def size(self):
        """
        Return the size of `self` (:class:`QuakeValue`), if it is of the type `stdvec` or `veq`.

        Raises:
	        RuntimeError: if the underlying :class:`QuakeValue` type is not `stdvec` or `veq`.

        """
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
        """
        Return a slice of the given :class:`QuakeValue` as a new :class:`QuakeValue`.

        Note:
            The underlying :class:`QuakeValue` must be a `list` or `veq`.

        Args:
            start (int): The index to begin the slice from.
            count (int): The number of elements to extract after the `start` index.
        Returns:
            :class:`QuakeValue`: A new `QuakeValue` containing a slice of `self`
            from the `start` element to the `start + count` element.
        """
        raise RuntimeError("QuakeValue.slice not implemented")

    def __neg__(self):
        """
        Return the negation of `self` (:class:`QuakeValue`).

        Raises:
            RuntimeError: if the underlying :class:`QuakeValue` type is not a float.

        .. code-block:: python

            # Example:
            kernel, value = cudaq.make_kernel(float)
            new_value: QuakeValue = -value
        """
        with self.ctx, Location.unknown(), self.pyKernel.insertPoint:
            thisVal = self.mlirValue
            if IntegerType.isinstance(thisVal.type):
                thisVal = self.__intToFloat(thisVal)
            return QuakeValue(arith.NegFOp(thisVal).result, self.pyKernel)

    def __mul__(self, other):
        """
        Return the product of `self` (:class:`QuakeValue`) with `other` (float).
        
        Raises:
	        RuntimeError: if the underlying :class:`QuakeValue` type is not a float.

        .. code-block:: python

            # Example:
            kernel, value = cudaq.make_kernel(float)
            new_value: QuakeValue = value * 5.0
        """
        with self.ctx, Location.unknown(), self.pyKernel.insertPoint:
            thisVal, otherVal, opStr = self.__checkTypesAndCreateQuakeValue(
                other, 'Mul')
            return QuakeValue(
                getattr(arith, opStr)(thisVal, otherVal).result, self.pyKernel)

    def __rmul__(self, other):
        """
        Return the product of `other` (float) with `self` (:class:`QuakeValue`).

        Raises:
	        RuntimeError: if the underlying :class:`QuakeValue` type is not a float.
        
        .. code-block:: python

            # Example:
            kernel, value = cudaq.make_kernel(float)
            new_value: QuakeValue = 5.0 * value
        """
        with self.ctx, Location.unknown(), self.pyKernel.insertPoint:
            thisVal, otherVal, opStr = self.__checkTypesAndCreateQuakeValue(
                other, 'Mul')
            return QuakeValue(
                getattr(arith, opStr)(otherVal, thisVal).result, self.pyKernel)

    def __truediv__(self, other):
        """
        Return the division of `self` (:class:`QuakeValue`) with `other` (float).

        Raises:
	        RuntimeError: if the underlying :class:`QuakeValue` type is not a float.

        .. code-block:: python

            # Example:
            kernel, value = cudaq.make_kernel(float)
            new_value: QuakeValue = value / 5.0
        """
        with self.ctx, Location.unknown(), self.pyKernel.insertPoint:
            thisVal, otherVal, opStr = self.__checkTypesAndCreateQuakeValue(
                other, 'Div')
            if opStr == 'DivIOp':
                opStr = 'DivSIOp'

            return QuakeValue(
                getattr(arith, opStr)(thisVal, otherVal).result, self.pyKernel)

    def __rtruediv__(self, other):
        """
        Return the division of `other` (float) with `self` (:class:`QuakeValue`).

        Raises:
	        RuntimeError: if the underlying :class:`QuakeValue` type is not a float.
        
        .. code-block:: python

            # Example:
            kernel, value = cudaq.make_kernel(float)
            new_value: QuakeValue = 5.0 / value
        """
        with self.ctx, Location.unknown(), self.pyKernel.insertPoint:
            thisVal, otherVal, opStr = self.__checkTypesAndCreateQuakeValue(
                other, 'Div')
            if opStr == 'DivIOp':
                opStr = 'DivSIOp'

            return QuakeValue(
                getattr(arith, opStr)(otherVal, thisVal).result, self.pyKernel)

    def __add__(self, other):
        """
        Return the sum of `self` (:class:`QuakeValue`) and `other` (float).

        Raises:
	        RuntimeError: if the underlying :class:`QuakeValue` type is not a float.
        
        .. code-block:: python

            # Example:
            kernel, value = cudaq.make_kernel(float)
            new_value: QuakeValue = value + 5.0
        """
        with self.ctx, Location.unknown(), self.pyKernel.insertPoint:
            thisVal, otherVal, opStr = self.__checkTypesAndCreateQuakeValue(
                other, 'Add')
            return QuakeValue(
                getattr(arith, opStr)(thisVal, otherVal).result, self.pyKernel)

    def __radd__(self, other):
        """
        Return the sum of `other` (float) and `self` (:class:`QuakeValue`).

        Raises:
	        RuntimeError: if the underlying :class:`QuakeValue` type is not a float.
        
        .. code-block:: python

            # Example:
            kernel, value = cudaq.make_kernel(float)
            new_value: QuakeValue = 5.0 + value
        """
        with self.ctx, Location.unknown(), self.pyKernel.insertPoint:
            thisVal, otherVal, opStr = self.__checkTypesAndCreateQuakeValue(
                other, 'Add')
            return QuakeValue(
                getattr(arith, opStr)(otherVal, thisVal).result, self.pyKernel)

    def __sub__(self, other):
        """
        Return the difference of `self` (:class:`QuakeValue`) and `other` (float).

        Raises:
	        RuntimeError: if the underlying :class:`QuakeValue` type is not a float.
        
        .. code-block:: python

            # Example:
            kernel, value = cudaq.make_kernel(float)
            new_value: QuakeValue = value - 5.0
        """
        with self.ctx, Location.unknown(), self.pyKernel.insertPoint:
            thisVal, otherVal, opStr = self.__checkTypesAndCreateQuakeValue(
                other, 'Sub')
            return QuakeValue(
                getattr(arith, opStr)(thisVal, otherVal).result, self.pyKernel)

    def __rsub__(self, other):
        """
        Return the difference of `other` (float) and `self` (:class:`QuakeValue`).

        Raises:
	        RuntimeError: if the underlying :class:`QuakeValue` type is not a float.

        .. code-block:: python

            # Example:
            kernel, value = cudaq.make_kernel(float)
            new_value: QuakeValue = 5.0 - value
        """
        with self.ctx, Location.unknown(), self.pyKernel.insertPoint:
            thisVal, otherVal, opStr = self.__checkTypesAndCreateQuakeValue(
                other, 'Sub')
            return QuakeValue(
                getattr(arith, opStr)(otherVal, thisVal).result, self.pyKernel)

    def __getitem__(self, idx):
        """
        Return the element of `self` at the provided `index`.

        Note:
	        Only `list` or :class:`qvector` type :class:`QuakeValue`'s may be indexed.

        Args:
	        index (int): The element of `self` that you'd like to return.
        Returns:
	        :class:`QuakeValue`: 
	        A new :class:`QuakeValue` for the `index` element of `self`.
        Raises:
	        RuntimeError: if `self` is a non-subscriptable :class:`QuakeValue`.
        """
        with self.ctx, Location.unknown(), self.pyKernel.insertPoint:
            if cc.StdvecType.isinstance(self.mlirValue.type):
                eleTy = cc.StdvecType.getElementType(self.mlirValue.type)
                arrTy = cc.ArrayType.get(self.ctx, eleTy)
                arrPtrTy = cc.PointerType.get(self.ctx, arrTy)
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
