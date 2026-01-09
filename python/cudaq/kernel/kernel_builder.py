# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import random
import re
import string
import uuid
import weakref
from functools import partialmethod
from typing import get_origin

import numpy as np
from cudaq.mlir.ir import (BoolAttr, Block, Context, Module, TypeAttr, UnitAttr,
                           FunctionType, DictAttr, F32Type, F64Type, NoneType,
                           ArrayAttr, Location, FloatAttr, StringAttr,
                           IntegerAttr, IntegerType, ComplexType,
                           InsertionPoint, SymbolTable, DenseI32ArrayAttr,
                           FlatSymbolRefAttr)
from cudaq.mlir.passmanager import PassManager
from cudaq.mlir.dialects import (complex as complexDialect, arith, quake, cc,
                                 func, math)
from cudaq.mlir._mlir_libs._quakeDialects import (
    cudaq_runtime, gen_vector_of_complex_constant, load_intrinsic)
from .captured_data import CapturedDataStorage
from .common.fermionic_swap import fermionic_swap_builder
from .common.givens import givens_builder
from .kernel_decorator import isa_kernel_decorator
from .quake_value import QuakeValue
from .utils import (emitFatalError, emitWarning, nvqppPrefix, getMLIRContext,
                    recover_func_op, mlirTypeToPyType, cudaq__unique_attr_name,
                    mlirTypeFromPyType, emitErrorIfInvalidPauli,
                    recover_value_of, globalRegisteredOperations,
                    recover_calling_module)

kDynamicPtrIndex: int = -2147483648

qvector = cudaq_runtime.qvector

# This file reproduces the cudaq::kernel_builder in Python


def __generalOperation(self,
                       opName,
                       parameters,
                       controls,
                       target,
                       isAdj=False,
                       context=None):
    """
    This is a utility function that applies a general quantum operation to the
    internal PyKernel MLIR ModuleOp.
    """
    opCtor = getattr(quake, '{}Op'.format(opName.title()))
    if hasattr(self, 'qkeModule'):
        del self.qkeModule

    if quake.RefType.isinstance(target.mlirValue.type):
        opCtor([], parameters, controls, [target.mlirValue], is_adj=isAdj)
        return

    # Must be a `veq`, get the size
    size = quake.VeqSizeOp(self.getIntegerType(), target.mlirValue)

    def body(idx):
        extracted = quake.ExtractRefOp(quake.RefType.get(context),
                                       target.mlirValue,
                                       -1,
                                       index=idx).result
        opCtor([], parameters, controls, [extracted], is_adj=isAdj)

    self.createInvariantForLoop(size, body)


def get_parameter_value(self, parameter):
    paramVal = parameter
    if isinstance(parameter, float):
        fty = mlirTypeFromPyType(float, self.ctx)
        paramVal = arith.ConstantOp(fty, FloatAttr.get(fty, parameter))
    elif isinstance(parameter, QuakeValue):
        paramVal = parameter.mlirValue
    return paramVal


def __singleTargetOperation(self, opName, target, isAdj=False):
    """
    Utility function for adding a single target quantum operation to the MLIR
    representation for the PyKernel.
    """
    with self.insertPoint, self.loc:
        __generalOperation(self,
                           opName, [], [],
                           target,
                           isAdj=isAdj,
                           context=self.ctx)


def __singleTargetControlOperation(self, opName, control, target, isAdj=False):
    """
    Utility function for adding a single target controlled quantum operation to
    the MLIR representation for the PyKernel.
    """
    with self.insertPoint, self.loc:
        fwdControls = None
        if isinstance(control, list):
            fwdControls = [c.mlirValue for c in control]
        elif quake.RefType.isinstance(
                control.mlirValue.type) or quake.VeqType.isinstance(
                    control.mlirValue.type):
            fwdControls = [control.mlirValue]
        else:
            emitFatalError(f"invalid control type for {opName}.")

        __generalOperation(self,
                           opName, [],
                           fwdControls,
                           target,
                           isAdj=isAdj,
                           context=self.ctx)


def __singleTargetSingleParameterOperation(self,
                                           opName,
                                           parameter,
                                           target,
                                           isAdj=False):
    """
    Utility function for adding a single target, one parameter quantum operation
    to the MLIR representation for the PyKernel.
    """
    with self.insertPoint, self.loc:
        __generalOperation(self,
                           opName, [get_parameter_value(self, parameter)], [],
                           target,
                           isAdj=isAdj,
                           context=self.ctx)


def __singleTargetSingleParameterControlOperation(self,
                                                  opName,
                                                  parameter,
                                                  controls,
                                                  target,
                                                  isAdj=False):
    """
    Utility function for adding a single target, one parameter, controlled
    quantum operation to the MLIR representation for the PyKernel.
    """
    with self.insertPoint, self.loc:
        fwdControls = None
        if isinstance(controls, list):
            fwdControls = [c.mlirValue for c in controls]
        elif quake.RefType.isinstance(
                controls.mlirValue.type) or quake.VeqType.isinstance(
                    controls.mlirValue.type):
            fwdControls = [controls.mlirValue]
        else:
            emitFatalError(f"invalid controls type for {opName}.")

        __generalOperation(self,
                           opName, [get_parameter_value(self, parameter)],
                           fwdControls,
                           target,
                           isAdj=isAdj,
                           context=self.ctx)


def supportCommonCast(mlirType, otherTy, arg, FromType, ToType, PyType):
    argEleTy = cc.StdvecType.getElementType(mlirType)
    eleTy = cc.StdvecType.getElementType(otherTy)
    if ToType.isinstance(eleTy) and FromType.isinstance(argEleTy):
        return [PyType(i) for i in arg]
    return None


def __generalCustomOperation(self, opName, *args):
    """
    Utility function for adding a generic quantum operation to the MLIR
    representation for the PyKernel.

    A controlled version can be invoked by passing additional arguments to the
    operation. For an N-qubit operation, the last N arguments are treated as
    `targets` and excess arguments as `controls`.
    """

    global globalRegisteredOperations
    unitary = globalRegisteredOperations[opName]

    numTargets = int(np.log2(np.sqrt(unitary.size)))

    qubits = []
    if hasattr(self, 'qkeModule'):
        del self.qkeModule
    with self.insertPoint, self.loc:
        for arg in args:
            if isinstance(arg, QuakeValue):
                qubits.append(arg.mlirValue)
            else:
                emitFatalError(f"invalid argument type passed to {opName}.")

        targets = []
        controls = []

        if numTargets == len(qubits):
            targets = qubits
        elif numTargets < len(qubits):
            numControls = len(qubits) - numTargets
            targets = qubits[-numTargets:]
            controls = qubits[:numControls]
        else:
            emitFatalError(
                f"too few arguments passed to {opName}, expected ({numTargets})"
            )

        globalName = f'{nvqppPrefix}{opName}_generator_{numTargets}.rodata'
        currentST = SymbolTable(self.module.operation)
        if not globalName in currentST:
            with InsertionPoint(self.module.body):
                gen_vector_of_complex_constant(self.loc, self.module,
                                               globalName, unitary.tolist())

        quake.CustomUnitarySymbolOp([],
                                    generator=FlatSymbolRefAttr.get(globalName),
                                    parameters=[],
                                    controls=controls,
                                    targets=targets,
                                    is_adj=False)
        return


class PyKernel(object):
    """
    The :class:`Kernel` provides an API for dynamically constructing quantum
    circuits.  The :class:`Kernel` programmatically represents the circuit as an
    MLIR function using the Quake dialect.

    Attributes:
        name (:obj:`str`): The name of the :class:`Kernel` function. Read-only.
        arguments (List[:class:`QuakeValue`]): The arguments accepted by the 
            :class:`Kernel` function. Read-only.
        argument_count (int): The number of arguments accepted by the 
            :class:`Kernel` function. Read-only.

    """

    def __init__(self, argTypeList):
        self.ctx = getMLIRContext()

        self.conditionalOnMeasure = False
        self.regCounter = 0
        self.loc = Location.unknown(context=self.ctx)
        self.module = Module.create(loc=self.loc)
        self.uniqId = id(self)
        self.uniqName = "PythonKernelBuilderInstance.." + hex(self.uniqId)
        self.name = self.uniqName
        self.funcName = nvqppPrefix + self.name
        self.funcNameEntryPoint = self.uniqName + '.PyKernelFakeEntryPoint'
        strAttr = StringAttr.get(self.funcNameEntryPoint, context=self.ctx)
        attr = DictAttr.get({self.funcName: strAttr}, context=self.ctx)
        self.module.operation.attributes.__setitem__('quake.mangled_name_map',
                                                     attr)

        self.capturedDataStorage = CapturedDataStorage(ctx=self.ctx,
                                                       loc=self.loc,
                                                       name=self.name,
                                                       module=self.module)
        # List of in-place applied noise channels (rather than pre-registered
        # noise classes)
        self.appliedNoiseChannels = []

        with self.ctx, InsertionPoint(self.module.body), self.loc:
            self.mlirArgTypes = [
                mlirTypeFromPyType(argType[0], self.ctx, argInstance=argType[1])
                for argType in
                [self.__processArgType(ty) for ty in argTypeList]
            ]

            self.funcOp = func.FuncOp(self.funcName, (self.mlirArgTypes, []),
                                      loc=self.loc)
            self.funcOp.attributes.__setitem__('cudaq-entrypoint',
                                               UnitAttr.get())
            self.funcOp.attributes.__setitem__('cudaq-kernel', UnitAttr.get())
            e = self.funcOp.add_entry_block()
            self.arguments = [self.__createQuakeValue(b) for b in e.arguments]
            self.argument_count = len(self.arguments)

            with InsertionPoint(e):
                func.ReturnOp([])

            self.insertPoint = InsertionPoint.at_block_begin(e)

        self._finalizer = weakref.finalize(self, PyKernel._cleanup,
                                           self.capturedDataStorage)

    @staticmethod
    def _cleanup(capturedDataStorage):
        """
        Cleanup function to be called when the `PyKernel` instance is garbage
        collected. This resource management method is used with
        `weakref.finalize()` to ensure proper cleanup of resources. Note that
        this method is intentionally empty since `CapturedDataStorage` has its
        own `finalizer`. However, it is still included for maintaining the
        reference to `CapturedDataStorage` until the `PyKernel` instance is
        garbage collected ensuring proper cleanup order.
        """
        pass

    def __processArgType(self, ty):
        """
        Process input argument type. Specifically, try to infer the element type
        for a list, e.g. list[float].
        """
        if ty in [cudaq_runtime.qvector, cudaq_runtime.qubit]:
            return ty, None
        if get_origin(ty) == list or isinstance(ty, list):
            if '[' in str(ty) and ']' in str(ty):
                allowedTypeMap = {
                    'int': int,
                    'bool': bool,
                    'float': float,
                    'complex': complex,
                    'numpy.complex128': np.complex128,
                    'numpy.complex64': np.complex64,
                    'pauli_word': cudaq_runtime.pauli_word
                }
                # Infer the slice type
                result = re.search(r'ist\[(.*)\]', str(ty))
                eleTyName = result.group(1)
                if 'cudaq_runtime.pauli_word' in str(ty):
                    eleTyName = 'pauli_word'
                pyType = allowedTypeMap[eleTyName]
                if eleTyName != None and eleTyName in allowedTypeMap:
                    return list, [pyType()]
                emitFatalError(f'Invalid type for kernel builder {ty}')
        return ty, None

    def getIntegerAttr(self, type, value):
        """
        Return an MLIR Integer Attribute of the given IntegerType.
        """
        return IntegerAttr.get(type, value)

    def getIntegerType(self, width=64):
        """
        Return an MLIR `IntegerType` of the given bit width (defaults to 64
        bits).
        """
        return IntegerType.get_signless(width)

    def getConstantInt(self, value, width=64):
        """
        Create a constant integer operation and return its MLIR result Value.
        Takes as input the concrete integer value. Can specify the integer bit
        width.
        """
        ty = self.getIntegerType(width)
        return arith.ConstantOp(ty, self.getIntegerAttr(ty, value)).result

    def getFloatType(self, width=64):
        """
        Return an MLIR float type (single or double precision).
        """
        # Note:
        # `numpy.float64` is the same as `float` type, with width of 64 bit.
        # `numpy.float32` type has width of 32 bit.
        return F64Type.get() if width == 64 else F32Type.get()

    def getFloatAttr(self, type, value):
        """
        Return an MLIR float attribute (single or double precision).
        """
        return FloatAttr.get(type, value)

    def getConstantFloat(self, value, width=64):
        """
        Create a constant float operation and return its MLIR result Value.
        Takes as input the concrete float value.
        """
        ty = self.getFloatType(width=width)
        return self.getConstantFloatWithType(value, ty)

    def getConstantFloatWithType(self, value, ty):
        """
        Create a constant float operation and return its MLIR result Value.
        Takes as input the concrete float value.
        """
        return arith.ConstantOp(ty, self.getFloatAttr(ty, value)).result

    def getComplexType(self, width=64):
        """
        Return an MLIR complex type (single or double precision).
        """
        # Note:
        # `numpy.complex128` is the same as `complex` type,
        # with element width of 64bit (`np.complex64` and `float`)
        # `numpy.complex64` type has element type of `np.float32`.
        return self.getComplexTypeWithElementType(
            self.getFloatType(width=width))

    def getComplexTypeWithElementType(self, eTy):
        """
        Return an MLIR complex type (single or double precision).
        """
        return ComplexType.get(eTy)

    def simulationPrecision(self):
        """
        Return precision for the current simulation backend, see
        `cudaq_runtime.SimulationPrecision`.
        """
        target = cudaq_runtime.get_target()
        return target.get_precision()

    def simulationDType(self):
        """
        Return the data type for the current simulation backend, either
        `numpy.complex128` or `numpy.complex64`.
        """
        if self.simulationPrecision() == cudaq_runtime.SimulationPrecision.fp64:
            return self.getComplexType(width=64)
        return self.getComplexType(width=32)

    def ifPointerThenLoad(self, value):
        """
        If the given value is of pointer type, load the pointer and return that
        new value.
        """
        if cc.PointerType.isinstance(value.type):
            return cc.LoadOp(value).result
        return value

    def ifNotPointerThenStore(self, value):
        """
        If the given value is not of a pointer type, allocate a slot on the
        stack, store the the value in the slot, and return the slot address.
        """
        if not cc.PointerType.isinstance(value.type):
            slot = cc.AllocaOp(cc.PointerType.get(value.type, self.ctx),
                               TypeAttr.get(value.type)).result
            cc.StoreOp(value, slot)
            return slot
        return value

    def __createStdvecWithKnownValues(self, listElementValues):
        # Turn this List into a StdVec<T>
        arrSize = self.getConstantInt(len(listElementValues))
        elemTy = listElementValues[0].type if len(
            listElementValues) > 0 else self.getFloatType()
        arrTy = cc.ArrayType.get(elemTy)
        alloca = cc.AllocaOp(cc.PointerType.get(arrTy),
                             TypeAttr.get(elemTy),
                             seqSize=arrSize).result

        for i, v in enumerate(listElementValues):
            eleAddr = cc.ComputePtrOp(
                cc.PointerType.get(elemTy), alloca, [self.getConstantInt(i)],
                DenseI32ArrayAttr.get([kDynamicPtrIndex],
                                      context=self.ctx)).result
            cc.StoreOp(v, eleAddr)

        vecTy = elemTy
        if cc.PointerType.isinstance(vecTy):
            vecTy = cc.PointerType.getElementType(vecTy)

        return cc.StdvecInitOp(cc.StdvecType.get(vecTy), alloca,
                               length=arrSize).result

    def promoteOperandType(self, ty, operand):
        if ComplexType.isinstance(ty):
            complexType = ComplexType(ty)
            floatType = complexType.element_type
            if ComplexType.isinstance(operand.type):
                otherComplexType = ComplexType(operand.type)
                otherFloatType = otherComplexType.element_type
                if (floatType != otherFloatType):
                    real = self.promoteOperandType(
                        floatType,
                        complexDialect.ReOp(operand).result)
                    imag = self.promoteOperandType(
                        floatType,
                        complexDialect.ImOp(operand).result)
                    operand = complexDialect.CreateOp(complexType, real,
                                                      imag).result
            else:
                real = self.promoteOperandType(floatType, operand)
                imag = self.getConstantFloatWithType(0.0, floatType)
                operand = complexDialect.CreateOp(complexType, real,
                                                  imag).result

        if F64Type.isinstance(ty):
            if F32Type.isinstance(operand.type):
                operand = arith.ExtFOp(ty, operand).result
            if IntegerType.isinstance(operand.type):
                operand = arith.SIToFPOp(ty, operand).result

        if F32Type.isinstance(ty):
            if F64Type.isinstance(operand.type):
                operand = arith.TruncFOp(ty, operand).result
            if IntegerType.isinstance(operand.type):
                operand = arith.SIToFPOp(ty, operand).result

        return operand

    def __getMLIRValueFromPythonArg(self, arg, argTy):
        """
        Given a python runtime argument, create and return an equivalent
        constant MLIR Value.
        """
        pyType = type(arg)
        mlirType = mlirTypeFromPyType(pyType,
                                      self.ctx,
                                      argInstance=arg,
                                      argTypeToCompareTo=argTy)

        if IntegerType.isinstance(mlirType):
            return self.getConstantInt(arg, mlirType.width)

        if F64Type.isinstance(mlirType):
            return self.getConstantFloat(arg)

        if ComplexType.isinstance(mlirType):
            return complexDialect.CreateOp(mlirType,
                                           self.getConstantFloat(arg.real),
                                           self.getConstantFloat(
                                               arg.imag)).result

        if cc.StdvecType.isinstance(mlirType):
            size = self.getConstantInt(len(arg))
            eleTy = cc.StdvecType.getElementType(mlirType)
            arrTy = cc.ArrayType.get(eleTy, context=self.ctx)
            alloca = cc.AllocaOp(cc.PointerType.get(arrTy, self.ctx),
                                 TypeAttr.get(eleTy),
                                 seqSize=size).result

            def body(idx):
                eleAddr = cc.ComputePtrOp(
                    cc.PointerType.get(eleTy, self.ctx), alloca, [idx],
                    DenseI32ArrayAttr.get([-2147483648],
                                          context=self.ctx)).result
                element = arg[body.counter]
                elementVal = None
                if IntegerType.isinstance(eleTy):
                    elementVal = self.getConstantInt(element)
                elif F64Type.isinstance(eleTy):
                    elementVal = self.getConstantFloat(element)
                elif cc.StdvecType.isinstance(eleTy):
                    elementVal = self.__getMLIRValueFromPythonArg(
                        element, eleTy)
                else:
                    emitFatalError(
                        f"CUDA-Q kernel builder could not process runtime "
                        f"list-like element type ({pyType}).")

                cc.StoreOp(elementVal, eleAddr)
                # Python is weird, but interesting.
                body.counter += 1

            body.counter = 0
            self.createInvariantForLoop(size, body)
            return cc.StdvecInitOp(cc.StdvecType.get(eleTy, self.ctx),
                                   alloca,
                                   length=size).result

        emitFatalError(
            f"CUDA-Q kernel builder could not translate runtime argument of "
            f"type {pyType} to internal IR value.")

    def createInvariantForLoop(self,
                               endVal,
                               bodyBuilder,
                               startVal=None,
                               stepVal=None,
                               isDecrementing=False):
        """
        Create an invariant loop using the CC dialect. 
        """
        startVal = self.getConstantInt(0) if startVal == None else startVal
        stepVal = self.getConstantInt(1) if stepVal == None else stepVal

        iTy = self.getIntegerType()
        inputs = [startVal]
        resultTys = [iTy]

        loop = cc.LoopOp(resultTys, inputs, BoolAttr.get(False))

        whileBlock = Block.create_at_start(loop.whileRegion, [iTy])
        with InsertionPoint(whileBlock):
            condPred = IntegerAttr.get(
                iTy, 2) if not isDecrementing else IntegerAttr.get(iTy, 4)
            cc.ConditionOp(
                arith.CmpIOp(condPred, whileBlock.arguments[0], endVal).result,
                whileBlock.arguments)

        bodyBlock = Block.create_at_start(loop.bodyRegion, [iTy])
        with InsertionPoint(bodyBlock):
            bodyBuilder(bodyBlock.arguments[0])
            cc.ContinueOp(bodyBlock.arguments)

        stepBlock = Block.create_at_start(loop.stepRegion, [iTy])
        with InsertionPoint(stepBlock):
            incr = arith.AddIOp(stepBlock.arguments[0], stepVal).result
            cc.ContinueOp([incr])

        loop.attributes.__setitem__('invariant', UnitAttr.get())
        return

    def __createQuakeValue(self, value):
        return QuakeValue(value, self)

    def __cloneOrGetFunction(self, astName, currentModule, target):
        """
        Get a the function with the given name.
        If the function is already present in `currentModule`, just return it.
        Otherwise, determine if `target` is a builder or a decorator and merge
        its module into `currentModule`. Once merged, return the function (from
        the current module!) and the current module.
        """
        thisSymbolTable = SymbolTable(currentModule.operation)
        if astName in thisSymbolTable:
            return thisSymbolTable[astName], currentModule
        if isa_kernel_decorator(target):
            otherModule = target.qkeModule
            fulluniq = nvqppPrefix + target.uniqName
            cudaq_runtime.updateModule(fulluniq, currentModule, otherModule)
            fn = recover_func_op(currentModule, fulluniq)
            assert fn and "function may not disappear from module"
            return fn, currentModule
        otherModule = target.module
        cudaq_runtime.updateModule(target.funcName, currentModule, otherModule)
        fn = recover_func_op(currentModule, target.funcName)
        assert fn and "function may not disappear from module"
        return fn, currentModule

    def __addAllCalledFunctionsRecursively(self, otherFunc, currentModule,
                                           otherModule):
        """
        Search the given `FuncOp` for all `CallOps` recursively.  If found, see
        if the called function is in the current `ModuleOp` for this
        `kernel_builder`, if so do nothing. If it is not found, then find it in
        the other `ModuleOp`, clone it, and add it to this `ModuleOp`.
        """

        def walk(topLevel, functor):
            for region in topLevel.regions:
                for block in region:
                    for op in block:
                        functor(op)
                        walk(op, functor)

        def visitAllCallOps(funcOp):

            def functor(op):
                calleeName = ''
                if isinstance(op, func.CallOp) or isinstance(op, quake.ApplyOp):
                    calleeName = FlatSymbolRefAttr(
                        op.attributes['callee']).value

                if len(calleeName) == 0:
                    return

                currentST = SymbolTable(currentModule.operation)
                if calleeName in currentST:
                    return

                otherST = SymbolTable(otherModule.operation)
                if calleeName not in otherST:
                    emitFatalError(
                        f"Invalid called function `{calleeName}`- cannot find"
                        f" the function in the symbol table")

                cloned = otherST[calleeName].operation.clone()
                if 'cudaq-entrypoint' in cloned.operation.attributes:
                    cloned.operation.attributes.__delitem__('cudaq-entrypoint')
                print("adding", cloned)
                currentModule.body.append(cloned)

                visitAllCallOps(cloned)

            walk(funcOp, functor)

        visitAllCallOps(otherFunc)
        return

    def __applyControlOrAdjoint(self, target, isAdjoint, controls, *args):
        """
        Utility method for adding a Quake `ApplyOp` in the case of
        `cudaq.control` or `cudaq.adjoint`. This function will search
        recursively for all required function operations and add them to the
        module.
        """
        if hasattr(self, 'qkeModule'):
            del self.qkeModule
        with self.insertPoint, self.loc:
            if isinstance(target, cc.CreateLambdaOp):
                otherFuncCloned = target
                otherModule = self.module
                otherFTy = FunctionType(
                    TypeAttr(target.attributes['function_type']).value).inputs
            else:
                otherFuncCloned, otherModule = self.__cloneOrGetFunction(
                    target.name, self.module, target)
                assert isinstance(otherFuncCloned, func.FuncOp)
                self.__addAllCalledFunctionsRecursively(otherFuncCloned,
                                                        self.module,
                                                        otherModule)
                otherFTy = []
                for a in otherFuncCloned.body.blocks[0].arguments:
                    otherFTy.append(a.type)
            mlirValues = []
            for i, v in enumerate(args):
                argTy = otherFTy[i]
                if not isinstance(v, QuakeValue):
                    # here we have to map constant Python data to an MLIR Value
                    value = self.__getMLIRValueFromPythonArg(v, argTy)
                else:
                    value = v.mlirValue
                inTy = value.type

                if (quake.VeqType.isinstance(inTy) and
                        quake.VeqType.isinstance(argTy)):
                    if quake.VeqType.hasSpecifiedSize(
                            inTy) and not quake.VeqType.hasSpecifiedSize(argTy):
                        value = quake.RelaxSizeOp(argTy, value).result

                mlirValues.append(value)
            if isAdjoint or len(controls) > 0:
                quake.ApplyOp([], [],
                              controls,
                              mlirValues,
                              callee=FlatSymbolRefAttr.get(
                                  otherFuncCloned.name.value),
                              is_adj=isAdjoint)
            elif isinstance(otherFuncCloned, cc.CreateLambdaOp):
                cc.CallCallableOp([], otherFuncCloned, mlirValues)
            else:
                func.CallOp(otherFuncCloned, mlirValues)

    def __str__(self, canonicalize=True):
        """
        Return a string representation of this kernels MLIR Module.
        """
        if canonicalize:
            pm = PassManager.parse(
                "builtin.module(func.func(unwind-lowering,canonicalize,"
                "cse,quake-add-metadata),quake-propagate-metadata)",
                context=self.ctx)
            cloned = cudaq_runtime.cloneModule(self.module)
            pm.run(cloned)
            return str(cloned)
        return str(self.module)

    def init_qalloc(self, statePtr, wasCreated):
        """
        Generate pattern to convert `initializer` to a `state` object.
        """
        with self.ctx, self.insertPoint, self.loc:
            i64Ty = self.getIntegerType()
            veqTy = quake.VeqType.get()
            numQubits = quake.GetNumberOfQubitsOp(i64Ty, statePtr).result
            qubits = quake.AllocaOp(veqTy, size=numQubits).result
            ini = quake.InitializeStateOp(veqTy, qubits, statePtr).result
            if wasCreated:
                quake.DeleteStateOp(statePtr)
            return ini

    def get_state_ref(self, stateWrapper):
        with self.ctx, self.insertPoint, self.loc:
            stateTy = cc.PointerType.get(cc.StateType.get())
            refVal = stateWrapper.get_state_refval()
            i64Ty = self.getIntegerType()
            intAttr = IntegerAttr.get(i64Ty, refVal)
            valRef = arith.ConstantOp(i64Ty, intAttr).result
            return cc.CastOp(stateTy, valRef)

    def qalloc(self, initializer=None):
        """
        Allocate a register of qubits of size `qubit_count` and return a handle
        to them as a :class:`QuakeValue`.

        Args:
            initializer (Union[`int`,`QuakeValue`, `list[T]`): The number of
            qubits to allocate or a concrete state to allocate and initialize
            the qubits.
        Returns:
            :class:`QuakeValue`: A handle to the allocated qubits in the MLIR.

        ```python
            # Example:
            kernel = cudaq.make_kernel()
            qubits = kernel.qalloc(10)
        ```
        """
        with self.ctx, self.insertPoint, self.loc:
            # If the initializer is an integer, create `veq<N>`
            if isinstance(initializer, int):
                veqTy = quake.VeqType.get(initializer)
                return self.__createQuakeValue(quake.AllocaOp(veqTy).result)

            if isinstance(initializer, list):
                initializer = np.array(initializer, dtype=type(initializer[0]))

            if isinstance(initializer, np.ndarray):
                if len(initializer.shape) != 1:
                    raise RuntimeError("invalid initializer for qalloc "
                                       "(np.ndarray must be 1D, vector-like)")

                if initializer.dtype not in [
                        complex, np.complex128, np.complex64, float, np.float64,
                        np.float32, int
                ]:
                    raise RuntimeError("invalid initializer for qalloc (must "
                                       "be int, float, or complex dtype)")

                # Get current simulation precision and convert the initializer
                # if needed.
                simulationPrecision = self.simulationPrecision()
                if simulationPrecision == cudaq_runtime.SimulationPrecision.fp64:
                    if initializer.dtype not in [complex, np.complex128]:
                        initializer = initializer.astype(dtype=np.complex128)

                if simulationPrecision == cudaq_runtime.SimulationPrecision.fp32:
                    if initializer.dtype != np.complex64:
                        initializer = initializer.astype(dtype=np.complex64)

                if initializer.dtype == np.complex64:
                    fltTy = self.getFloatType(width=32)
                    eleTy = ComplexType.get(fltTy)
                else:
                    assert initializer.dtype == np.complex128
                    fltTy = self.getFloatType(width=64)
                    eleTy = ComplexType.get(fltTy)

                # Get the size of the array
                size = len(initializer)
                numQubits = np.log2(size)
                if not numQubits.is_integer():
                    raise RuntimeError("invalid input state size for qalloc "
                                       "(not a power of 2)")

                # check state is normalized
                norm = sum([np.conj(a) * a for a in initializer])
                if np.abs(norm.imag) > 1e-4 or np.abs(1. - norm.real) > 1e-4:
                    raise RuntimeError(
                        "invalid input state for qalloc (not normalized)")

                # Read the values from the `np.array` and copy them in a
                # constant array. The builder object resolves all symbols
                # immediately.
                arrTy = cc.ArrayType.get(eleTy, size)
                arrVals = []
                for i in range(size):
                    rePart = FloatAttr.get(fltTy, initializer[i].real)
                    imPart = FloatAttr.get(fltTy, initializer[i].imag)
                    av = ArrayAttr.get([rePart, imPart])
                    arrVals.append(av)
                vals = ArrayAttr.get(arrVals)
                data = cc.ConstantArrayOp(arrTy, vals).result
                buff = cc.AllocaOp(cc.PointerType.get(arrTy),
                                   TypeAttr.get(arrTy)).result
                cc.StoreOp(data, buff)

                i64Ty = self.getIntegerType()
                intAttr = IntegerAttr.get(i64Ty, size)
                lenny = arith.ConstantOp(i64Ty, intAttr).result
                stateTy = cc.PointerType.get(cc.StateType.get())
                statePtr = quake.CreateStateOp(stateTy, buff, lenny)
                init = self.init_qalloc(statePtr, True)
                return self.__createQuakeValue(init)

            # Captured state (from somewhere else).
            if isinstance(initializer, cudaq_runtime.State):
                stateRef = self.get_state_ref(initializer)
                init = self.init_qalloc(stateRef, False)
                return self.__createQuakeValue(init)

            # If the initializer is a QuakeValue, see if it is
            # an integer or a `stdvec` type
            if isinstance(initializer, QuakeValue):
                veqTy = quake.VeqType.get()
                if IntegerType.isinstance(initializer.mlirValue.type):
                    # This is an integer size
                    return self.__createQuakeValue(
                        quake.AllocaOp(veqTy,
                                       size=initializer.mlirValue).result)

                if cc.StdvecType.isinstance(initializer.mlirValue.type):
                    size = cc.StdvecSizeOp(self.getIntegerType(),
                                           initializer.mlirValue).result
                    value = initializer.mlirValue
                    eleTy = cc.StdvecType.getElementType(value.type)
                    numQubits = math.CountTrailingZerosOp(size).result
                    qubits = quake.AllocaOp(veqTy, size=numQubits).result
                    ptrTy = cc.PointerType.get(eleTy)
                    data = cc.StdvecDataOp(ptrTy, value).result
                    init = quake.InitializeStateOp(veqTy, qubits, data).result
                    return self.__createQuakeValue(init)

                # State pointer
                if cc.PointerType.isinstance(initializer.mlirValue.type):
                    valuePtrTy = initializer.mlirValue.type
                    valueTy = cc.PointerType.getElementType(valuePtrTy)
                    if cc.StateType.isinstance(valueTy):
                        statePtr = initializer.mlirValue

                        i64Ty = self.getIntegerType()
                        numQubits = quake.GetNumberOfQubitsOp(i64Ty,
                                                              statePtr).result

                        veqTy = quake.VeqType.get()
                        qubits = quake.AllocaOp(veqTy, size=numQubits).result
                        init = quake.InitializeStateOp(veqTy, qubits,
                                                       statePtr).result
                        return self.__createQuakeValue(init)

            # If no initializer, create a single qubit
            if initializer == None:
                qubitTy = quake.RefType.get()
                return self.__createQuakeValue(quake.AllocaOp(qubitTy).result)

            raise RuntimeError(
                f"invalid initializer argument for qalloc: {initializer}")

    def __isPauliWordType(self, ty):
        """
        A Pauli word type in our MLIR dialects is a `cc.charspan`. Return True
        if the provided type is equivalent to this, False otherwise.
        """
        return cc.CharspanType.isinstance(ty)

    def exp_pauli(self, theta, *args):
        """
        Apply a general Pauli tensor product rotation, `exp(i theta P)`, on the
        specified qubit register. The Pauli tensor product is provided as a
        string, e.g. `XXYX` for a 4-qubit term. The angle parameter can be
        provided as a concrete float or a `QuakeValue`.
        """
        with self.ctx, self.insertPoint, self.loc:
            quantumVal = None
            qubitsList = []
            pauliWordVal = None
            for arg in args:
                if isinstance(arg, cudaq_runtime.SpinOperatorTerm):
                    arg = arg.get_pauli_word()
                if isinstance(arg, cudaq_runtime.SpinOperator):
                    if arg.term_count > 1:
                        emitFatalError(
                            'exp_pauli operation requires a '
                            'SpinOperator composed of a single term.')
                    arg, *_ = arg
                    arg = arg.get_pauli_word()

                if isinstance(arg, str):
                    retTy = cc.PointerType.get(
                        cc.ArrayType.get(IntegerType.get_signless(8),
                                         int(len(arg) + 1)))
                    pauliWordVal = cc.CreateStringLiteralOp(retTy, arg)
                elif isinstance(arg, QuakeValue) and quake.VeqType.isinstance(
                        arg.mlirValue.type):
                    quantumVal = arg.mlirValue
                elif isinstance(arg, QuakeValue) and self.__isPauliWordType(
                        arg.mlirValue.type):
                    pauliWordVal = arg.mlirValue
                elif isinstance(arg, QuakeValue) and quake.RefType.isinstance(
                        arg.mlirValue.type):
                    qubitsList.append(arg.mlirValue)

            thetaVal = None
            if isinstance(theta, float):
                fty = mlirTypeFromPyType(float, self.ctx)
                thetaVal = arith.ConstantOp(fty, FloatAttr.get(fty,
                                                               theta)).result
            else:
                thetaVal = theta.mlirValue

            if len(qubitsList) > 0:
                quantumVal = quake.ConcatOp(
                    quake.VeqType.get(),
                    [quantumVal] if quantumVal is not None else [] +
                    qubitsList).result
            quake.ExpPauliOp([], [thetaVal], [], [quantumVal],
                             pauli=pauliWordVal)

    def givens_rotation(self, angle, qubitA, qubitB):
        """
        Add Givens rotation kernel (theta angle as a QuakeValue) to the kernel
        builder object.
        """
        givens_builder(self, angle, qubitA, qubitB)

    def fermionic_swap(self, angle, qubitA, qubitB):
        """
        Add Fermionic SWAP rotation kernel (phi angle as a QuakeValue) to the
        kernel builder object.
        """
        fermionic_swap_builder(self, angle, qubitA, qubitB)

    def from_state(self, qubits, state):
        emitFatalError("from_state not implemented.")

    def u3(self, theta, phi, delta, target):
        """
        Apply the universal three-parameters operator to target qubit.  The
        three parameters are Euler angles - θ, φ, and λ.

        ```python
            # Example
            kernel = cudaq.make_kernel()
            q = cudaq.qubit()
            kernel.u3(np.pi, np.pi, np.pi / 2, q)
        ```
        """
        with self.ctx, self.insertPoint, self.loc:
            parameters = [
                get_parameter_value(self, p) for p in [theta, phi, delta]
            ]

            if quake.RefType.isinstance(target.mlirValue.type):
                quake.U3Op([], parameters, [], [target.mlirValue])
                return

            # Must be a `veq`, get the size
            size = quake.VeqSizeOp(self.getIntegerType(), target.mlirValue)

            def body(idx):
                extracted = quake.ExtractRefOp(quake.RefType.get(),
                                               target.mlirValue,
                                               -1,
                                               index=idx).result
                quake.U3Op([], parameters, [], [extracted])

            self.createInvariantForLoop(size, body)

    def cu3(self, theta, phi, delta, controls, target):
        """
        Controlled u3 operation.  The controls parameter is expected to be a
        list of QuakeValue.

        ```python
            # Example:
            kernel = cudaq.make_kernel()
            qubits = kernel.qalloc(2)
            kernel.cu3(np.pi, np.pi, np.pi / 2, qubits[0], qubits[1]))
        ```
        """
        with self.insertPoint, self.loc:
            fwdControls = None
            if isinstance(controls, list):
                fwdControls = [c.mlirValue for c in controls]
            elif quake.RefType.isinstance(
                    controls.mlirValue.type) or quake.VeqType.isinstance(
                        controls.mlirValue.type):
                fwdControls = [controls.mlirValue]
            else:
                emitFatalError(f"invalid controls type for cu3.")

            quake.U3Op(
                [], [get_parameter_value(self, p) for p in [theta, phi, delta]],
                fwdControls, [target.mlirValue])

    def cswap(self, controls, qubitA, qubitB):
        """
        Controlled swap of the states of the provided qubits.  The controls
        parameter is expected to be a list of QuakeValue.

        ```python
            # Example:
            kernel = cudaq.make_kernel()
            # Allocate qubit/s to the `kernel`.
            qubits = kernel.qalloc(2)
            # Place the 0th qubit in the 1-state.
            kernel.x(qubits[0])
            # Swap their states.
            kernel.swap(qubits[0], qubits[1]))
        ```
        """
        fwdControls = None
        if isinstance(controls, list):
            fwdControls = [c.mlirValue for c in controls]
        elif quake.RefType.isinstance(
                controls.mlirValue.type) or quake.VeqType.isinstance(
                    controls.mlirValue.type):
            fwdControls = [controls.mlirValue]
        else:
            emitFatalError(
                f"Invalid control type for cswap ({type(controls)}).")

        with self.insertPoint, self.loc:
            quake.SwapOp([], [], fwdControls,
                         [qubitA.mlirValue, qubitB.mlirValue])

    def swap(self, qubitA, qubitB):
        """
        Swap the states of the provided qubits. 

        ```python
            # Example:
            kernel = cudaq.make_kernel()
            # Allocate qubit/s to the `kernel`.
            qubits = kernel.qalloc(2)
            # Place the 0th qubit in the 1-state.
            kernel.x(qubits[0])
            # Swap their states.
            kernel.swap(qubits[0], qubits[1]))
        ```
        """
        with self.insertPoint, self.loc:
            quake.SwapOp([], [], [], [qubitA.mlirValue, qubitB.mlirValue])

    def reset(self, target):
        """
        Reset the provided qubit or qubits.
        """
        with self.ctx, self.insertPoint, self.loc:
            if not quake.VeqType.isinstance(target.mlirValue.type):
                quake.ResetOp([], target.mlirValue)
                return

            # target is a VeqType
            if quake.VeqType.hasSpecifiedSize(target.mlirValue.type):
                size = quake.VeqType.getSize(target.mlirValue.type)
                for i in range(size):
                    extracted = quake.ExtractRefOp(quake.RefType.get(),
                                                   target.mlirValue, i).result
                    quake.ResetOp([], extracted)
                return
            else:
                emitFatalError(
                    'reset operation broadcasting on qvector not supported yet.'
                )

    def mz(self, target, regName=None):
        """
        Measure the given qubit or qubits in the Z-basis. The optional
        `register_name` may be used to retrieve results of this measurement
        after execution on the QPU. If the measurement call is saved as a
        variable, it will return a :class:`QuakeValue` handle to the measurement
        instruction.

        Args:
        target (:class:`QuakeValue`): The qubit or qubits to measure.
        register_name (Optional[:obj:`str`]): The optional name to provide the 
            results of the measurement. Defaults to an empty string. 

        Returns:
        :class:`QuakeValue`: A handle to this measurement operation in the MLIR.

        Note:
        Measurements may be applied both mid-circuit and at the end of 
        the circuit. Mid-circuit measurements are currently only supported 
        through the use of :func:`c_if`.

        ```python
            # Example:
            kernel = cudaq.make_kernel()
            # Allocate qubit/s to measure.
            qubit = kernel.qalloc()
            # Measure the qubit/s in the Z-basis.
            kernel.mz(target=qubit))
        ```
        """
        with self.ctx, self.insertPoint, self.loc:
            i1Ty = IntegerType.get_signless(1)
            qubitTy = target.mlirValue.type
            retTy = i1Ty
            measTy = quake.MeasureType.get()
            stdvecTy = cc.StdvecType.get(i1Ty)
            if quake.VeqType.isinstance(target.mlirValue.type):
                retTy = stdvecTy
                measTy = cc.StdvecType.get(measTy)
            if regName is not None:
                res = quake.MzOp(measTy, [], [target.mlirValue],
                                 registerName=StringAttr.get(regName,
                                                             context=self.ctx))
            else:
                res = quake.MzOp(measTy, [], [target.mlirValue])
            disc = quake.DiscriminateOp(retTy, res)
            return self.__createQuakeValue(disc.result)

    def mx(self, target, regName=None):
        """
        Measure the given qubit or qubits in the X-basis. The optional
        `register_name` may be used to retrieve results of this measurement
        after execution on the QPU. If the measurement call is saved as a
        variable, it will return a :class:`QuakeValue` handle to the measurement
        instruction.

        Args:
        target (:class:`QuakeValue`): The qubit or qubits to measure.
        register_name (Optional[:obj:`str`]): The optional name to provide the 
            results of the measurement. Defaults to an empty string. 

        Returns:
        :class:`QuakeValue`: A handle to this measurement operation in the MLIR.

        Note:
        Measurements may be applied both mid-circuit and at the end of 
        the circuit. Mid-circuit measurements are currently only supported 
        through the use of :func:`c_if`.

        ```python
            kernel = cudaq.make_kernel()
            # Allocate qubit/s to measure.
            qubit = kernel.qalloc()
            # Measure the qubit/s in the X-basis.
            kernel.mx(qubit))
        ```
        """
        with self.ctx, self.insertPoint, self.loc:
            i1Ty = IntegerType.get_signless(1)
            qubitTy = target.mlirValue.type
            retTy = i1Ty
            measTy = quake.MeasureType.get()
            stdvecTy = cc.StdvecType.get(i1Ty)
            if quake.VeqType.isinstance(target.mlirValue.type):
                retTy = stdvecTy
                measTy = cc.StdvecType.get(measTy)
            if regName is not None:
                res = quake.MxOp(measTy, [], [target.mlirValue],
                                 registerName=StringAttr.get(regName,
                                                             context=self.ctx))
            else:
                res = quake.MxOp(measTy, [], [target.mlirValue])
            disc = quake.DiscriminateOp(retTy, res)
            return self.__createQuakeValue(disc.result)

    def my(self, target, regName=None):
        """
        Measure the given qubit or qubits in the Y-basis. The optional
        `register_name` may be used to retrieve results of this measurement
        after execution on the QPU. If the measurement call is saved as a
        variable, it will return a :class:`QuakeValue` handle to the measurement
        instruction.

        Args:
        target (:class:`QuakeValue`): The qubit or qubits to measure.
        register_name (Optional[:obj:`str`]): The optional name to provide the 
            results of the measurement. Defaults to an empty string. 

        Returns:
        :class:`QuakeValue`: A handle to this measurement operation in the MLIR.

        Note:
        Measurements may be applied both mid-circuit and at the end of 
        the circuit. Mid-circuit measurements are currently only supported 
        through the use of :func:`c_if`.

        ```python
            # Example:
            kernel = cudaq.make_kernel()
            # Allocate qubit/s to measure.
            qubit = kernel.qalloc()
            # Measure the qubit/s in the Y-basis.
            kernel.my(qubit))
        ```
        """
        with self.ctx, self.insertPoint, self.loc:
            i1Ty = IntegerType.get_signless(1)
            qubitTy = target.mlirValue.type
            retTy = i1Ty
            measTy = quake.MeasureType.get()
            stdvecTy = cc.StdvecType.get(i1Ty)
            if quake.VeqType.isinstance(target.mlirValue.type):
                retTy = stdvecTy
                measTy = cc.StdvecType.get(measTy)
            if regName is not None:
                res = quake.MyOp(measTy, [], [target.mlirValue],
                                 registerName=StringAttr.get(regName,
                                                             context=self.ctx))
            else:
                res = quake.MyOp(measTy, [], [target.mlirValue])
            disc = quake.DiscriminateOp(retTy, res)
            return self.__createQuakeValue(disc.result)

    def adjoint(self, otherKernel, *target_arguments):
        """
        Apply the adjoint of the `target` kernel in-place to `self`.

        Args:
        target (:class:`Kernel`): The kernel to take the adjoint of.
        *target_arguments (Optional[:class:`QuakeValue`]): The arguments to the 
            `target` kernel. Leave empty if the `target` kernel doesn't accept 
            any arguments.

        Raises:
        RuntimeError: if the `*target_arguments` passed to the adjoint call
            don't match the argument signature of `target`.

        ```python
            # Example:
            target_kernel = cudaq.make_kernel()
            qubit = target_kernel.qalloc()
            target_kernel.x(qubit)
            # Apply the adjoint of `target_kernel` to `kernel`.
            kernel = cudaq.make_kernel()
            kernel.adjoint(target_kernel))
        ```
        """
        self.__applyControlOrAdjoint(otherKernel, True, [], *target_arguments)
        return

    def control(self, target, control, *target_arguments):
        """
        Apply the `target` kernel as a controlled operation in-place to `self`.
        Uses the provided `control` as control qubit/s for the operation.

        Args:
        target (:class:`Kernel`): The kernel to apply as a controlled 
            operation in-place to self.
        control (:class:`QuakeValue`): The control qubit or register to 
            use when applying `target`.
        *target_arguments (Optional[:class:`QuakeValue`]): The arguments to the 
            `target` kernel. Leave empty if the `target` kernel doesn't accept 
            any arguments.

        Raises:
        RuntimeError: if the `*target_arguments` passed to the control 
            call don't match the argument signature of `target`.

        ```python
            # Example:
            # Create a `Kernel` that accepts a qubit as an argument.
            # Apply an X-gate on that qubit.
            target_kernel, qubit = cudaq.make_kernel(cudaq.qubit)
            target_kernel.x(qubit)
            # Create another `Kernel` that will apply `target_kernel`
            # as a controlled operation.
            kernel = cudaq.make_kernel()
            control_qubit = kernel.qalloc()
            target_qubit = kernel.qalloc()
            # In this case, `control` performs the equivalent of a 
            # controlled-X gate between `control_qubit` and `target_qubit`.
            kernel.control(target_kernel, control_qubit, target_qubit))
        ```
        """
        self.__applyControlOrAdjoint(target, False, [control.mlirValue],
                                     *target_arguments)
        return

    def apply_call(self, target, *target_arguments):
        """
        Apply a call to the given `target` kernel within the function-body of
        `self` at the provided target arguments.

        Args:
        target (:class:`Kernel`): The kernel to call from within `self`.
        *target_arguments (Optional[:class:`QuakeValue`]):
            The arguments to the `target` kernel. Leave empty if the `target`
            kernel doesn't accept any arguments.

        Raises:
        RuntimeError: if the `*target_arguments` passed to the apply 
            call don't match the argument signature of `target`.

        ```python
            # Example:
            # Build a `Kernel` that's parameterized by a `cudaq.qubit`.
            target_kernel, other_qubit = cudaq.make_kernel(cudaq.qubit)
            target_kernel.x(other_qubit)
            # Build a `Kernel` that will call `target_kernel` within its
            # own function body.
            kernel = cudaq.make_kernel()
            qubit = kernel.qalloc()
            # Use `qubit` as the argument to `target_kernel`.
            kernel.apply_call(target_kernel, qubit)
            # The final measurement of `qubit` should return the 1-state.
            kernel.mz(qubit))
        ```
        """
        if isa_kernel_decorator(target):
            target = self.resolve_callable_arg(self.insertPoint, target)
        self.__applyControlOrAdjoint(target, False, [], *target_arguments)

    def resolve_callable_arg(self, insPt, target):
        """
        `target` must be a callable. For a simple callable (a `func.FuncOp`),
        resolution is trivial. If the callable is a decorator with lambda lifted
        arguments, then all the lifted arguments must be resolved into a
        closure here.
        Returns a `CreateLambdaOp` closure.
        """
        cudaq_runtime.updateModule(self.uniqName, self.module, target.qkeModule)
        # build the closure to capture the lifted `args`
        thisPyMod = recover_calling_module()
        if target.defModule != thisPyMod:
            m = target.defModule
        else:
            m = None
        fulluniq = nvqppPrefix + target.uniqName
        fn = recover_func_op(self.module, fulluniq)
        funcTy = fn.type
        if target.firstLiftedPos:
            moduloInTys = funcTy.inputs[:target.firstLiftedPos]
        else:
            moduloInTys = funcTy.inputs
        callableTy = cc.CallableType.get(self.ctx, moduloInTys, funcTy.results)
        with insPt, self.loc:
            lamb = cc.CreateLambdaOp(callableTy, loc=self.loc)
            lamb.attributes.__setitem__('function_type', TypeAttr.get(funcTy))
            initRegion = lamb.initRegion
            initBlock = Block.create_at_start(initRegion, moduloInTys)
            inner = InsertionPoint(initBlock)
            with inner:
                vs = []
                for ba in initBlock.arguments:
                    vs.append(ba)
                for i, a in enumerate(target.liftedArgs):
                    v = recover_value_of(a, m)
                    if isa_kernel_decorator(v):
                        # The recursive step
                        v = self.resolve_callable_arg(inner, v)
                    else:
                        argTy = funcTy.inputs[target.firstLiftedPos + i]
                        v = self.__getMLIRValueFromPythonArg(v, argTy)
                    vs.append(v)
                if funcTy.results:
                    call = func.CallOp(fn, vs).result
                    cc.ReturnOp(call.results)
                else:
                    func.CallOp(fn, vs)
                    cc.ReturnOp([])
                return lamb

    def c_if(self, measurement, function):
        """
        Apply the `function` to the :class:`Kernel` if the provided single-qubit
        `measurement` returns the 1-state.

        Args:
        measurement (:class:`QuakeValue`): The handle to the single qubit 
            measurement instruction.
        function (Callable): The function to conditionally apply to the 
            :class:`Kernel`.

        Raises:
        RuntimeError: If the provided `measurement` is on more than 1 qubit.

        ```python
            # Example:
            # Create a kernel and allocate a single qubit.
            kernel = cudaq.make_kernel()
            qubit = kernel.qalloc()
            # Define a function that performs certain operations on the
            # kernel and the qubit.
            def then_function():
                kernel.x(qubit)
            kernel.x(qubit)
            # Measure the qubit.
            measurement = kernel.mz(qubit)
            # Apply `then_function` to the `kernel` if the qubit was measured
            # in the 1-state.
            kernel.c_if(measurement, then_function))
        ```
        """
        with self.insertPoint, self.loc:
            conditional = measurement.mlirValue
            if not IntegerType.isinstance(conditional.type):
                emitFatalError("c_if conditional must be of type `bool`.")

            # [RFC]:
            # The register names in the conditional tests need to be double
            # checked; the code here may need to be adjusted to reflect the
            # additional quake.discriminate conversion of the measurement.
            if isinstance(conditional.owner.opview, quake.MzOp):
                regName = StringAttr(
                    conditional.owner.attributes['registerName']).value
                if len(regName) == 0:
                    conditional.owner.attributes.__setitem__(
                        'registerName',
                        StringAttr.get('auto_register_{}'.format(
                            self.regCounter)))
                    self.regCounter += 1

            if self.getIntegerType(1) != conditional.type:
                # not equal to 0, then compare with 1
                condPred = IntegerAttr.get(self.getIntegerType(), 1)
                conditional = arith.CmpIOp(condPred, conditional,
                                           self.getConstantInt(0)).result

            ifOp = cc.IfOp([], conditional, [])
            thenBlock = Block.create_at_start(ifOp.thenRegion, [])
            with InsertionPoint(thenBlock):
                tmpIp = self.insertPoint
                self.insertPoint = InsertionPoint(thenBlock)
                function()
                self.insertPoint = tmpIp
                cc.ContinueOp([])
            self.conditionalOnMeasure = True

    def for_loop(self, start, stop, function):
        """
        Add a for loop that starts from the given `start` index, ends at the
        given `stop` index (non inclusive), applying the provided `function`
        within `self` at each iteration. The step value is provided to mutate
        the iteration variable after every iteration.

        Args:
        start (int or :class:`QuakeValue`): The beginning iterator value for
            the for loop.
        stop (int or :class:`QuakeValue`): The final iterator value
            (non-inclusive) for the for loop.
        function (Callable): The callable function to apply within the `kernel`
            at each iteration.

        ```python
            # Example:
            # Create a kernel function that takes an `int` argument.
            kernel, size = cudaq.make_kernel(int)
            # Parameterize the allocated number of qubits by the int.
            qubits = kernel.qalloc(size)
            kernel.h(qubits[0])

            def foo(index: int):
                # A function that will be applied to `kernel` in a for loop.
                kernel.cx(qubits[index], qubits[index+1])

            # Create a for loop in `kernel`, parameterized by the `size`
            # argument for its `stop` iterator.
            kernel.for_loop(start=0, stop=size-1, function=foo)

            # Execute the kernel, passing along a concrete value (5) for 
            # the `size` argument.
            counts = cudaq.sample(kernel, 5)
            print(counts)
        ```
        """
        with self.insertPoint, self.loc:
            iTy = mlirTypeFromPyType(int, self.ctx)
            startVal = None
            endVal = None
            stepVal = None

            if isinstance(start, int):
                startVal = arith.ConstantOp(iTy, IntegerAttr.get(iTy,
                                                                 start)).result
            elif isinstance(start, QuakeValue):
                startVal = start.mlirValue
            else:
                emitFatalError(
                    f"invalid start value passed to for_loop: {start}")

            if isinstance(stop, int):
                endVal = arith.ConstantOp(iTy, IntegerAttr.get(iTy,
                                                               stop)).result
            elif isinstance(stop, QuakeValue):
                endVal = stop.mlirValue
            else:
                emitFatalError(f"invalid stop value passed to for_loop: {stop}")

            stepVal = arith.ConstantOp(iTy, IntegerAttr.get(iTy, 1)).result
            inputs = [startVal]
            resultTys = [iTy]
            loop = cc.LoopOp(resultTys, inputs, BoolAttr.get(False))

            whileBlock = Block.create_at_start(loop.whileRegion, [iTy])
            with InsertionPoint(whileBlock):
                condPred = IntegerAttr.get(iTy, 2)
                # if not `isDecrementing` else `IntegerAttr.get(iTy, 4)`
                cc.ConditionOp(
                    arith.CmpIOp(condPred, whileBlock.arguments[0],
                                 endVal).result, whileBlock.arguments)

            bodyBlock = Block.create_at_start(loop.bodyRegion, [iTy])
            with InsertionPoint(bodyBlock):
                tmpIp = self.insertPoint
                self.insertPoint = InsertionPoint(bodyBlock)
                function(self.__createQuakeValue(bodyBlock.arguments[0]))
                self.insertPoint = tmpIp
                cc.ContinueOp(bodyBlock.arguments)

            stepBlock = Block.create_at_start(loop.stepRegion, [iTy])
            with InsertionPoint(stepBlock):
                incr = arith.AddIOp(stepBlock.arguments[0], stepVal).result
                cc.ContinueOp([incr])
            loop.attributes.__setitem__('invariant', UnitAttr.get())

    def create_noise_channel_class(self, kraus_channel):
        class_name = "cudaq_gen_kraus_channel_" + str(uuid.uuid4())

        def initSubClass(self, *args):
            cudaq_runtime.KrausChannel.__init__(self, kraus_channel.get_ops())

        new_class = type(class_name, (cudaq_runtime.KrausChannel,), {
            "__init__": initSubClass,
            "num_parameters": 0
        })
        return new_class

    def process_channel_param(self, param):
        # Noise channel parameters
        if isinstance(param, float):
            return self.getConstantFloat(param)
        # Check that it's a MLIR value of float type
        elif isinstance(
                param,
                QuakeValue) and (F64Type.isinstance(param.mlirValue.type) or
                                 F32Type.isinstance(param.mlirValue.type)):
            return param.mlirValue
        else:
            emitFatalError("Noise channel parameter must be float")

    def apply_noise(self, noise_channel, *args):
        """
        Apply a noise channel to the provided qubit or qubits.
        """
        if isinstance(noise_channel, cudaq_runtime.KrausChannel):
            # If we have an instance of a KrausChannel, create a subclass
            noise_channel = self.create_noise_channel_class(noise_channel)
            self.appliedNoiseChannels.append(noise_channel)

        if not issubclass(noise_channel, cudaq_runtime.KrausChannel):
            if not hasattr(noise_channel, 'num_parameters'):
                emitFatalError(
                    'apply_noise kraus channels must have `num_parameters` '
                    'constant class attribute specified.')

            # We needs to have noise channel parameters + qubit arguments
            if isinstance(args[0], list):
                if len(args[0]) != noise_channel.num_parameters:
                    emitFatalError(f"Invalid number of arguments passed to "
                                   f"apply_noise for channel `{noise_channel}`")
            elif len(args) <= noise_channel.num_parameters:
                emitFatalError(f"Invalid number of arguments passed to "
                               f"apply_noise for channel `{noise_channel}`")

        with self.insertPoint, self.loc:
            noise_channel_params = []
            target_qubits = []

            if isinstance(args[0], list):
                # If the first argument is a list, assuming that it is the list
                # of noise channel parameters.
                noise_channel_params = [
                    self.process_channel_param(p) for p in args[0]
                ]
                # Qubit arguments
                for p in args[1:]:
                    if not (isinstance(p, QuakeValue) and
                            quake.RefType.isinstance(p.mlirValue.type)):
                        emitFatalError("Invalid qubit operand type")
                    target_qubits.append(p.mlirValue)
            else:
                for i, p in enumerate(args):
                    if i < noise_channel.num_parameters:
                        noise_channel_params.append(
                            self.process_channel_param(p))
                    else:
                        # Qubit arguments
                        if not (isinstance(p, QuakeValue) and
                                quake.RefType.isinstance(p.mlirValue.type)):
                            emitFatalError("Invalid qubit operand type")
                        target_qubits.append(p.mlirValue)

            params = self.__createStdvecWithKnownValues(noise_channel_params)
            asVeq = quake.ConcatOp(quake.VeqType.get(), target_qubits).result
            channel_key = hash(noise_channel)
            quake.ApplyNoiseOp([params], [asVeq],
                               key=self.getConstantInt(channel_key))

    def compile(self):
        """
        A `PyKernel` can be dynamically extended up until it is reified to be
        used in a launch scenario. We reify the kernel as-is here.
        """
        if not hasattr(self, 'qkeModule'):
            self.qkeModule = cudaq_runtime.cloneModule(self.module)
            ctx = getMLIRContext()
            pm = PassManager.parse("builtin.module(aot-prep-pipeline)",
                                   context=ctx)
            try:
                pm.run(self.qkeModule)
            except:
                raise RuntimeError("could not compile code for '" +
                                   self.uniqName + "'.")
            self.qkeModule.operation.attributes.__setitem__(
                cudaq__unique_attr_name,
                StringAttr.get(self.uniqName, context=ctx))

    def __call__(self, *args):
        """
        Just-In-Time (JIT) compile `self` (:class:`Kernel`), and call the kernel
        function at the provided concrete arguments.

        Args:
            *arguments (Optional[Any]): The concrete values to evaluate the 
                kernel function at. Leave empty if the `target` kernel doesn't 
                accept any arguments.

        ```python
            # Example:
            # Create a kernel that accepts an int and float as its 
            # arguments.
            kernel, qubit_count, angle = cudaq.make_kernel(int, float)
            # Parameterize the number of qubits by `qubit_count`.
            qubits = kernel.qalloc(qubit_count)
            # Apply an `rx` rotation on the first qubit by `angle`.
            kernel.rx(angle, qubits[0])
            # Call the `Kernel` on the given number of qubits (5) and at 
            a concrete angle (pi).
            kernel(5, 3.14))
        ```
        """
        if len(self.appliedNoiseChannels) > 0:
            noise_model = cudaq_runtime.get_noise()
            if noise_model is not None:
                # Note: the runtime would already warn about `apply_noise`
                # called but no noise model provided.  Here, we just ignore the
                # registration of inline noise applications.
                for noise_channel in self.appliedNoiseChannels:
                    noise_model.register_channel(noise_channel)

        if len(args) != len(self.mlirArgTypes):
            emitFatalError(f"Invalid number of arguments passed to kernel "
                           f"`{self.funcName}` ({len(args)} provided, "
                           f"{len(self.mlirArgTypes)} required")
        # validate the argument types
        processedArgs = []
        for i, arg in enumerate(args):
            # Handle `list[str]` separately - we allow this only for
            # `list[cudaq.pauli_word]` inputs
            if issubclass(type(arg), list) and len(arg) and all(
                    isinstance(a, str) for a in arg):
                [emitErrorIfInvalidPauli(a) for a in arg]
                processedArgs.append([cudaq_runtime.pauli_word(a) for a in arg])
                continue

            # Handle `str` input separately - we allow this for
            # `cudaq.pauli_word` inputs
            if isinstance(arg, str):
                emitErrorIfInvalidPauli(arg)
                processedArgs.append(cudaq_runtime.pauli_word(arg))
                continue

            argType = type(arg)
            listType = None
            if argType == list:
                if len(arg) == 0:
                    processedArgs.append(arg)
                    continue
                listType = list[type(arg[0])]
            mlirType = mlirTypeFromPyType(argType, self.ctx)

            if cc.StdvecType.isinstance(mlirType):
                # Support passing `list[int]` to a `list[float]` argument
                if cc.StdvecType.isinstance(self.mlirArgTypes[i]):
                    maybeCasted = supportCommonCast(mlirType,
                                                    self.mlirArgTypes[i], arg,
                                                    IntegerType, F64Type, float)
                    if maybeCasted != None:
                        processedArgs.append(maybeCasted)
                        continue

                    # Support passing `list[float]` to a `list[complex]`
                    # argument
                    maybeCasted = supportCommonCast(mlirType,
                                                    self.mlirArgTypes[i], arg,
                                                    F64Type, ComplexType,
                                                    complex)
                    if maybeCasted != None:
                        processedArgs.append(maybeCasted)
                        continue

            if (mlirType != self.mlirArgTypes[i] and
                    listType != mlirTypeToPyType(self.mlirArgTypes[i])):
                emitFatalError(
                    f"Invalid runtime argument type ({type(arg)} provided,"
                    f" {mlirTypeToPyType(self.mlirArgTypes[i])} required)")

            # Convert `numpy` arrays to lists
            if cc.StdvecType.isinstance(mlirType):
                # Validate that the length of this argument is greater than or
                # equal to the number of unique quake value extractions
                if len(arg) < len(self.arguments[i].knownUniqueExtractions):
                    emitFatalError(
                        f"Invalid runtime list argument - {len(arg)} elements "
                        f"in list but kernel code has at least "
                        f"{len(self.arguments[i].knownUniqueExtractions)} "
                        f"known unique extractions.")
                if hasattr(arg, "tolist"):
                    processedArgs.append(arg.tolist())
                else:
                    processedArgs.append(arg)
            else:
                processedArgs.append(arg)

        retTy = NoneType.get(self.module.context)
        self.compile()
        specialized = cudaq_runtime.cloneModule(self.qkeModule)
        cudaq_runtime.marshal_and_launch_module(self.name, specialized, retTy,
                                                *processedArgs)

    def __getattr__(self, attr_name):
        # Search attributes in instance, class, base classes
        try:
            return object.__getattribute__(self, attr_name)
        except AttributeError:
            raise AttributeError(f"'{attr_name}' is not supported on PyKernel")


setattr(PyKernel, 'h', partialmethod(__singleTargetOperation, 'h'))
setattr(PyKernel, 'x', partialmethod(__singleTargetOperation, 'x'))
setattr(PyKernel, 'y', partialmethod(__singleTargetOperation, 'y'))
setattr(PyKernel, 'z', partialmethod(__singleTargetOperation, 'z'))
setattr(PyKernel, 's', partialmethod(__singleTargetOperation, 's'))
setattr(PyKernel, 't', partialmethod(__singleTargetOperation, 't'))
setattr(PyKernel, 'sdg', partialmethod(__singleTargetOperation, 's',
                                       isAdj=True))
setattr(PyKernel, 'tdg', partialmethod(__singleTargetOperation, 't',
                                       isAdj=True))

setattr(PyKernel, 'ch', partialmethod(__singleTargetControlOperation, 'h'))
setattr(PyKernel, 'cx', partialmethod(__singleTargetControlOperation, 'x'))
setattr(PyKernel, 'cy', partialmethod(__singleTargetControlOperation, 'y'))
setattr(PyKernel, 'cz', partialmethod(__singleTargetControlOperation, 'z'))
setattr(PyKernel, 'cs', partialmethod(__singleTargetControlOperation, 's'))
setattr(PyKernel, 'ct', partialmethod(__singleTargetControlOperation, 't'))

setattr(PyKernel, 'rx',
        partialmethod(__singleTargetSingleParameterOperation, 'rx'))
setattr(PyKernel, 'ry',
        partialmethod(__singleTargetSingleParameterOperation, 'ry'))
setattr(PyKernel, 'rz',
        partialmethod(__singleTargetSingleParameterOperation, 'rz'))
setattr(PyKernel, 'r1',
        partialmethod(__singleTargetSingleParameterOperation, 'r1'))

setattr(PyKernel, 'crx',
        partialmethod(__singleTargetSingleParameterControlOperation, 'rx'))
setattr(PyKernel, 'cry',
        partialmethod(__singleTargetSingleParameterControlOperation, 'ry'))
setattr(PyKernel, 'crz',
        partialmethod(__singleTargetSingleParameterControlOperation, 'rz'))
setattr(PyKernel, 'cr1',
        partialmethod(__singleTargetSingleParameterControlOperation, 'r1'))


def make_kernel(*args):
    """
    Create a :class:`Kernel`: An empty kernel function to be used for quantum 
    program construction. This kernel is non-parameterized if it accepts no 
    arguments, else takes the provided types as arguments. 

    Returns a kernel if it is non-parameterized, else a tuple containing the 
    kernel and a :class:`QuakeValue` for each kernel argument.

.. code-block:: python

    # Example:
    # Non-parameterized kernel.
    kernel = cudaq.make_kernel()

    # Example:
    # Parameterized kernel that accepts an `int` and `float` as arguments.
    kernel, int_value, float_value = cudaq.make_kernel(int, float)
    """

    kernel = PyKernel([*args])
    if len([*args]) == 0:
        return kernel

    return kernel, *kernel.arguments


def isa_dynamic_kernel(object):
    """
    Return True if and only if object is an instance of PyKernel.
    """
    return isinstance(object, PyKernel)
