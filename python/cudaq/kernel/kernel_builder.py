# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from functools import partialmethod
import hashlib
import random
import re
import string
import sys
import numpy as np
from typing import get_origin, List
from .quake_value import QuakeValue
from .kernel_decorator import PyKernelDecorator
from .utils import mlirTypeFromPyType, nvqppPrefix, emitFatalError, mlirTypeToPyType, emitErrorIfInvalidPauli
from .common.givens import givens_builder
from .common.fermionic_swap import fermionic_swap_builder

from ..mlir.ir import *
from ..mlir.passmanager import *
from ..mlir.execution_engine import *
from ..mlir.dialects import quake, cc
from ..mlir.dialects import builtin, func, arith, math
from ..mlir._mlir_libs._quakeDialects import cudaq_runtime, register_all_dialects


## [PYTHON_VERSION_FIX]
## Refer: https://peps.python.org/pep-0616/
def remove_prefix(inputStr: str, prefix: str) -> str:
    if inputStr.startswith(prefix):
        return inputStr[len(prefix):]
    else:
        return inputStr[:]


qvector = cudaq_runtime.qvector

# This file reproduces the cudaq::kernel_builder in Python

# We need static initializers to run in the CAPI `ExecutionEngine`,
# so here we run a simple JIT compile at global scope
with Context():
    module = Module.parse(r"""
llvm.func @none() {
  llvm.return
}""")
    ExecutionEngine(module)


def __generalOperation(self,
                       opName,
                       parameters,
                       controls,
                       target,
                       isAdj=False,
                       context=None):
    """
    This is a utility function that applies a general quantum 
    operation to the internal PyKernel MLIR ModuleOp.
    """
    opCtor = getattr(quake, '{}Op'.format(opName.title()))

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


def __singleTargetOperation(self, opName, target, isAdj=False):
    """
    Utility function for adding a single target quantum operation to the 
    MLIR representation for the PyKernel.
    """
    with self.insertPoint, self.loc:
        __generalOperation(self,
                           opName, [], [],
                           target,
                           isAdj=isAdj,
                           context=self.ctx)


def __singleTargetControlOperation(self, opName, control, target, isAdj=False):
    """
    Utility function for adding a single target controlled quantum operation to the 
    MLIR representation for the PyKernel.
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
    Utility function for adding a single target, one parameter quantum operation to the 
    MLIR representation for the PyKernel.
    """
    with self.insertPoint, self.loc:
        paramVal = None
        if isinstance(parameter, float):
            fty = mlirTypeFromPyType(float, self.ctx)
            paramVal = arith.ConstantOp(fty, FloatAttr.get(fty, parameter))
        else:
            paramVal = parameter.mlirValue
        __generalOperation(self,
                           opName, [paramVal], [],
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
    Utility function for adding a single target, one parameter, controlled quantum operation to the 
    MLIR representation for the PyKernel.
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

        paramVal = parameter
        if isinstance(parameter, float):
            fty = mlirTypeFromPyType(float, self.ctx)
            paramVal = arith.ConstantOp(fty, FloatAttr.get(fty, parameter))
        elif isinstance(parameter, QuakeValue):
            paramVal = parameter.mlirValue

        __generalOperation(self,
                           opName, [paramVal],
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


class PyKernel(object):
    """
    The :class:`Kernel` provides an API for dynamically constructing quantum 
    circuits. The :class:`Kernel` programmatically represents the circuit as an MLIR 
    function using the Quake dialect.

    Attributes:
        name (:obj:`str`): The name of the :class:`Kernel` function. Read-only.
        arguments (List[:class:`QuakeValue`]): The arguments accepted by the 
            :class:`Kernel` function. Read-only.
        argument_count (int): The number of arguments accepted by the 
            :class:`Kernel` function. Read-only.
    """

    def __init__(self, argTypeList):
        self.ctx = Context()
        register_all_dialects(self.ctx)
        quake.register_dialect(self.ctx)
        cc.register_dialect(self.ctx)
        cudaq_runtime.registerLLVMDialectTranslation(self.ctx)

        self.stateHashes = []
        self.metadata = {'conditionalOnMeasure': False}
        self.regCounter = 0
        self.loc = Location.unknown(context=self.ctx)
        self.module = Module.create(loc=self.loc)
        self.funcName = '{}__nvqppBuilderKernel_{}'.format(
            nvqppPrefix, ''.join(
                random.choice(string.ascii_uppercase + string.digits)
                for _ in range(10)))
        self.name = remove_prefix(self.funcName, nvqppPrefix)
        self.funcNameEntryPoint = self.funcName + '_PyKernelEntryPointRewrite'
        attr = DictAttr.get(
            {
                self.funcName:
                    StringAttr.get(self.funcNameEntryPoint, context=self.ctx)
            },
            context=self.ctx)
        self.module.operation.attributes.__setitem__('quake.mangled_name_map',
                                                     attr)

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
            e = self.funcOp.add_entry_block()
            self.arguments = [self.__createQuakeValue(b) for b in e.arguments]
            self.argument_count = len(self.arguments)

            with InsertionPoint(e):
                func.ReturnOp([])

            self.insertPoint = InsertionPoint.at_block_begin(e)

    def __del__(self):
        """
        When a kernel builder is deleted we need to clean up 
        any state data if there is any.
        """
        cudaq_runtime.deletePointersToStateData(self.stateHashes)

    def __processArgType(self, ty):
        """
        Process input argument type. Specifically, try to infer the 
        element type for a list, e.g. list[float]. 
        """
        if ty in [cudaq_runtime.qvector, cudaq_runtime.qubit]:
            return ty, None
        if get_origin(ty) == list or isinstance(ty(), list):
            if '[' in str(ty) and ']' in str(ty):
                allowedTypeMap = {
                    'int': int,
                    'bool': bool,
                    'float': float,
                    'complex': complex,
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
        Return an MLIR `IntegerType` of the given bit width (defaults to 64 bits).
        """
        return IntegerType.get_signless(width)

    def getConstantInt(self, value, width=64):
        """
        Create a constant integer operation and return its MLIR result Value.
        Takes as input the concrete integer value. Can specify the integer bit width.
        """
        ty = self.getIntegerType(width)
        return arith.ConstantOp(ty, self.getIntegerAttr(ty, value)).result

    def getConstantFloat(self, value):
        """
        Create a constant float operation and return its MLIR result Value.
        Takes as input the concrete float value. 
        """
        ty = F64Type.get()
        return arith.ConstantOp(ty, FloatAttr.get(ty, value)).result

    def __getMLIRValueFromPythonArg(self, arg, argTy):
        """
        Given a python runtime argument, create and return an equivalent constant MLIR Value.
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
            return complex.CreateOp(mlirType, self.getConstantFloat(arg.real),
                                    self.getConstantFloat(arg.imag)).result

        if cc.StdvecType.isinstance(mlirType):
            size = self.getConstantInt(len(arg))
            eleTy = cc.StdvecType.getElementType(mlirType)
            arrTy = cc.ArrayType.get(self.ctx, eleTy)
            alloca = cc.AllocaOp(cc.PointerType.get(self.ctx, arrTy),
                                 TypeAttr.get(eleTy),
                                 seqSize=size).result

            def body(idx):
                eleAddr = cc.ComputePtrOp(
                    cc.PointerType.get(self.ctx, eleTy), alloca, [idx],
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
                        f"CUDA Quantum kernel builder could not process runtime list-like element type ({pyType})."
                    )

                cc.StoreOp(elementVal, eleAddr)
                # Python is weird, but interesting.
                body.counter += 1

            body.counter = 0
            self.createInvariantForLoop(size, body)
            return cc.StdvecInitOp(cc.StdvecType.get(self.ctx, eleTy), alloca,
                                   size).result

        emitFatalError(
            "CUDA Quantum kernel builder could not translate runtime argument of type {pyType} to internal IR value."
        )

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

    def __cloneOrGetFunction(self, name, currentModule, otherModule):
        """
        Get a the function with the given name. First look in the
        current `ModuleOp` for this `kernel_builder`, if found return it as is. If
        not found, find it in the other `kernel_builder` `ModuleOp` and return a
        clone of it. Throw an exception if no kernel with the given name is found
        """
        thisSymbolTable = SymbolTable(currentModule.operation)
        if name in thisSymbolTable:
            return thisSymbolTable[name]

        otherSymbolTable = SymbolTable(otherModule.operation)
        if name in otherSymbolTable:
            cloned = otherSymbolTable[name].operation.clone()
            currentModule.body.append(cloned)
            if 'cudaq-entrypoint' in cloned.operation.attributes:
                cloned.operation.attributes.__delitem__('cudaq-entrypoint')
            return cloned

        emitFatalError(f"Could not find function with name {name}")

    def __addAllCalledFunctionsRecursively(self, otherFunc, currentModule,
                                           otherModule):
        """
        Search the given `FuncOp` for all `CallOps` recursively.
        If found, see if the called function is in the current `ModuleOp`
        for this `kernel_builder`, if so do nothing. If it is not found,
        then find it in the other `ModuleOp`, clone it, and add it to this
        `ModuleOp`.
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
                        f"Invalid called function `{calleeName}`- cannot find the function in the symbol table"
                    )

                cloned = otherST[calleeName].operation.clone()
                if 'cudaq-entrypoint' in cloned.operation.attributes:
                    cloned.operation.attributes.__delitem__('cudaq-entrypoint')
                currentModule.body.append(cloned)

                visitAllCallOps(cloned)

            walk(funcOp, functor)

        visitAllCallOps(otherFunc)
        return

    def __applyControlOrAdjoint(self, target, isAdjoint, controls, *args):
        """
        Utility method for adding a Quake `ApplyOp` in the case of cudaq.control or 
        cudaq.adjoint. This function will search recursively for all required function 
        operations and add them tot he module. 
        """
        with self.insertPoint, self.loc:
            otherModule = Module.parse(str(target.module), self.ctx)

            otherFuncCloned = self.__cloneOrGetFunction(
                nvqppPrefix + target.name, self.module, otherModule)
            self.__addAllCalledFunctionsRecursively(otherFuncCloned,
                                                    self.module, otherModule)
            otherFTy = otherFuncCloned.body.blocks[0].arguments
            mlirValues = []
            for i, v in enumerate(args):
                argTy = otherFTy[i].type
                if not isinstance(v, QuakeValue):
                    # here we have to map constant Python data
                    # to an MLIR Value
                    value = self.__getMLIRValueFromPythonArg(v, argTy)

                else:
                    value = v.mlirValue
                inTy = value.type

                if (quake.VeqType.isinstance(inTy) and
                        quake.VeqType.isinstance(argTy)):
                    if quake.VeqType.getSize(
                            inTy) and not quake.VeqType.getSize(argTy):
                        value = quake.RelaxSizeOp(argTy, value).result

                mlirValues.append(value)
            if isAdjoint or len(controls) > 0:
                quake.ApplyOp([], [],
                              controls,
                              mlirValues,
                              callee=FlatSymbolRefAttr.get(
                                  otherFuncCloned.name.value),
                              is_adj=isAdjoint)
            else:
                func.CallOp(otherFuncCloned, mlirValues)

    def __str__(self, canonicalize=True):
        """
        Return a string representation of this kernels MLIR Module.
        """
        if canonicalize:
            pm = PassManager.parse("builtin.module(canonicalize,cse)",
                                   context=self.ctx)
            cloned = cudaq_runtime.cloneModule(self.module)
            pm.run(cloned)
            return str(cloned)
        return str(self.module)

    def qalloc(self, initializer=None):
        """
        Allocate a register of qubits of size `qubit_count` and return a 
        handle to them as a :class:`QuakeValue`.

        Args:
            initializer (Union[`int`,`QuakeValue`, `list[T]`): The number of qubits to allocate or a concrete state to allocate and initialize the qubits.
        Returns:
            :class:`QuakeValue`: A handle to the allocated qubits in the MLIR.

        ```python
            # Example:
            kernel = cudaq.make_kernel()
            qubits = kernel.qalloc(10)
        ```
        """
        with self.insertPoint, self.loc:
            # If the initializer is an integer, create `veq<N>`
            if isinstance(initializer, int):
                veqTy = quake.VeqType.get(self.ctx, initializer)
                return self.__createQuakeValue(quake.AllocaOp(veqTy).result)

            if isinstance(initializer, list):
                initializer = np.array(initializer, dtype=type(initializer[0]))

            if isinstance(initializer, np.ndarray):
                if len(initializer.shape) != 1:
                    raise RuntimeError(
                        "invalid initializer for qalloc (np.ndarray must be 1D, vector-like)"
                    )

                if initializer.dtype not in [
                        complex, np.complex128, np.complex64
                ]:
                    raise RuntimeError(
                        "qalloc state data must be of complex dtype.")

                # Get the current simulation precision
                currentTarget = cudaq_runtime.get_target()
                simulationPrecision = currentTarget.get_precision()
                if initializer.dtype in [np.complex128, complex]:
                    if simulationPrecision == cudaq_runtime.SimulationPrecision.fp32:
                        raise RuntimeError(
                            "qalloc input state is complex128 but simulator is on complex64 floating point type."
                        )

                if initializer.dtype == np.complex64:
                    if simulationPrecision == cudaq_runtime.SimulationPrecision.fp64:
                        raise RuntimeError(
                            "qalloc input state is complex64 but simulator is on complex128 floating point type."
                        )

                # Compute a unique hash string for the state data
                hashValue = hashlib.sha1(initializer).hexdigest(
                )[:10] + self.name.removeprefix('__nvqppBuilderKernel_')

                # Get the size of the array
                size = len(initializer)

                floatType = F64Type.get(
                    self.ctx
                ) if simulationPrecision == cudaq_runtime.SimulationPrecision.fp64 else F32Type.get(
                    self.ctx)
                complexType = ComplexType.get(floatType)
                ptrComplex = cc.PointerType.get(self.ctx, complexType)
                i32Ty = self.getIntegerType(32)
                globalTy = cc.StructType.get(self.ctx, [ptrComplex, i32Ty])
                globalName = f'nvqpp.state.{hashValue}'
                setStateName = f'nvqpp.set.state.{hashValue}'
                with InsertionPoint.at_block_begin(self.module.body):
                    cc.GlobalOp(TypeAttr.get(globalTy),
                                globalName,
                                external=True)
                    setStateFunc = func.FuncOp(setStateName,
                                               FunctionType.get(
                                                   inputs=[ptrComplex],
                                                   results=[]),
                                               loc=self.loc)
                    entry = setStateFunc.add_entry_block()
                    kDynamicPtrIndex: int = -2147483648
                    with InsertionPoint(entry):
                        zero = self.getConstantInt(0)
                        address = cc.AddressOfOp(
                            cc.PointerType.get(self.ctx, globalTy),
                            FlatSymbolRefAttr.get(globalName))
                        ptr = cc.ComputePtrOp(
                            cc.PointerType.get(self.ctx, ptrComplex), address,
                            [zero, zero],
                            DenseI32ArrayAttr.get(
                                [kDynamicPtrIndex, kDynamicPtrIndex],
                                context=self.ctx))
                        cc.StoreOp(entry.arguments[0], ptr)
                        func.ReturnOp([])

                zero = self.getConstantInt(0)
                numQubits = np.log2(size)
                if not numQubits.is_integer():
                    raise RuntimeError(
                        "invalid input state size for qalloc (not a power of 2)"
                    )

                # check state is normalized
                norm = sum([np.conj(a) * a for a in initializer])
                if np.abs(norm.imag) > 1e-4 or np.abs(1. - norm.real) > 1e-4:
                    raise RuntimeError(
                        "invalid input state for qalloc (not normalized)")

                veqTy = quake.VeqType.get(self.ctx, int(numQubits))
                qubits = quake.AllocaOp(veqTy).result
                address = cc.AddressOfOp(cc.PointerType.get(self.ctx, globalTy),
                                         FlatSymbolRefAttr.get(globalName))
                ptr = cc.ComputePtrOp(
                    cc.PointerType.get(self.ctx, ptrComplex), address,
                    [zero, zero],
                    DenseI32ArrayAttr.get([kDynamicPtrIndex, kDynamicPtrIndex],
                                          context=self.ctx))
                loaded = cc.LoadOp(ptr)
                qubits = quake.InitializeStateOp(qubits.type, qubits,
                                                 loaded).result

                # Record the unique hash value
                if hashValue not in self.stateHashes:
                    self.stateHashes.append(hashValue)

                # Store the pointer to the array data
                cudaq_runtime.storePointerToStateData(
                    self.name, hashValue, initializer,
                    cudaq_runtime.SimulationPrecision.fp64)

                return self.__createQuakeValue(qubits)

            # If the initializer is a QuakeValue, see if it is
            # a integer or a `stdvec` type
            if isinstance(initializer, QuakeValue):
                veqTy = quake.VeqType.get(self.ctx)
                if IntegerType.isinstance(initializer.mlirValue.type):
                    # This is an integer size
                    return self.__createQuakeValue(
                        quake.AllocaOp(veqTy,
                                       size=initializer.mlirValue).result)

                if cc.StdvecType.isinstance(initializer.mlirValue.type):
                    # This is a state to initialize to
                    size = cc.StdvecSizeOp(self.getIntegerType(),
                                           initializer.mlirValue).result
                    numQubits = math.CountTrailingZerosOp(size).result
                    qubits = quake.AllocaOp(veqTy, size=numQubits).result
                    ptrTy = cc.PointerType.get(
                        self.ctx,
                        cc.StdvecType.getElementType(
                            initializer.mlirValue.type))
                    initials = cc.StdvecDataOp(ptrTy, initializer.mlirValue)
                    quake.InitializeStateOp(veqTy, qubits, initials)
                    return self.__createQuakeValue(qubits)

            # If no initializer, create a single qubit
            if initializer == None:
                qubitTy = quake.RefType.get(self.ctx)
                return self.__createQuakeValue(quake.AllocaOp(qubitTy).result)

            raise RuntimeError("invalid initializer argument for qalloc.")

    def __isPauliWordType(self, ty):
        """
        A Pauli word type in our MLIR dialects is a `cc.charspan`. Return 
        True if the provided type is equivalent to this, False otherwise.
        """
        return cc.CharspanType.isinstance(ty)

    def exp_pauli(self, theta, *args):
        """
        Apply a general Pauli tensor product rotation, `exp(i theta P)`, on 
        the specified qubit register. The Pauli tensor product is provided 
        as a string, e.g. `XXYX` for a 4-qubit term. The angle parameter 
        can be provided as a concrete float or a `QuakeValue`.
        """
        with self.insertPoint, self.loc:
            quantumVal = None
            qubitsList = []
            pauliWordVal = None
            for arg in args:
                if isinstance(arg, cudaq_runtime.SpinOperator):
                    if arg.get_term_count() > 1:
                        emitFatalError(
                            'exp_pauli operation requires a SpinOperator composed of a single term.'
                        )
                    arg = arg.to_string(False)

                if isinstance(arg, str):
                    retTy = cc.PointerType.get(
                        self.ctx,
                        cc.ArrayType.get(self.ctx, IntegerType.get_signless(8),
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
                quantumVal = quake.ConcatOp(quake.VeqType.get(
                    self.ctx), [quantumVal] if quantumVal is not None else [] +
                                            qubitsList).result
            quake.ExpPauliOp(thetaVal, quantumVal, pauli=pauliWordVal)

    def givens_rotation(self, angle, qubitA, qubitB):
        """
        Add Givens rotation kernel (theta angle as a QuakeValue) to the
        kernel builder object
        """
        givens_builder(self, angle, qubitA, qubitB)

    def fermionic_swap(self, angle, qubitA, qubitB):
        """
        Add Fermionic SWAP rotation kernel (phi angle as a QuakeValue) to the
        kernel builder object
        """
        fermionic_swap_builder(self, angle, qubitA, qubitB)

    def from_state(self, qubits, state):
        emitFatalError("from_state not implemented.")

    def cswap(self, controls, qubitA, qubitB):
        """
        Controlled swap of the states of the provided qubits. 
        The controls parameter is expected to be a list of QuakeValue.

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
        with self.insertPoint, self.loc:
            if not quake.VeqType.isinstance(target.mlirValue.type):
                quake.ResetOp([], target.mlirValue)
                return

            # target is a VeqType
            size = quake.VeqType.getSize(target.mlirValue.type)
            if size:
                for i in range(size):
                    extracted = quake.ExtractRefOp(quake.RefType.get(self.ctx),
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
        `register_name` may be used to retrieve results of this measurement after 
        execution on the QPU. If the measurement call is saved as a variable, it will 
        return a :class:`QuakeValue` handle to the measurement instruction.

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
        with self.insertPoint, self.loc:
            i1Ty = IntegerType.get_signless(1, context=self.ctx)
            qubitTy = target.mlirValue.type
            retTy = i1Ty
            measTy = quake.MeasureType.get(self.ctx)
            stdvecTy = cc.StdvecType.get(self.ctx, i1Ty)
            if quake.VeqType.isinstance(target.mlirValue.type):
                retTy = stdvecTy
                measTy = cc.StdvecType.get(self.ctx, measTy)
            res = quake.MzOp(
                measTy, [], [target.mlirValue],
                registerName=StringAttr.get(regName, context=self.ctx)
                if regName is not None else '')
            disc = quake.DiscriminateOp(retTy, res)
            return self.__createQuakeValue(disc.result)

    def mx(self, target, regName=None):
        """
        Measure the given qubit or qubits in the X-basis. The optional 
        `register_name` may be used to retrieve results of this measurement after 
        execution on the QPU. If the measurement call is saved as a variable, it will 
        return a :class:`QuakeValue` handle to the measurement instruction.

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
        with self.insertPoint, self.loc:
            i1Ty = IntegerType.get_signless(1, context=self.ctx)
            qubitTy = target.mlirValue.type
            retTy = i1Ty
            measTy = quake.MeasureType.get(self.ctx)
            stdvecTy = cc.StdvecType.get(self.ctx, i1Ty)
            if quake.VeqType.isinstance(target.mlirValue.type):
                retTy = stdvecTy
                measTy = cc.StdvecType.get(self.ctx, measTy)
            res = quake.MxOp(
                measTy, [], [target.mlirValue],
                registerName=StringAttr.get(regName, context=self.ctx)
                if regName is not None else '')
            disc = quake.DiscriminateOp(retTy, res)
            return self.__createQuakeValue(disc.result)

    def my(self, target, regName=None):
        """
        Measure the given qubit or qubits in the Y-basis. The optional 
        `register_name` may be used to retrieve results of this measurement after 
        execution on the QPU. If the measurement call is saved as a variable, it will
        return a :class:`QuakeValue` handle to the measurement instruction.

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
        with self.insertPoint, self.loc:
            i1Ty = IntegerType.get_signless(1, context=self.ctx)
            qubitTy = target.mlirValue.type
            retTy = i1Ty
            measTy = quake.MeasureType.get(self.ctx)
            stdvecTy = cc.StdvecType.get(self.ctx, i1Ty)
            if quake.VeqType.isinstance(target.mlirValue.type):
                retTy = stdvecTy
                measTy = cc.StdvecType.get(self.ctx, measTy)
            res = quake.MyOp(
                measTy, [], [target.mlirValue],
                registerName=StringAttr.get(regName, context=self.ctx)
                if regName is not None else '')
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
        RuntimeError: if the `*target_arguments` passed to the adjoint call don't 
            match the argument signature of `target`.

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
        Apply the `target` kernel as a controlled operation in-place to 
        `self`.Uses the provided `control` as control qubit/s for the operation.

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
        Apply a call to the given `target` kernel within the function-body 
        of `self` at the provided target arguments.

        Args:
        target (:class:`Kernel`): The kernel to call from within `self`.
        *target_arguments (Optional[:class:`QuakeValue`]): The arguments to the `target` kernel. 
            Leave empty if the `target` kernel doesn't accept any arguments.

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
        if isinstance(target, PyKernelDecorator):
            target.compile()
        self.__applyControlOrAdjoint(target, False, [], *target_arguments)

    def c_if(self, measurement, function):
        """
        Apply the `function` to the :class:`Kernel` if the provided 
        single-qubit `measurement` returns the 1-state. 

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
            # The register names in the conditional tests need to be double checked;
            # The code here may need to be adjusted to reflect the additional
            # quake.discriminate conversion of the measurement.
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
                conditional = arith.CmpIOp(condPred, condition,
                                           self.getConstantInt(0)).result

            ifOp = cc.IfOp([], conditional)
            thenBlock = Block.create_at_start(ifOp.thenRegion, [])
            with InsertionPoint(thenBlock):
                tmpIp = self.insertPoint
                self.insertPoint = InsertionPoint(thenBlock)
                function()
                self.insertPoint = tmpIp
                cc.ContinueOp([])
            self.metadata['conditionalOnMeasure'] = True

    def for_loop(self, start, stop, function):
        """Add a for loop that starts from the given `start` index, 
        ends at the given `stop` index (non inclusive), applying the 
        provided `function` within `self` at each iteration. The step 
        value is provided to mutate the iteration variable after every iteration.

        Args:
        start (int or :class:`QuakeValue`): The beginning iterator value for the for loop.
        stop (int or :class:`QuakeValue`): The final iterator value (non-inclusive) for the for loop.
        function (Callable): The callable function to apply within the `kernel` at
            each iteration.

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

    def __call__(self, *args):
        """Just-In-Time (JIT) compile `self` (:class:`Kernel`), and call 
        the kernel function at the provided concrete arguments.

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
        if len(args) != len(self.mlirArgTypes):
            emitFatalError(
                f"Invalid number of arguments passed to kernel `{self.funcName}` ({len(args)} provided, {len(self.mlirArgTypes)} required"
            )

        def getListType(eleType: type):
            if sys.version_info < (3, 9):
                return List[eleType]
            else:
                return list[eleType]

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
                listType = getListType(type(arg[0]))
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

                    # Support passing `list[float]` to a `list[complex]` argument
                    maybeCasted = supportCommonCast(mlirType,
                                                    self.mlirArgTypes[i], arg,
                                                    F64Type, ComplexType,
                                                    complex)
                    if maybeCasted != None:
                        processedArgs.append(maybeCasted)
                        continue

            if mlirType != self.mlirArgTypes[
                    i] and listType != mlirTypeToPyType(self.mlirArgTypes[i]):
                emitFatalError(
                    f"Invalid runtime argument type ({type(arg)} provided, {mlirTypeToPyType(self.mlirArgTypes[i])} required)"
                )

            # Convert `numpy` arrays to lists
            if cc.StdvecType.isinstance(mlirType):
                # Validate that the length of this argument is
                # greater than or equal to the number of unique
                # quake value extractions
                if len(arg) < len(self.arguments[i].knownUniqueExtractions):
                    emitFatalError(
                        f"Invalid runtime list argument - {len(arg)} elements in list but kernel code has at least {len(self.arguments[i].knownUniqueExtractions)} known unique extractions."
                    )
                if hasattr(arg, "tolist"):
                    processedArgs.append(arg.tolist())
                else:
                    processedArgs.append(arg)
            else:
                processedArgs.append(arg)

        cudaq_runtime.pyAltLaunchKernel(self.name, self.module, *processedArgs)


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
