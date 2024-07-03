# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import uuid
import numpy as np

from ..mlir.ir import *
from ..mlir.passmanager import *
from ..mlir.execution_engine import *
from ..mlir.dialects import quake, cc
from ..mlir.dialects import builtin, func, arith
from ..mlir._mlir_libs._quakeDialects import cudaq_runtime

kDynamicPtrIndex: int = -2147483648


class CapturedDataStorage(object):
    """
    Captured data storage is used to store and load captured
    arrays and cudaq states so we don't have to copy them in
    MLIR.
    """

    def __init__(self, **kwargs):
        self.cudaqStateIDs = kwargs[
            'cudaqStateIDs'] if 'cudaqStateIDs' in kwargs else []
        self.arrayIDs = kwargs['arrayIDs'] if 'arrayIDs' in kwargs else []

        self.ctx = kwargs['ctx'] if 'ctx' in kwargs else None
        self.loc = kwargs['loc'] if 'loc' in kwargs else None
        self.name = kwargs['name'] if 'name' in kwargs else None
        self.module = kwargs['module'] if 'module' in kwargs else None

    def setKernelContext(self, ctx, loc, name, module):
        self.ctx = ctx
        self.loc = loc
        self.name = name
        self.module = module

    def __del__(self):
        """
        Remove pointers to stored data for the current kernel.
        """
        cudaq_runtime.deletePointersToCudaqState(self.cudaqStateIDs)
        cudaq_runtime.deletePointersToStateData(self.arrayIDs)

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

    def storeCudaqState(self, value: cudaq_runtime.State):
        # Compute a unique ID for the state data
        stateID = self.name + str(uuid.uuid4())
        stateTy = cc.StateType.get(self.ctx)
        statePtrTy = cc.PointerType.get(self.ctx, stateTy)

        # Generate a function that stores the state value in a global
        globalTy = statePtrTy
        globalName = f'nvqpp.cudaq.state.{stateID}'
        setStateName = f'nvqpp.set.cudaq.state.{stateID}'
        with InsertionPoint.at_block_begin(self.module.body):
            cc.GlobalOp(TypeAttr.get(globalTy), globalName, external=True)
            setStateFunc = func.FuncOp(setStateName,
                                       FunctionType.get(inputs=[statePtrTy],
                                                        results=[]),
                                       loc=self.loc)
            entry = setStateFunc.add_entry_block()
            with InsertionPoint(entry):
                zero = self.getConstantInt(0)
                address = cc.AddressOfOp(cc.PointerType.get(self.ctx, globalTy),
                                         FlatSymbolRefAttr.get(globalName))
                ptr = cc.CastOp(cc.PointerType.get(self.ctx, statePtrTy),
                                address)

                cc.StoreOp(entry.arguments[0], ptr)
                func.ReturnOp([])

        # Record the unique hash value
        if stateID not in self.cudaqStateIDs:
            self.cudaqStateIDs.append(stateID)

        # Store the state into a global variable
        cudaq_runtime.storePointerToCudaqState(self.name, stateID, value)

        # Return the pointer to the stored state
        zero = self.getConstantInt(0)
        address = cc.AddressOfOp(cc.PointerType.get(self.ctx, globalTy),
                                 FlatSymbolRefAttr.get(globalName)).result
        ptr = cc.CastOp(cc.PointerType.get(self.ctx, statePtrTy),
                        address).result
        statePtr = cc.LoadOp(ptr).result
        return statePtr

    def storeArray(self, array: np.ndarray):
        # Compute a unique hash string for the array data
        arrayId = self.name + str(uuid.uuid4())

        # Get the current simulation precision
        currentTarget = cudaq_runtime.get_target()
        simulationPrecision = currentTarget.get_precision()

        floatType = F64Type.get(
            self.ctx
        ) if simulationPrecision == cudaq_runtime.SimulationPrecision.fp64 else F32Type.get(
            self.ctx)
        complexType = ComplexType.get(floatType)
        ptrComplex = cc.PointerType.get(self.ctx, complexType)
        i32Ty = self.getIntegerType(32)
        globalTy = cc.StructType.get(self.ctx, [ptrComplex, i32Ty])
        globalName = f'nvqpp.state.{arrayId}'
        setStateName = f'nvqpp.set.state.{arrayId}'
        with InsertionPoint.at_block_begin(self.module.body):
            cc.GlobalOp(TypeAttr.get(globalTy), globalName, external=True)
            setStateFunc = func.FuncOp(setStateName,
                                       FunctionType.get(inputs=[ptrComplex],
                                                        results=[]),
                                       loc=self.loc)
            entry = setStateFunc.add_entry_block()
            with InsertionPoint(entry):
                zero = self.getConstantInt(0)
                address = cc.AddressOfOp(cc.PointerType.get(self.ctx, globalTy),
                                         FlatSymbolRefAttr.get(globalName))
                ptr = cc.CastOp(cc.PointerType.get(self.ctx, ptrComplex),
                                address)
                cc.StoreOp(entry.arguments[0], ptr)
                func.ReturnOp([])

        zero = self.getConstantInt(0)

        address = cc.AddressOfOp(cc.PointerType.get(self.ctx, globalTy),
                                 FlatSymbolRefAttr.get(globalName))
        ptr = cc.CastOp(cc.PointerType.get(self.ctx, ptrComplex), address)

        # Record the unique hash value
        if arrayId not in self.arrayIDs:
            self.arrayIDs.append(arrayId)

        # Store the pointer to the array data
        cudaq_runtime.storePointerToStateData(
            self.name, arrayId, array, cudaq_runtime.SimulationPrecision.fp64)

        return cc.LoadOp(ptr).result
