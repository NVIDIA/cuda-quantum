# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import ast
import importlib
import graphlib
import textwrap
import numpy as np
import os
import sys
from collections import deque
from types import FunctionType

from cudaq.mlir._mlir_libs._quakeDialects import (
    cudaq_runtime, load_intrinsic, gen_vector_of_complex_constant,
    register_all_dialects)
from cudaq.mlir.dialects import arith, cc, complex, func, math, quake
from cudaq.mlir.ir import (BoolAttr, Block, BlockArgument, Context, ComplexType,
                           DenseBoolArrayAttr, DenseI32ArrayAttr,
                           DenseI64ArrayAttr, DictAttr, F32Type, F64Type,
                           FlatSymbolRefAttr, FloatAttr, FunctionType,
                           InsertionPoint, IntegerAttr, IntegerType, Location,
                           Module, StringAttr, SymbolTable, TypeAttr, UnitAttr)
from cudaq.mlir.passmanager import PassManager
from .analysis import FindDepKernelsVisitor, ValidateArgumentAnnotations, ValidateReturnStatements
from .captured_data import CapturedDataStorage
from .utils import (Color, globalAstRegistry, globalKernelRegistry,
                    globalRegisteredOperations, globalRegisteredTypes,
                    nvqppPrefix, mlirTypeFromAnnotation, mlirTypeFromPyType,
                    mlirTypeToPyType, mlirTryCreateStructType)

State = cudaq_runtime.State

# This file implements the CUDA-Q Python AST to MLIR conversion.
# It provides a `PyASTBridge` class that implements the `ast.NodeVisitor` type
# to walk the Python AST for a `cudaq.kernel` annotated function and generate
# valid MLIR code using `Quake`, `CC`, `Arith`, and `Math` dialects.

# CC Dialect `ComputePtrOp` in C++ sets the
# dynamic index as `std::numeric_limits<int32_t>::min()`
# (see CCOps.tc line 898). We'll duplicate that
# here by just setting it manually
kDynamicPtrIndex: int = -2147483648

ALLOWED_TYPES_IN_A_DATACLASS = [int, float, bool, cudaq_runtime.qview]


class PyScopedSymbolTable(object):

    def __init__(self):
        self.symbolTable = deque()

    def pushScope(self):
        self.symbolTable.append({})

    def popScope(self):
        self.symbolTable.pop()

    def numLevels(self):
        return len(self.symbolTable)

    def add(self, symbol, value, level=-1):
        """
        Add a symbol to the scoped symbol table at any scope level.
        """
        self.symbolTable[level][symbol] = value

    def __contains__(self, symbol):
        for st in reversed(self.symbolTable):
            if symbol in st:
                return True

        return False

    def __setitem__(self, symbol, value):
        # default to nearest surrounding scope
        self.add(symbol, value)
        return

    def __getitem__(self, symbol):
        for st in reversed(self.symbolTable):
            if symbol in st:
                return st[symbol]

        raise RuntimeError(
            f"{symbol} is not a valid variable name in this scope.")

    def clear(self):
        while len(self.symbolTable):
            self.symbolTable.pop()
        return


class CompilerError(RuntimeError):
    """
    Custom exception class for improved error diagnostics.
    """

    def __init__(self, *args, **kwargs):
        RuntimeError.__init__(self, *args, **kwargs)


class PyStack(object):
    '''
    Takes care of managing values produced while vising Python
    AST nodes. Each visit to a node is expected to match one
    stack frame. Values produced (meaning pushed) by child frames
    are accessible (meaning can be popped) by the parent. A frame
    cannot access the value it produced (it is owned by the parent).
    '''
    class Frame(object):
        def __init__(self, parent = None):
            self.entries = None
            self.parent = parent

    def __init__(self, error_handler = None):
        def default_error_handler(msg):
            raise RuntimeError(msg)
        self._frame = None
        self.emitError = error_handler or default_error_handler

    def pushFrame(self):
        '''
        A new frame should be pushed to process a new node in the AST.
        '''
        if self._frame and not self._frame.entries:
            self._frame.entries = deque()
        self._frame = PyStack.Frame(parent = self._frame)

    def popFrame(self):
        '''
        A frame should be popped once a node in the AST has been processed.
        '''
        if not self._frame:
            self.emitError("stack has no frames to pop")
        elif self._frame.entries:
            self.emitError("all values must be processed before popping a frame")
        else:
            self._frame = self._frame.parent

    def pushValue(self, value):
        '''
        Pushes a value to the make it available to the parent frame.
        '''
        if not self._frame:
            self.emitError("cannot push value to empty stack")
        elif not self._frame.parent:
            self.emitError("no parent frame is defined to push values to")
        else:
            self._frame.parent.entries.append(value)

    def popValue(self):
        '''
        Pops the most recently produced (pushed) value by a child frame.
        '''
        if not self._frame:
            self.emitError("value stack is empty")
        elif not self._frame.entries:
            # This is the only error that may be directly user-facing even when
            # the bridge is doing its processing correctly.
            # We hence give a somewhat general error.
            # For internal purposes, the error might be better stated as something like:
            # either this frame has not had a child or the child did not produce any values
            self.emitError("no valid value was created")
        else:
            return self._frame.entries.pop()

    @property
    def isEmpty(self):
        '''
        Returns true if and only if there are no remaining stack frames.
        '''
        return not self._frame
    
    @property
    def currentNumValues(self):
        '''
        Returns the number of values that are accessible for processing by the current frame.
        '''
        if not self._frame:
            self.emitError("no frame defined for empty stack")
        elif self._frame.entries: 
            return len(self._frame.entries)
        return 0


class PyASTBridge(ast.NodeVisitor):
    """
    The `PyASTBridge` class implements the `ast.NodeVisitor` type to convert a 
    python function definition (annotated with cudaq.kernel) to an MLIR `ModuleOp`
    containing a `func.FuncOp` representative of the original python function but leveraging 
    the Quake and CC dialects provided by CUDA-Q. This class keeps track of a 
    MLIR Value stack that is pushed to and popped from during visitation of the 
    function AST nodes. We leverage the auto-generated MLIR Python bindings for the internal 
    C++ CUDA-Q dialects to build up the MLIR code. 

    For kernels that call other kernels, we require that the `ModuleOp` contain the 
    kernel being called. This is enabled via the `FindDepKernelsVisitor` in the local 
    analysis module, and is handled by the below `compile_to_mlir` function. For 
    callable block arguments, we leverage runtime-known callable argument function names 
    and synthesize them away with an internal C++ MLIR pass. 
    """

    def __init__(self, capturedDataStorage: CapturedDataStorage, **kwargs):
        """
        The constructor. Initializes the `mlir.Value` stack, the `mlir.Context`, and the 
        `mlir.Module` that we will be building upon. This class keeps track of a 
        symbol table, which maps variable names to constructed `mlir.Values`. 
        """
        self.valueStack = PyStack(lambda msg: self.emitFatalError(f'processing error - {msg}', self.currentNode))
        self.knownResultType = kwargs[
            'knownResultType'] if 'knownResultType' in kwargs else None
        if 'existingModule' in kwargs:
            self.module = kwargs['existingModule']
            self.ctx = self.module.context
            self.loc = Location.unknown(context=self.ctx)
        else:
            self.ctx = Context()
            register_all_dialects(self.ctx)
            quake.register_dialect(context=self.ctx)
            cc.register_dialect(context=self.ctx)
            cudaq_runtime.registerLLVMDialectTranslation(self.ctx)
            self.loc = Location.unknown(context=self.ctx)
            self.module = Module.create(loc=self.loc)

        # Create a new captured data storage or use the existing one
        # passed from the current kernel decorator.
        self.capturedDataStorage = capturedDataStorage
        if (self.capturedDataStorage == None):
            self.capturedDataStorage = CapturedDataStorage(ctx=self.ctx,
                                                           loc=self.loc,
                                                           name=None,
                                                           module=self.module)
        else:
            self.capturedDataStorage.setKernelContext(ctx=self.ctx,
                                                      loc=self.loc,
                                                      name=None,
                                                      module=self.module)

        # If the driver of this AST bridge instance has indicated
        # that there is a return type from analysis on the Python AST,
        # then we want to set the known result type so that the
        # FuncOp can have it.
        if 'returnTypeIsFromPython' in kwargs and kwargs[
                'returnTypeIsFromPython'] and self.knownResultType is not None:
            self.knownResultType = mlirTypeFromPyType(self.knownResultType,
                                                      self.ctx)

        self.capturedVars = kwargs[
            'capturedVariables'] if 'capturedVariables' in kwargs else {}
        self.dependentCaptureVars = {}
        self.locationOffset = kwargs[
            'locationOffset'] if 'locationOffset' in kwargs else ('', 0)
        self.disableEntryPointTag = kwargs[
            'disableEntryPointTag'] if 'disableEntryPointTag' in kwargs else False
        self.disableNvqppPrefix = kwargs[
            'disableNvqppPrefix'] if 'disableNvqppPrefix' in kwargs else False
        self.symbolTable = PyScopedSymbolTable()
        self.indent_level = 0
        self.indent = 4 * " "
        self.buildingEntryPoint = False
        self.inForBodyStack = deque()
        self.inIfStmtBlockStack = deque()
        self.currentAssignVariableName = None
        self.walkingReturnNode = False
        self.controlNegations = []
        self.pushPointerValue = False
        self.verbose = 'verbose' in kwargs and kwargs['verbose']
        self.currentNode = None

    def debug_msg(self, msg, node=None):
        if self.verbose:
            print(f'{self.indent * self.indent_level}{msg()}')
            if node is not None:
                print(
                    textwrap.indent(ast.unparse(node),
                                    (self.indent * (self.indent_level + 1))))

    def emitWarning(self, msg, astNode=None):
        """
        Emit a warning, providing the user with source file information and
        the offending code.
        """
        codeFile = os.path.basename(self.locationOffset[0])
        if astNode == None:
            astNode = self.currentNode
        lineNumber = '' if astNode == None else astNode.lineno + self.locationOffset[
            1] - 1

        print(Color.BOLD, end='')
        msg = codeFile + ":" + str(
            lineNumber
        ) + ": " + Color.YELLOW + "warning: " + Color.END + Color.BOLD + msg + (
            "\n\t (offending source -> " + ast.unparse(astNode) + ")" if
            hasattr(ast, 'unparse') and astNode is not None else '') + Color.END
        print(msg)

    def emitFatalError(self, msg, astNode=None):
        """
        Emit a fatal error, providing the user with source file information and
        the offending code.
        """
        codeFile = os.path.basename(self.locationOffset[0])
        if astNode == None:
            astNode = self.currentNode
        lineNumber = '' if astNode == None or not hasattr(astNode, 'lineno') else astNode.lineno + self.locationOffset[
            1] - 1
        
        try:
            offending_source = "\n\t (offending source -> " + ast.unparse(astNode) + ")"
        except:
            offending_source = ''

        print(Color.BOLD, end='')
        msg = codeFile + ":" + str(
            lineNumber
        ) + ": " + Color.RED + "error: " + Color.END + Color.BOLD + msg + offending_source + Color.END
        raise CompilerError(msg)

    def getVeqType(self, size=None):
        """
        Return a `quake.VeqType`. Pass the size of the `quake.veq` if known. 
        """
        if size == None:
            return quake.VeqType.get()
        return quake.VeqType.get(size)

    def getRefType(self):
        """
        Return a `quake.RefType`.
        """
        return quake.RefType.get()

    def isQuantumType(self, ty):
        """
        Return True if the given type is quantum (is a `VeqType` or `RefType`). 
        Return False otherwise.
        """
        return quake.RefType.isinstance(ty) or quake.VeqType.isinstance(
            ty) or quake.StruqType.isinstance(ty)

    # FIXME: Needs to be revised when we introduce the proper type distinction
    # between boolean and measurements.
    def isMeasureResultType(self, ty, value):
        """
        Return true if the given type is a qubit measurement result type (an i1
        type).
        """
        if hasattr(value, 'owner') and hasattr(
                value.owner,
                'name') and not 'quake.discriminate' == value.owner.name:
            return False
        return IntegerType.isinstance(ty) and ty == IntegerType.get_signless(1)

    def getIntegerType(self, width=64):
        """
        Return an MLIR `IntegerType` of the given bit width (defaults to 64
        bits).
        """
        return IntegerType.get_signless(width)

    def getIntegerAttr(self, type, value):
        """
        Return an MLIR Integer Attribute of the given `IntegerType`.
        """
        return IntegerAttr.get(type, value)

    def getFloatType(self, width=64):
        """
        Return an MLIR float type (single or double precision).
        """
        # Note:
        # `numpy.float64` is the same as `float` type, with width of 64 bit.
        # `numpy.float32` type has width of 32 bit.
        if width == 64:
            return F64Type.get()
        elif width == 32:
            return F32Type.get()
        else:
            self.emitFatalError(
                f'unsupported width {width} requested for float type',
                self.currentNode)

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

    def getConstantComplex(self, value, width=64):
        """
        Create a constant complex operation and return its MLIR result Value.
        Takes as input the concrete complex value.
        """
        ty = self.getComplexType(width=width)
        return complex.CreateOp(ty,
                                self.getConstantFloat(value.real, width=width),
                                self.getConstantFloat(value.imag,
                                                      width=width)).result

    def getConstantComplexWithElementType(self, value, eTy):
        """
        Create a constant complex operation and return its MLIR result Value.
        Takes as input the concrete complex value.
        """
        ty = self.getComplexTypeWithElementType(eTy)
        return complex.CreateOp(ty,
                                self.getConstantFloatWithType(value.real, eTy),
                                self.getConstantFloatWithType(value.imag,
                                                              eTy)).result

    def getConstantInt(self, value, width=64):
        """
        Create a constant integer operation and return its MLIR result Value.
        Takes as input the concrete integer value. Can specify the integer bit
        width.
        """
        ty = self.getIntegerType(width)
        return arith.ConstantOp(ty, self.getIntegerAttr(ty, value)).result

    def __arithmetic_to_bool(self, value):
        """
        Converts an integer or floating point value to a bool by 
        comparing it to zero.
        """
        if self.getIntegerType(1) == value.type:
            return value
        if IntegerType.isinstance(value.type):
            zero = self.getConstantInt(0, width=IntegerType(value.type).width)
            condPred = IntegerAttr.get(self.getIntegerType(), 1)
            return arith.CmpIOp(condPred, value, zero).result
        elif F32Type.isinstance(value.type):
            zero = self.getConstantFloat(0, width=32)
            condPred = IntegerAttr.get(self.getIntegerType(), 13)
            return arith.CmpFOp(condPred, value, zero).result
        elif F64Type.isinstance(value.type):
            zero = self.getConstantFloat(0, width=64)
            condPred = IntegerAttr.get(self.getIntegerType(), 13)
            return arith.CmpFOp(condPred, value, zero).result
        else:
            self.emitFatalError("value cannot be converted to bool",
                                self.currentNode)

    def changeOperandToType(self, ty, operand, allowDemotion=False):
        """
        Change the type of an operand to a specified type. This function primarily 
        handles type conversions and promotions to higher types (complex > float > int). 
        Demotion of floating type to integer is not allowed by default.
        Regardless of whether demotion is allowed, types will be cast to smaller widths.
        """
        if ty == operand.type:
            return operand

        if ComplexType.isinstance(ty):
            complexType = ComplexType(ty)
            floatType = complexType.element_type
            if ComplexType.isinstance(operand.type):
                otherComplexType = ComplexType(operand.type)
                otherFloatType = otherComplexType.element_type
                if (floatType != otherFloatType):
                    real = self.changeOperandToType(
                        floatType,
                        complex.ReOp(operand).result)
                    imag = self.changeOperandToType(
                        floatType,
                        complex.ImOp(operand).result)
                    return complex.CreateOp(complexType, real, imag).result
            else:
                real = self.changeOperandToType(floatType, operand)
                imag = self.getConstantFloatWithType(0.0, floatType)
                return complex.CreateOp(complexType, real, imag).result

        if (cc.StdvecType.isinstance(ty)):
            eleTy = cc.StdvecType.getElementType(ty)
            if cc.StdvecType.isinstance(operand.type):
                return self.__copyVectorAndCastElements(
                    operand, eleTy, allowDemotion=allowDemotion)

        if F64Type.isinstance(ty):
            if F32Type.isinstance(operand.type):
                return cc.CastOp(ty, operand).result
            if IntegerType.isinstance(operand.type):
                zeroext = IntegerType(operand.type).width == 1
                return cc.CastOp(ty, operand, sint=not zeroext,
                                 zint=zeroext).result

        if F32Type.isinstance(ty):
            if F64Type.isinstance(operand.type):
                return cc.CastOp(ty, operand).result
            if IntegerType.isinstance(operand.type):
                zeroext = IntegerType(operand.type).width == 1
                return cc.CastOp(ty, operand, sint=not zeroext,
                                 zint=zeroext).result

        if IntegerType.isinstance(ty):
            if allowDemotion and (F64Type.isinstance(operand.type) or
                                  F32Type.isinstance(operand.type)):
                operand = cc.CastOp(ty, operand, sint=True, zint=False).result
            if IntegerType.isinstance(operand.type):
                requested_width = IntegerType(ty).width
                operand_width = IntegerType(operand.type).width
                if requested_width == operand_width:
                    return operand
                elif requested_width < operand_width:
                    if requested_width == 1:
                        return self.__arithmetic_to_bool(operand)
                    return cc.CastOp(ty, operand).result
                return cc.CastOp(ty,
                                 operand,
                                 sint=operand_width != 1,
                                 zint=operand_width == 1).result
        self.emitFatalError(
            f'cannot convert value of type {operand.type} to the requested type {ty}',
            self.currentNode)

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

    def pushValue(self, value):
        """
        Push an MLIR Value onto the stack for usage in a subsequent AST node
        visit method.
        """
        self.debug_msg(lambda: f'push {value}')
        self.valueStack.pushValue(value)

    def popValue(self):
        """
        Pop an MLIR Value from the stack. 
        """
        val = self.valueStack.popValue()
        self.debug_msg(lambda: f'pop {val}')
        return val
    
    def popAllValues(self, expectedNumVals):
        values = [self.popValue() for _ in range(self.valueStack.currentNumValues)]
        if len(values) != expectedNumVals:
            self.emitFatalError("processing error - no valid value was created", self.currentNode)
        return values

    def pushForBodyStack(self, bodyBlockArgs):
        """
        Indicate that we are entering a for loop body block. 
        """
        self.inForBodyStack.append(bodyBlockArgs)

    def popForBodyStack(self):
        """
        Indicate that we have left a for loop body block.
        """
        self.inForBodyStack.pop()

    def pushIfStmtBlockStack(self):
        """
        Indicate that we are entering an if statement then or else block.
        """
        self.inIfStmtBlockStack.append(0)

    def popIfStmtBlockStack(self):
        """
        Indicate that we have just left an if statement then or else block.
        """
        self.inIfStmtBlockStack.pop()

    def isInForBody(self):
        """
        Return True if the current insertion point is within a for body block. 
        """
        return len(self.inForBodyStack) > 0

    def isInIfStmtBlock(self):
        """
        Return True if the current insertion point is within an if statement
        then or else block.
        """
        return len(self.inIfStmtBlockStack) > 0

    def hasTerminator(self, block):
        """
        Return True if the given Block has a Terminator operation.
        """
        if len(block.operations) > 0:
            return cudaq_runtime.isTerminator(
                block.operations[len(block.operations) - 1])
        return False

    def isArithmeticType(self, type):
        """
        Return True if the given type is an integer, float, or complex type. 
        """
        return IntegerType.isinstance(type) or F64Type.isinstance(
            type) or F32Type.isinstance(type) or ComplexType.isinstance(type)

    def __isSupportedNumpyFunction(self, id):
        return id in ['sin', 'cos', 'sqrt', 'ceil', 'exp']

    def __isSupportedVectorFunction(self, id):
        return id in ['front', 'back', 'append']

    def __isSimpleGate(self, id):
        return id in ['h', 'x', 'y', 'z', 's', 't']

    def __isAdjointSimpleGate(self, id):
        return id in ['sdg', 'tdg']

    def __isControlledSimpleGate(self, id):
        if id == '' or id[0] != 'c':
            return False
        return self.__isSimpleGate(id[1:])

    def __isRotationGate(self, id):
        return id in ['rx', 'ry', 'rz', 'r1']

    def __isControlledRotationGate(self, id):
        if id == '' or id[0] != 'c':
            return False
        return self.__isRotationGate(id[1:])

    def __isMeasurementGate(self, id):
        return id in ['mx', 'my', 'mz']

    def __isUnitaryGate(self, id):
        return self.__isSimpleGate(id) or \
            self.__isRotationGate(id) or \
            self.__isAdjointSimpleGate(id) or \
            self.__isControlledSimpleGate(id) or \
            self.__isControlledRotationGate(id) or \
            id in ['swap', 'u3', 'exp_pauli'] or \
            id in globalRegisteredOperations

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
            slot = cc.AllocaOp(cc.PointerType.get(value.type),
                               TypeAttr.get(value.type)).result
            cc.StoreOp(value, slot)
            return slot
        return value

    def __createStdvecWithKnownValues(self, listElementValues):

        assert(len(set((v.type for v in listElementValues))) == 1)
        arrSize = self.getConstantInt(len(listElementValues))
        iTy = listElementValues[0].type
        alloca = cc.AllocaOp(cc.PointerType.get(cc.ArrayType.get(iTy)),
                             TypeAttr.get(iTy),
                             seqSize=arrSize).result

        for i, v in enumerate(listElementValues):
            eleAddr = cc.ComputePtrOp(
                cc.PointerType.get(iTy), alloca,
                [self.getConstantInt(i)],
                DenseI32ArrayAttr.get([kDynamicPtrIndex],
                                      context=self.ctx)).result
            cc.StoreOp(v, eleAddr)

        return cc.StdvecInitOp(cc.StdvecType.get(iTy), alloca,
                               length=arrSize).result

    def getStructMemberIdx(self, memberName, structTy):
        """
        For the given struct type and member variable name, return the index of
        the variable in the struct and the specific MLIR type for the variable.
        """
        if cc.StructType.isinstance(structTy):
            structName = cc.StructType.getName(structTy)
        else:
            structName = quake.StruqType.getName(structTy)
        structIdx = None
        if not globalRegisteredTypes.isRegisteredClass(structName):
            self.emitFatalError(f'Dataclass is not registered: {structName})')

        _, userType = globalRegisteredTypes.getClassAttributes(structName)
        for i, (k, _) in enumerate(userType.items()):
            if k == memberName:
                structIdx = i
                break
        if structIdx == None:
            self.emitFatalError(
                f'Invalid struct member: {structName}.{memberName} (members={[k for k,_ in userType.items()]})'
            )
        return structIdx, mlirTypeFromPyType(userType[memberName], self.ctx)

    # Create a new vector with source elements converted to the target element type if needed.
    def __copyVectorAndCastElements(self,
                                    source,
                                    targetEleType,
                                    allowDemotion=False):
        if not cc.PointerType.isinstance(source.type):
            if cc.StdvecType.isinstance(source.type):
                # Exit early if no copy is needed to avoid an unneeded store.
                sourceEleType = cc.StdvecType.getElementType(source.type)
                if (sourceEleType == targetEleType):
                    return source

        sourcePtr = source
        if not cc.PointerType.isinstance(sourcePtr.type):
            sourcePtr = self.ifNotPointerThenStore(sourcePtr)

        sourceType = cc.PointerType.getElementType(sourcePtr.type)
        if not cc.StdvecType.isinstance(sourceType):
            raise RuntimeError(
                f"expected vector type to copy and cast elements but received {sourceType}"
            )

        sourceEleType = cc.StdvecType.getElementType(sourceType)
        if (sourceEleType == targetEleType):
            return sourcePtr

        sourceArrType = cc.ArrayType.get(sourceEleType)
        sourceElePtrTy = cc.PointerType.get(sourceEleType)
        sourceArrElePtrTy = cc.PointerType.get(sourceArrType)
        sourceValue = self.ifPointerThenLoad(sourcePtr)
        sourceDataPtr = cc.StdvecDataOp(sourceArrElePtrTy, sourceValue).result
        sourceSize = cc.StdvecSizeOp(self.getIntegerType(), sourceValue).result

        targetElePtrType = cc.PointerType.get(targetEleType)
        targetTy = cc.ArrayType.get(targetEleType)
        targetArrElePtrTy = cc.PointerType.get(targetTy)
        targetVecTy = cc.StdvecType.get(targetEleType)
        targetPtr = cc.AllocaOp(targetArrElePtrTy,
                                TypeAttr.get(targetEleType),
                                seqSize=sourceSize).result

        rawIndex = DenseI32ArrayAttr.get([kDynamicPtrIndex], context=self.ctx)

        def bodyBuilder(iterVar):
            eleAddr = cc.ComputePtrOp(sourceElePtrTy, sourceDataPtr, [iterVar],
                                      rawIndex).result
            loadedEle = cc.LoadOp(eleAddr).result
            castedEle = self.changeOperandToType(targetEleType,
                                                 loadedEle,
                                                 allowDemotion=allowDemotion)
            targetEleAddr = cc.ComputePtrOp(targetElePtrType, targetPtr,
                                            [iterVar], rawIndex).result
            cc.StoreOp(castedEle, targetEleAddr)

        self.createInvariantForLoop(bodyBuilder, sourceSize)
        return cc.StdvecInitOp(targetVecTy, targetPtr, length=sourceSize).result

    def __insertDbgStmt(self, value, dbgStmt):
        """
        Insert a debug print out statement if the programmer requested. Handles 
        statements like `cudaq.dbg.ast.print_i64(i)`.
        """
        value = self.ifPointerThenLoad(value)
        printFunc = None
        printStr = '[cudaq-ast-dbg] '
        argsTy = [cc.PointerType.get(self.getIntegerType(8))]
        if dbgStmt == 'print_i64':
            if not IntegerType.isinstance(value.type):
                self.emitFatalError(
                    f"print_i64 requested, but value is not of integer type (type was {value.type})."
                )

            currentST = SymbolTable(self.module.operation)
            argsTy += [self.getIntegerType()]
            # If `printf` is not in the module, or if it is but the last
            # argument type is not an integer then we have to add it.
            if not 'print_i64' in currentST or not IntegerType.isinstance(
                    currentST['print_i64'].type.inputs[-1]):
                with InsertionPoint(self.module.body):
                    printOp = func.FuncOp('print_i64', (argsTy, []))
                    printOp.sym_visibility = StringAttr.get("private")
            currentST = SymbolTable(self.module.operation)
            printFunc = currentST['print_i64']
            printStr += '%ld\n'

        elif dbgStmt == 'print_f64':
            if not F64Type.isinstance(value.type):
                self.emitFatalError(
                    f"print_f64 requested, but value is not of float type (type was {value.type})."
                )

            currentST = SymbolTable(self.module.operation)
            argsTy += [self.getFloatType()]
            # If `printf` is not in the module, or if it is but the last
            # argument type is not an float then we have to add it
            if not 'print_f64' in currentST or not F64Type.isinstance(
                    currentST['print_f64'].type.inputs[-1]):
                with InsertionPoint(self.module.body):
                    printOp = func.FuncOp('print_f64', (argsTy, []))
                    printOp.sym_visibility = StringAttr.get("private")
            currentST = SymbolTable(self.module.operation)
            printFunc = currentST['print_f64']
            printStr += '%.12lf\n'
        else:
            raise self.emitFatalError(
                f"Invalid cudaq.dbg.ast statement - {dbgStmt}")

        strLitTy = cc.PointerType.get(
            cc.ArrayType.get(self.getIntegerType(8),
                             len(printStr) + 1))
        strLit = cc.CreateStringLiteralOp(strLitTy,
                                          StringAttr.get(printStr)).result
        strLit = cc.CastOp(cc.PointerType.get(self.getIntegerType(8)),
                           strLit).result
        func.CallOp(printFunc, [strLit, value])
        return

    def __load_vector_element(self, vector, index):
        """
        Load an element from a vector or array at the given index.
        
        Args:
            vector: MLIR Value of vector/array type
            index: MLIR Value containing integer index
            
        Returns:
            MLIR Value containing the loaded element
        """
        if cc.StdvecType.isinstance(vector.type):
            data_ptr = cc.StdvecDataOp(
                cc.PointerType.get(
                    cc.ArrayType.get(cc.StdvecType.getElementType(
                        vector.type))), vector).result
            return cc.LoadOp(
                cc.ComputePtrOp(
                    cc.PointerType.get(cc.StdvecType.getElementType(
                        vector.type)), data_ptr, [index],
                    DenseI32ArrayAttr.get([kDynamicPtrIndex]))).result
        return cc.LoadOp(
            cc.ComputePtrOp(
                cc.PointerType.get(
                    cc.ArrayType.getElementType(
                        cc.PointerType.getElementType(vector.type))), vector,
                [index], DenseI32ArrayAttr.get([kDynamicPtrIndex]))).result

    def __get_superior_type(self, t1, t2):
        """
        Get the superior numeric type between two MLIR types.
        Complex > F64 > F32 > Integer, with integers and complex promoting to the wider width.
        Returns None if no superior type can be determined.
        
        Args:
            t1: First MLIR type
            t2: Second MLIR type
            
        Returns:
            MLIR Type representing the superior type
        """

        def fp_type(fpt, ot):
            assert (F64Type.isinstance(fpt) or F32Type.isinstance(fpt))
            if F64Type.isinstance(ot):
                return ot
            if F32Type.isinstance(ot):
                return fpt
            if not IntegerType.isinstance(ot):
                return None
            if IntegerType(ot).width > 32:
                # matching Python behavior
                return F64Type.get()
            return fpt

        def complex_type(ct, ot):
            if ComplexType.isinstance(ot):
                ot = ComplexType(ot).element_type
            et = fp_type(ComplexType(ct).element_type, ot)
            if et is None:
                return None
            return self.getComplexTypeWithElementType(et)

        if ComplexType.isinstance(t1):
            return complex_type(t1, t2)
        if ComplexType.isinstance(t2):
            return complex_type(t2, t1)
        if F64Type.isinstance(t1) or F32Type.isinstance(t1):
            return fp_type(t1, t2)
        if F64Type.isinstance(t2) or F32Type.isinstance(t2):
            return fp_type(t2, t1)
        if IntegerType.isinstance(t1) and IntegerType.isinstance(t2):
            return self.getIntegerType(
                max(IntegerType(t1).width,
                    IntegerType(t2).width))
        return None

    def mlirTypeFromAnnotation(self, annotation):
        """
        Return the MLIR Type corresponding to the given kernel function argument
        type annotation.  Throws an exception if the programmer did not annotate
        function argument types.
        """
        msg = None
        try:
            return mlirTypeFromAnnotation(annotation, self.ctx, raiseError=True)
        except RuntimeError as e:
            msg = str(e)

        if msg is not None:
            self.emitFatalError(msg, annotation)

    def createForLoop(self, argTypes, bodyBuilder, inputs, evalCond, evalStep, orElseBuilder = None):

        # post-conditional would be a do-while loop
        isPostConditional = BoolAttr.get(False)
        loop = cc.LoopOp(argTypes, inputs, isPostConditional)

        whileBlock = Block.create_at_start(loop.whileRegion, argTypes)
        with InsertionPoint(whileBlock):
            condVal = evalCond(whileBlock.arguments)
            cc.ConditionOp(condVal, whileBlock.arguments)

        bodyBlock = Block.create_at_start(loop.bodyRegion, argTypes)
        with InsertionPoint(bodyBlock):
            self.symbolTable.pushScope()
            self.pushForBodyStack(bodyBlock.arguments)
            bodyBuilder(bodyBlock.arguments)
            if not self.hasTerminator(bodyBlock):
                cc.ContinueOp(bodyBlock.arguments)
            self.popForBodyStack()
            self.symbolTable.popScope()

        stepBlock = Block.create_at_start(loop.stepRegion, argTypes)
        with InsertionPoint(stepBlock):
            stepVals = evalStep(stepBlock.arguments)
            cc.ContinueOp(stepVals)

        if orElseBuilder:
            elseBlock = Block.create_at_start(loop.elseRegion, argTypes)
            with InsertionPoint(elseBlock):
                self.symbolTable.pushScope()
                orElseBuilder(elseBlock.arguments)
                if not self.hasTerminator(elseBlock):
                    cc.ContinueOp(elseBlock.arguments)
                self.symbolTable.popScope()

        return loop

    def createMonotonicForLoop(self, bodyBuilder, startVal, stepVal, endVal, isDecrementing=False, orElseBuilder = None):

        iTy = self.getIntegerType()
        assert startVal.type == iTy
        assert stepVal.type == iTy
        assert endVal.type == iTy
        
        condPred = IntegerAttr.get(iTy, 4) if isDecrementing else IntegerAttr.get(iTy, 2)
        return self.createForLoop([iTy], 
                                 lambda args: bodyBuilder(args[0]),
                                 [startVal], 
                                 lambda args: arith.CmpIOp(condPred, args[0], endVal).result,
                                 lambda args: [arith.AddIOp(args[0], stepVal).result], 
                                 None if orElseBuilder is None else (lambda args: orElseBuilder(args[0])))

    def createInvariantForLoop(self, bodyBuilder, endVal):
        """
        Create an invariant loop using the CC dialect. 
        """

        startVal = self.getConstantInt(0)
        stepVal = self.getConstantInt(1)

        loop = self.createMonotonicForLoop(bodyBuilder, 
                                      startVal=startVal, 
                                      stepVal=stepVal, 
                                      endVal=endVal)
        loop.attributes.__setitem__('invariant', UnitAttr.get())

    def __deconstructAssignment(self, target, value, process=None):
        if process is not None:
            target, value = process(target, value)
        if isinstance(target, ast.Name):
            if value is not None:
                self.symbolTable[target.id] = value
        elif isinstance(target, ast.Tuple):
            if isinstance(value, ast.Tuple) or \
                isinstance(value, ast.List):
                nrArgs = len(value.elts)
                getItem = lambda idx: value.elts[idx]
            elif isinstance(value, tuple) or \
                isinstance(value, list):
                nrArgs = len(value)
                getItem = lambda idx: value[idx]
            else:
                value = self.ifPointerThenLoad(value)
                if cc.StructType.isinstance(value.type):
                    argTypes = cc.StructType.getTypes(value.type)
                    nrArgs = len(argTypes)
                    getItem = lambda idx: cc.ExtractValueOp(
                        argTypes[idx], value, [],
                        DenseI32ArrayAttr.get([idx], context=self.ctx)).result
                elif quake.StruqType.isinstance(value.type):
                    argTypes = quake.StruqType.getTypes(value.type)
                    nrArgs = len(argTypes)
                    getItem = lambda idx: quake.GetMemberOp(
                        argTypes[idx], value,
                        IntegerAttr.get(self.getIntegerType(32), idx)).result
                elif cc.StdvecType.isinstance(value.type):
                    # We will get a runtime error for out of bounds access
                    # FIXME: WRITE HELPER FUNCTION FOR GET VECTOR ELEMENT PTR
                    eleTy = cc.StdvecType.getElementType(value.type)
                    elePtrTy = cc.PointerType.get(eleTy)
                    arrTy = cc.ArrayType.get(eleTy)
                    ptrArrTy = cc.PointerType.get(arrTy)
                    vecPtr = cc.StdvecDataOp(ptrArrTy, value).result
                    attr = DenseI32ArrayAttr.get([kDynamicPtrIndex],
                                                 context=self.ctx)
                    nrArgs = len(target.elts)
                    getItem = lambda idx: cc.LoadOp(
                        cc.ComputePtrOp(elePtrTy, vecPtr, [
                            self.getConstantInt(idx)
                        ], attr).result).result
                elif quake.VeqType.isinstance(value.type):
                    # We will get a runtime error for out of bounds access
                    nrArgs = len(target.elts)
                    getItem = lambda idx: quake.ExtractRefOp(
                        quake.RefType.get(),
                        value,
                        -1,
                        index=self.getConstantInt(idx)).result
                else:
                    nrArgs = 0
            if nrArgs != len(target.elts):
                self.emitFatalError("shape mismatch in tuple deconstruction",
                                    self.currentNode)
            for i in range(nrArgs):
                self.__deconstructAssignment(target.elts[i],
                                             getItem(i),
                                             process=process)
        else:
            self.emitFatalError("unsupported target in tuple deconstruction",
                                self.currentNode)

    def __processRangeLoopIterationBounds(self, pyVals):
        """
        Analyze `range(...)` bounds and return the start, end, and step values,
        as well as whether or not this a decrementing range.
        """
        iTy = self.getIntegerType(64)
        zero = arith.ConstantOp(iTy, IntegerAttr.get(iTy, 0))
        one = arith.ConstantOp(iTy, IntegerAttr.get(iTy, 1))
        values = self.__groupValues(pyVals, [(1, 3)])

        isDecrementing = False
        if len(pyVals) == 3:
            # Find the step val and we need to know if its decrementing can be
            # incrementing or decrementing
            stepVal = values[2]
            if isinstance(pyVals[2], ast.Constant):
                pyStepVal = pyVals[2].value
            elif isinstance(pyVals[2], ast.UnaryOp) and \
                isinstance(pyVals[2].op, ast.USub) and \
                isinstance(pyVals[2].operand, ast.Constant):
                pyStepVal = -pyVals[2].operand.value
            else:
                self.emitFatalError(
                    'range step value must be a constant', self.currentNode)
            if pyStepVal == 0:
                self.emitFatalError("range step value must be non-zero", self.currentNode)
            isDecrementing = pyStepVal < 0

            # exclusive end
            endVal = values[1]

            # inclusive start
            startVal = values[0]

        elif len(pyVals) == 2:
            stepVal = one
            endVal = values[1]
            startVal = values[0]
        else:
            stepVal = one
            endVal = values[0]
            startVal = zero

        startVal = self.ifPointerThenLoad(startVal)
        endVal = self.ifPointerThenLoad(endVal)
        stepVal = self.ifPointerThenLoad(stepVal)

        for idx, v in enumerate([startVal, endVal, stepVal]):
            if not IntegerType.isinstance(v.type):
                # matching Python behavior to error on non-integer values
                self.emitFatalError(
                    "non-integer value in range expression",
                    pyVals[idx if len(pyVals) > 1 else 0])
        return startVal, endVal, stepVal, isDecrementing

    def __groupValues(self, pyvals, groups: list[int | tuple[int, int]]):
        '''
        Helper function that visits the given AST nodes (`pyvals`),
        and groups them according to the specified list.
        The list contains integers or tuples of two integers.
        Integer values have to be positive or -1, where -1
        indicates that any number of values is acceptable.
        Tuples of two integers (min, max) indicate that any number
        of values in [min, max] is acceptable.
        The list may only contain at most one negative integer or
        tuple (enforced via assert only).

        Emits a fatal error if any of the given `pyvals` did not
        generate a value. Emits a fatal error if there are too
        too many or too few values to satisfy the requested grouping.

        Returns a tuple of value groups. Each value group is
        either a single value (if the corresponding entry in `groups`
        equals 1), or a list of values.
        '''

        def group_values(numExpected, values, reverse):
            groupedVals = []
            current_idx = 0
            for nArgs in numExpected:
                if isinstance(nArgs, int) and \
                    nArgs == 1 and current_idx < len(values):
                    groupedVals.append(values[current_idx])
                    current_idx += 1
                    continue
                if isinstance(nArgs, tuple):
                    minNumArgs, maxNumArgs = nArgs
                    if minNumArgs == maxNumArgs:
                        nArgs = minNumArgs
                if not isinstance(nArgs, int) or nArgs < 0:
                    break
                if current_idx + nArgs > len(values):
                    self.emitFatalError("missing value", self.currentNode)
                groupedVals.append(values[current_idx: current_idx + nArgs])
                if reverse: groupedVals[-1].reverse()
                current_idx += nArgs
            remaining = values[current_idx:]
            numExpected = numExpected[len(groupedVals):]
            if reverse:
                remaining.reverse()
                groupedVals.reverse()
                numExpected.reverse()
            return groupedVals, numExpected, remaining

        [self.visit(arg) for arg in pyvals]
        values = self.popAllValues(len(pyvals))
        groups.reverse()
        backVals, groups, values = group_values(groups, values, reverse=True)
        frontVals, groups, values = group_values(groups, values, reverse=False)
        if not groups:
            if values:
                self.emitFatalError("too many values", self.currentNode)
            groupedVals = *frontVals, *backVals
        else:
            assert len(groups) == 1 # ambiguous otherwise
            if isinstance(groups[0], tuple):
                minNumArgs, maxNumArgs = groups[0]
                assert 0 <= minNumArgs and (minNumArgs <= maxNumArgs or maxNumArgs < 0)
                if len(values) < minNumArgs:
                    self.emitFatalError("missing value", self.currentNode)
                if len(values) > maxNumArgs and maxNumArgs > 0:
                    self.emitFatalError("too many values", self.currentNode)
            groupedVals = *frontVals, values, *backVals
        return groupedVals[0] if len(groupedVals) == 1 else groupedVals    

    def visit(self, node):
        self.debug_msg(lambda: f'[Visit {type(node).__name__}]', node)
        self.indent_level += 1
        parentNode = self.currentNode
        self.currentNode = node
        numVals = 0 if isinstance(node, ast.Module) else self.valueStack.currentNumValues
        self.valueStack.pushFrame()
        super().visit(node)
        self.valueStack.popFrame()
        if isinstance(node, ast.Module):
            if not self.valueStack.isEmpty:
                self.emitFatalError("processing error - unprocessed frame(s) in value stack", node)
        elif self.valueStack.currentNumValues - numVals > 1:
            # Do **NOT** change this to be more permissive and allow
            # multiple values to be pushed without pushing proper 
            # frames for sub-nodes. If visiting a single node 
            # potentially produces more than one value, the bridge
            # quickly will be a mess because we will easily end up
            # with values in the wrong places.
            self.emitFatalError("must not generate more one value at a time in each frame", node)
        self.currentNode = parentNode
        self.indent_level -= 1

    # FIXME: using generic_visit the way we do seems incredibly dangerous;
    # we use this and make assumptions about what values are on the value stack
    # without any validation that we got the right values.
    # The whole value stack needs to be revised; we need to properly push and pop
    # not just individual values but groups of values to ensure that the right
    # pieces get the right arguments (and give a proper error otherwise).
    def generic_visit(self, node):
        self.debug_msg(lambda: f'[Generic Visit]', node)
        for field, value in reversed(list(ast.iter_fields(node))):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self.visit(item)
            elif isinstance(value, ast.AST):
                self.visit(value)

    def visit_FunctionDef(self, node):
        """
        Create an MLIR `func.FuncOp` for the given FunctionDef AST node. For the
        top-level FunctionDef, this will add the `FuncOp` to the `ModuleOp`
        body, annotate the `FuncOp` with `cudaq-entrypoint` if it is an Entry
        Point CUDA-Q kernel, and visit the rest of the FunctionDef body. If this
        is an inner FunctionDef, this will treat the function as a CC lambda
        function and add the cc.callable-typed value to the symbol table, keyed
        on the FunctionDef name.

        We keep track of the top-level function name as well as its internal
        MLIR name, prefixed with the __nvqpp__mlirgen__ prefix.
        """

        if self.buildingEntryPoint:
            # This is an inner function def, we will
            # treat it as a cc.callable (cc.create_lambda)
            self.debug_msg(lambda: f'Visiting inner FunctionDef {node.name}')

            arguments = node.args.args
            if len(arguments):
                self.emitFatalError(
                    "inner function definitions cannot have arguments.", node)

            ty = cc.CallableType.get([])
            createLambda = cc.CreateLambdaOp(ty)
            initRegion = createLambda.initRegion
            initBlock = Block.create_at_start(initRegion, [])
            # TODO: process all captured variables in the main function
            # definition first to avoid reusing code not defined in the
            # same or parent scope of the produced MLIR.
            with InsertionPoint(initBlock):
                [self.visit(n) for n in node.body]
                cc.ReturnOp([])
            self.symbolTable[node.name] = createLambda.result
            return

        with self.ctx, InsertionPoint(self.module.body), self.loc:

            # Get the potential documentation string
            self.docstring = ast.get_docstring(node)

            # Get the argument types and argument names
            # this will throw an error if the types aren't annotated
            self.argTypes = [
                self.mlirTypeFromAnnotation(arg.annotation)
                for arg in node.args.args
            ]
            parentResultType = self.knownResultType
            if node.returns is not None and not (isinstance(
                    node.returns, ast.Constant) and
                                                 (node.returns.value is None)):
                self.knownResultType = self.mlirTypeFromAnnotation(node.returns)

            self.name = node.name
            self.capturedDataStorage.name = self.name

            # the full function name in MLIR is `__nvqpp__mlirgen__` + the function name
            if not self.disableNvqppPrefix:
                fullName = nvqppPrefix + node.name
            else:
                fullName = node.name

            # Create the FuncOp
            f = func.FuncOp(fullName, (self.argTypes, [] if self.knownResultType
                                       == None else [self.knownResultType]),
                            loc=self.loc)
            self.kernelFuncOp = f

            # Set this kernel as an entry point if the argument types are classical only
            areQuantumTypes = [self.isQuantumType(ty) for ty in self.argTypes]
            f.attributes.__setitem__('cudaq-kernel', UnitAttr.get())
            if True not in areQuantumTypes and not self.disableEntryPointTag:
                f.attributes.__setitem__('cudaq-entrypoint', UnitAttr.get())

            # Create the entry block
            self.entry = f.add_entry_block()

            # Set the insertion point to the start of the entry block
            with InsertionPoint(self.entry):
                self.symbolTable.pushScope()
                # Process function arguments like any other assignments.
                if node.args.args:
                    assignNode = ast.Assign()
                    if len(node.args.args) == 1:
                        assignNode.targets = [ast.Name(node.args.args[0].arg)]
                        assignNode.value = self.entry.arguments[0]
                    else:
                        assignNode.targets = [ast.Tuple([ast.Name(arg.arg) for arg in node.args.args])]
                        assignNode.value = [self.entry.arguments[idx] for idx in range(len(self.entry.arguments.types))]
                    assignNode.lineno = node.lineno
                    self.visit_Assign(assignNode)

                # Intentionally set after we process the argument assignment,
                # since we currently treat value vs reference semantics slightly
                # differently when we have arguments vs when we have local values.
                # To not make this distinction, we would need to add support
                # for having proper reference arguments, which we don't want to.
                # Barring that, we at least try to be nice and give errors on
                # assignments that may lead to unexpected (i.e. non-Pythonic)
                # behavior.
                self.buildingEntryPoint = True
                [self.visit(n) for n in node.body]
                # Add the return operation
                if not self.hasTerminator(self.entry):
                    # If the function has a known (non-None) return type, emit
                    # an `undef` of that type and return it; else return void
                    if self.knownResultType is not None:
                        undef = cc.UndefOp(self.knownResultType).result
                        func.ReturnOp([undef])
                    else:
                        func.ReturnOp([])
                self.buildingEntryPoint = False
                self.symbolTable.popScope()

            if True not in areQuantumTypes:
                attr = DictAttr.get(
                    {
                        fullName:
                            StringAttr.get(
                                fullName + '_PyKernelEntryPointRewrite',
                                context=self.ctx)
                    },
                    context=self.ctx)
                self.module.operation.attributes.__setitem__(
                    'quake.mangled_name_map', attr)

            globalKernelRegistry[node.name] = f
            self.symbolTable.clear()
            self.knownResultType = parentResultType

    def visit_Expr(self, node):
        """
        Implement `ast.Expr` visitation to screen out all multi-line
        `docstrings`. These are differentiated from other strings at the
        node-type level. Strings we may care about will have been assigned to a
        variable (hence `ast.Assign` nodes), while other strings will exist as
        standalone expressions with no uses.
        """
        if hasattr(node, 'value') and isinstance(node.value, ast.Constant):
            self.debug_msg(lambda: f'[(Inline) Visit Constant]', node.value)
            constant = node.value
            if isinstance(constant.value, str):
                return

        self.visit(node.value)
        if self.valueStack.currentNumValues > 0:
            # An `ast.Expr` object is created when an expression
            # is used as a statement. This expression may produce
            # a value, which is ignored (not assigned) in the 
            # Python code. We hence need to pop that value to
            # match that behavior and ignore it.
            self.popValue()

    def visit_Lambda(self, node):
        """
        Map a lambda expression in a CUDA-Q kernel to a CC Lambda (a Value of
        `cc.callable` type using the `cc.create_lambda` operation). Note that we
        extend Python with a novel syntax to specify a list of independent
        statements (Python lambdas must have a single statement) by allowing
        programmers to return a Tuple where each element is an independent
        statement.

        ```python
            functor = lambda : (h(qubits), x(qubits), ry(np.pi, qubits))  # qubits captured from parent region
            # is equivalent to 
            def functor(qubits):
                h(qubits)
                x(qubits)
                ry(np.pi, qubits)
        ```
        """
        arguments = node.args.args
        if len(arguments):
            self.emitFatalError("CUDA-Q lambdas cannot have arguments.", node)

        ty = cc.CallableType.get([])
        createLambda = cc.CreateLambdaOp(ty)
        initBlock = Block.create_at_start(createLambda.initRegion, [])
        with InsertionPoint(initBlock):
            # Python lambdas can only have a single statement.
            # Here we will enhance our language by processing a single Tuple
            # statement as a set of statements for each element of the tuple
            if isinstance(node.body, ast.Tuple):
                self.debug_msg(lambda: f'[(Inline) Visit Tuple]', node.body)
                [self.visit(element) for element in node.body.elts]
            else:
                self.visit(
                    node.body)  # only one statement in a python lambda :(
            cc.ReturnOp([])
        self.pushValue(createLambda.result)
        return

    def visit_Assign(self, node):
        """
        Map an assign operation in the AST to an equivalent variable value
        assignment in the MLIR. This method will first see if this is a tuple
        assignment, enabling one to assign multiple values in a single
        statement.

        For all assignments, the variable name will be used as a key for the
        symbol table, mapping to the corresponding MLIR Value. For values of
        `ref` / `veq`, `i1`, or `cc.callable`, the values will be stored
        directly in the table. For all other values, the variable will be
        allocated with a `cc.alloca` op, and the loaded value will be stored in
        the symbol table.
        """

        def check_not_captured(name):
            if name in self.capturedVars:
                self.emitFatalError(
                    "CUDA-Q does not allow assignment to variable {} captured from parent scope."
                    .format(name), node)

        def process_assignment(target, value):
            if isinstance(target, ast.Tuple):

                if isinstance(value, ast.Tuple) or \
                    isinstance(value, ast.List):
                    return target, value

                if isinstance(value, ast.AST):
                    self.visit(value)
                    value = self.popValue()
                    return target, value

                return target, value

            # Handle simple `var = expr`
            if isinstance(target, ast.Name):
                if target.id in self.capturedVars:
                    # Local variable shadows the captured one
                    del self.capturedVars[target.id]

                def processValue():
                    # Retain the variable name for potential children (like `mz(q, registerName=...)`)
                    self.currentAssignVariableName = target.id
                    self.visit(value)
                    self.currentAssignVariableName = None
                    return self.popValue()

                if not target.id in self.symbolTable or \
                    target.id in self.symbolTable.symbolTable[-1]:
                    if isinstance(value, ast.AST):
                        value = processValue()

                    # If `buildingEntryPoint` is not set we are processing function arguments.
                    # Function arguments are always passed by value and should hence be preserved
                    # as such to make sure they are not modified.
                    storeAsValue = not self.buildingEntryPoint or \
                        self.isQuantumType(value.type) or \
                        cc.CallableType.isinstance(value.type) or \
                        self.isMeasureResultType(value.type, value)

                    # The target variable is not yet defined or has been 
                    # defined within the current scope and we can simply
                    # add or modify the symbol table entry.
                    # If we have a pointer, we don't create a new copy;
                    # Python behavior is to assign by pointer.
                    if storeAsValue or cc.PointerType.isinstance(value.type):
                        # FIXME: if the value is already in the symbol table, we need to
                        # make sure that the *existing* reference is updated!
                        return target, value
                    if cc.StructType.isinstance(value.type) and \
                        cc.StructType.getName(value.type) != 'tuple':
                        # When we create structs locally, we create them as pointers.
                        # For tuples, we prevent any modification by preventing
                        # the creation of item pointers.
                        # For dataclasses, their fields can be modified and that
                        # any such modification is properly reflected in all values
                        # that contain a reference to the same value.
                        # However, when we pass dataclasses as function arguments,
                        # or return them from functions, we pass them as values.
                        # We hence want to give a proper error and enforce that
                        # these values cannot be modified to avoid surprising (i.e.
                        # non-Pythonic) behavior.
                        self.emitFatalError("cannot create reference to dataclass passed to or returned from function - use `.copy()` to create a new value that can be assigned", node)
                    address = cc.AllocaOp(cc.PointerType.get(value.type), TypeAttr.get(value.type)).result
                    # FIXME: if the value is already in the symbol table, we need to
                    # make sure that the *existing* reference is updated!
                    cc.StoreOp(value, address)
                    return target, address

                # The target variable exists in a parent scope;
                # if it is a pointer, we can update the pointer,
                # and otherwise the assignment must fail;
                # if we pushed an update to the innermost scope
                # in the table, then that update is not reflected
                # in the parent scope.
                symbolTableEntry = self.symbolTable[target.id]
                if cc.PointerType.isinstance(symbolTableEntry.type):
                    expectedTy = cc.PointerType.getElementType(symbolTableEntry.type)
                else:
                    self.emitFatalError("variable captured from parent scope cannot be modified", node)

                if isinstance(value, ast.AST):
                    value = processValue()
                if cc.PointerType.isinstance(value.type):
                    valPtrElemTy = cc.PointerType.getElementType(value.type)
                    if cc.StdvecType.isinstance(valPtrElemTy) or \
                        cc.StructType.isinstance(valPtrElemTy):
                        # Vectors and data classes are supposed to be assigned by reference,
                        # but we cannot currently deal with this properly right now:
                        # If we update the symbol table to contain the new reference,
                        # then the change is either not reflected in the parent scope, or
                        # reflected regardless of whether we ever entered the child scope
                        # that does the update. If we load the value and store the new value
                        # in the existing pointer defined in the parent scope, then any
                        # future changes to the assigned value will not be reflected.
                        self.emitFatalError("cannot assign a reference that could modify variable in parent scope", node)
                '''
                if cc.PointerType.isinstance(value.type):
                    #l1 = [0]
                    #if True:
                    #    l2 = [1]
                    #    l1 = l2
                    #    l2[0] = 5
                    # l1[0] and l2[0] should be 5 now
                    # l1[0] = 3
                    # l1[0] and l2[0] should be 3 now
                '''
                value = self.changeOperandToType(expectedTy, self.ifPointerThenLoad(value))
                cc.StoreOp(value, symbolTableEntry)
                return target, None 

            # Make sure we process arbitrary combinations
            # of subscript and attributes
            target_root = target
            while isinstance(target_root, ast.Subscript) or \
                isinstance(target_root, ast.Attribute):
                target_root = target_root.value

            # Handle assignments like `listVar[IDX] = expr`
            if isinstance(target, ast.Subscript) and \
                isinstance(target_root, ast.Name) and \
                target_root.id in self.symbolTable:
                check_not_captured(target_root.id)

                # Visit_Subscript will try to load any pointer and return it
                # but here we want the pointer, so flip that flag
                self.pushPointerValue = True
                # Visit the subscript node, get the pointer value
                self.visit(target)
                # Reset the push pointer value flag
                self.pushPointerValue = False

                ptrVal = self.popValue()
                if self.isQuantumType(ptrVal.type):
                    self.emitFatalError("quantum data type cannot be updated", node)
                if not cc.PointerType.isinstance(ptrVal.type):
                    self.emitFatalError(
                        "Invalid CUDA-Q subscript assignment, variable must be a pointer.",
                        node)

                # Visit the value being assigned
                self.visit(node.value)
                # FIXME: WE NEED TO FOLLOW SIMILAR LOGIC AS ABOVE;
                # LOAD IF NEEDED, CAST TYPE, CHECK WE DON'T CREATE
                # AN ALIAS OF A REFERENCE IF WE NEED TO LOAD
                valueToStore = self.popValue()
                # Store the value
                cc.StoreOp(valueToStore, ptrVal)
                return target_root, None

            # Handle assignments like `classVar.attr = expr`
            if isinstance(target, ast.Attribute) and \
                isinstance(target_root, ast.Name) and \
                target_root.id in self.symbolTable:
                
                check_not_captured(target_root.id)

                self.pushPointerValue = True
                # Visit the attribute node, get the pointer value
                self.visit(target)
                # Reset the push pointer value flag
                self.pushPointerValue = False

                ptrVal = self.popValue()
                if self.isQuantumType(ptrVal.type):
                    self.emitFatalError("quantum data type cannot be updated", node)
                if not cc.PointerType.isinstance(ptrVal.type):
                    self.emitFatalError("invalid CUDA-Q attribute assignment",
                                        node)
                # Visit the value being assigned
                self.visit(node.value)
                # FIXME: WE NEED TO FOLLOW SIMILAR LOGIC AS ABOVE;
                # LOAD IF NEEDED, CAST TYPE, CHECK WE DON'T CREATE
                # AN ALIAS OF A REFERENCE IF WE NEED TO LOAD
                valueToStore = self.popValue()
                # Store the value
                cc.StoreOp(valueToStore, ptrVal)
                return target_root, None

            self.emitFatalError("Invalid target for assignment", node)

        if len(node.targets) > 1:
            # I am not entirely sure what kinds of Python language constructs would
            # result in having more than 1 target here, hence giving an error on it for now.
            # (It would be easy to process this as target tuple, but it may not be correct to do so.)
            self.emitFatalError(
                "CUDA-Q does not allow multiple targets in assignment", node)
        self.__deconstructAssignment(node.targets[0],
                                     node.value,
                                     process=process_assignment)

    def visit_Attribute(self, node):
        """
        Visit an attribute node and map to valid MLIR code. This method
        specifically looks for attributes like method calls, or common
        attributes we'll see from ubiquitous external modules like `numpy`.
        """
        if isinstance(node.value,
                      ast.Name) and not node.value.id in self.symbolTable:

            if node.value.id in ['np', 'numpy', 'math']:
                if node.attr == 'complex64':
                    self.pushValue(self.getComplexType(width=32))
                elif node.attr == 'complex128':
                    self.pushValue(self.getComplexType(width=64))
                elif node.attr == 'float64':
                    self.pushValue(self.getFloatType(width=64))
                elif node.attr == 'float32':
                    self.pushValue(self.getFloatType(width=32))
                elif node.attr == 'int64':
                    self.pushValue(self.getIntegerType(width=64))
                elif node.attr == 'int32':
                    self.pushValue(self.getIntegerType(width=32))
                elif node.attr == 'pi':
                    self.pushValue(self.getConstantFloat(np.pi))
                elif node.attr == 'e':
                    self.pushValue(self.getConstantFloat(np.e))
                elif node.attr == 'euler_gamma':
                    self.pushValue(self.getConstantFloat(np.euler_gamma))
                elif node.attr == 'array' or self.__isSupportedNumpyFunction(
                        node.attr):
                    return
                else:
                    self.emitFatalError(
                        "{}.{} is not supported".format(node.value.id,
                                                        node.attr), node)
                return

            if node.value.id == 'cudaq':
                if node.attr in [
                        'DepolarizationChannel', 'AmplitudeDampingChannel',
                        'PhaseFlipChannel', 'BitFlipChannel', 'PhaseDamping',
                        'ZError', 'XError', 'YError', 'Pauli1', 'Pauli2',
                        'Depolarization1', 'Depolarization2'
                ]:
                    self.emitFatalError("noise channels may only be used as part of call expressions", node)

                # must be handled by the parent
                return

            if node.attr == 'ctrl' or node.attr == 'adj':
                # to be processed by the caller
                return

        def process_potential_ptr_types(value):
            """
            Helper function to process anything that the parent may assign to, 
            depending on whether value is a pointer or not.
            """
            valType = value.type
            if cc.PointerType.isinstance(valType):
                valType = cc.PointerType.getElementType(valType)

            if quake.StruqType.isinstance(valType):
                # Need to extract value instead of load from compute pointer.
                structIdx, memberTy = self.getStructMemberIdx(
                    node.attr, value.type)
                attr = IntegerAttr.get(self.getIntegerType(32), structIdx)
                self.pushValue(quake.GetMemberOp(memberTy, value, attr).result)
                return True

            if cc.PointerType.isinstance(value.type) and \
                cc.StructType.isinstance(valType):
                structIdx, memberTy = self.getStructMemberIdx(node.attr, valType)
                eleAddr = cc.ComputePtrOp(cc.PointerType.get(memberTy),
                                        value, [],
                                        DenseI32ArrayAttr.get([structIdx
                                                                ])).result

                if self.pushPointerValue:
                    self.pushValue(eleAddr)
                    return True

                # Load the value
                eleAddr = cc.LoadOp(eleAddr).result
                self.pushValue(eleAddr)
                return True

            elif cc.StructType.isinstance(value.type):
                if node.attr == 'copy':
                    # needs to be handled by the caller
                    return True

                if self.pushPointerValue:
                    self.emitFatalError("value cannot be modified - use `.copy()` to create a new value that can be modified", node)

                # Handle direct struct value - use ExtractValueOp (more efficient)
                structIdx, memberTy = self.getStructMemberIdx(node.attr, value.type)
                extractedValue = cc.ExtractValueOp(
                    memberTy, value, [],
                    DenseI32ArrayAttr.get([structIdx])).result

                self.pushValue(extractedValue)
                return True

            elif quake.VeqType.isinstance(valType) or \
                cc.StdvecType.isinstance(valType):
                return self.__isSupportedVectorFunction(node.attr)

            return False

        # Make sure we preserve pointers for structs
        if isinstance(node.value,
                      ast.Name) and node.value.id in self.symbolTable:
            value = self.symbolTable[node.value.id]
            processed = process_potential_ptr_types(value)
            if processed:
                return

        self.visit(node.value)
        value = self.popValue()
        if process_potential_ptr_types(value):
            return
        value = self.ifPointerThenLoad(value)

        if ComplexType.isinstance(value.type):
            if (node.attr == 'real'):
                self.pushValue(complex.ReOp(value).result)
            elif (node.attr == 'imag'):
                self.pushValue(complex.ImOp(value).result)
            else:
                self.emitFatalError("invalid attribute on complex value", node)
            return

        # `numpy` arrays have a size attribute
        if node.attr == 'size':
            if quake.VeqType.isinstance(value.type):
                self.pushValue(
                    quake.VeqSizeOp(self.getIntegerType(), value).result)
                return True
            if cc.StdvecType.isinstance(value.type):
                self.pushValue(
                    cc.StdvecSizeOp(self.getIntegerType(), value).result)
                return True

        self.emitFatalError("unrecognized attribute {}".format(node.attr),
                            node)

    def visit_Call(self, node):
        """
        Map a Python Call operation to equivalent MLIR. This method handles
        functions that are `ast.Name` and `ast.Attribute` objects.

        This function handles all built-in unitary and measurement gates
        as well as all the ways to adjoint and control them.
        General calls to previously seen CUDA-Q kernels or registered
        operations are supported. 

        ```python
            q, r = cudaq.qubit(), cudaq.qubit()
            qubits = cudaq.qubit(2)

            x(q) # apply x to q
            x(q, r) # apply x to q and r
            x(qubits) # for q in qubits: apply x 
            ry(np.pi, qubits)
        ```
        """
        global globalRegisteredOperations

        def convertArguments(expectedArgTypes, values):
            assert len(expectedArgTypes) == len(values)
            args = []
            for idx, expectedTy in enumerate(expectedArgTypes):
                arg = self.ifPointerThenLoad(values[idx])
                arg = self.changeOperandToType(expectedTy,
                                               arg,
                                               allowDemotion=True)
                args.append(arg)
            return args

        def getNegatedControlQubits(controls):
            negatedControlQubits = None
            if len(self.controlNegations):
                negCtrlBools = [None] * len(controls)
                for i, c in enumerate(controls):
                    negCtrlBools[i] = c in self.controlNegations
                negatedControlQubits = DenseBoolArrayAttr.get(negCtrlBools)
                self.controlNegations.clear()
            return negatedControlQubits

        def processFunctionCall(fType):
            nrArgs = len(fType.inputs)
            values = self.__groupValues(node.args, [(nrArgs, nrArgs)])
            values = convertArguments([t for t in fType.inputs], values)
            if len(fType.results) == 0:
                func.CallOp(otherKernel, values)
                return
            return func.CallOp(otherKernel, values).result

        def checkControlAndTargetTypes(controls, targets):
            """
            Check that the provided control and target operands are 
            of an appropriate type. Emit a fatal error if not. 
            """

            def is_qvec_or_qubits(vals):
                # We can either have a single item that is a vector of qubits,
                # or multiple single-qubit items.
                return all((quake.RefType.isinstance(v.type) for v in vals)) or \
                    (len(vals) == 1 and quake.VeqType.isinstance(vals[0].type))

            if len(controls) > 0 and not is_qvec_or_qubits(controls):
                self.emitFatalError(
                    f'invalid argument type for control operand', node)
            if len(targets) == 0:
                self.emitFatalError(f'missing argument for target operand',
                                    node)
            elif not is_qvec_or_qubits(targets):
                self.emitFatalError(f'invalid argument type for target operand',
                                    node)

        def processQuantumOperation(opName, controls, targets, *args, broadcast = lambda q: [q], **kwargs):
            opCtor = getattr(quake, f'{opName}Op')
            checkControlAndTargetTypes(controls, targets)
            if not broadcast:
                return opCtor(*args, controls, targets, **kwargs)
            elif quake.VeqType.isinstance(targets[0].type):
                assert len(targets) == 1
                def bodyBuilder(iterVal):
                    q = quake.ExtractRefOp(self.getRefType(),
                                           targets[0],
                                           -1,
                                           index=iterVal).result
                    opCtor(*args, controls, broadcast(q), **kwargs)
                veqSize = quake.VeqSizeOp(self.getIntegerType(),
                                          targets[0]).result
                self.createInvariantForLoop(bodyBuilder, veqSize)
            else:
                for target in targets:
                    opCtor(*args, controls, broadcast(target), **kwargs)

        def processQuakeCtor(opName, pyArgs, isCtrl, isAdj, numParams = 0, numTargets = 1):
            kwargs = {}
            if isCtrl:
                argGroups = [(numParams, numParams), (1, -1), (numTargets, numTargets)]
                 # FIXME: we could allow this as long as we have 1 target
                kwargs['broadcast'] = False
            elif numTargets == 1:
                # when we have a single target and no controls, we generally 
                # support any version of `x(qubit)`, `x(qvector)`, `x(q, r)`
                argGroups = [(numParams, numParams), 0, (1, -1)]
            else:
                argGroups = [(numParams, numParams), 0, (numTargets, numTargets)]
                kwargs['broadcast'] = False

            params, controls, targets = self.__groupValues(pyArgs, argGroups)
            if isCtrl:
                negatedControlQubits = getNegatedControlQubits(controls)
                kwargs['negated_qubit_controls'] = negatedControlQubits
            if isAdj:
                kwargs['is_adj'] = True
            params = [self.changeOperandToType(self.getFloatType(), param) for param in params]
            processQuantumOperation(opName, controls, targets, [], params, **kwargs)

        # do not walk the FunctionDef decorator_list arguments
        if isinstance(node.func, ast.Attribute):
            self.debug_msg(lambda: f'[(Inline) Visit Attribute]', node.func)
            if hasattr(
                    node.func.value, 'id'
            ) and node.func.value.id == 'cudaq' and node.func.attr == 'kernel':
                return

            # If we have a `func = ast.Attribute``, then it could be that we
            # have a previously defined kernel function call with manually
            # specified module names.
            # e.g. `cudaq.lib.test.hello.fermionic_swap``. In this case, we
            # assume FindDepKernels has found something like this, loaded it,
            # and now we just want to get the function name and call it.

            # First let's check for registered C++ kernels
            cppDevModNames = []
            value = node.func.value
            if isinstance(value, ast.Name) and value.id != 'cudaq':
                self.debug_msg(lambda: f'[(Inline) Visit Name]', value)
                cppDevModNames = [node.func.attr, value.id]
            else:
                self.debug_msg(lambda: f'[(Inline) Visit Attribute]', value)
                while isinstance(value, ast.Attribute):
                    cppDevModNames.append(value.attr)
                    value = value.value
                    if isinstance(value, ast.Name):
                        self.debug_msg(lambda: f'[(Inline) Visit Name]', value)
                        cppDevModNames.append(value.id)
                        break

            devKey = '.'.join(cppDevModNames[::-1])

            def get_full_module_path(partial_path):
                parts = partial_path.split('.')
                for module_name, module in sys.modules.items():
                    if module_name.endswith(parts[0]):
                        try:
                            obj = module
                            for part in parts[1:]:
                                obj = getattr(obj, part)
                            return f"{module_name}.{'.'.join(parts[1:])}"
                        except AttributeError:
                            continue
                return partial_path

            devKey = get_full_module_path(devKey)
            if cudaq_runtime.isRegisteredDeviceModule(devKey):
                maybeKernelName = cudaq_runtime.checkRegisteredCppDeviceKernel(
                    self.module, devKey + '.' + node.func.attr)
                if maybeKernelName == None:
                    maybeKernelName = cudaq_runtime.checkRegisteredCppDeviceKernel(
                        self.module, devKey)
                if maybeKernelName != None:
                    otherKernel = SymbolTable(
                        self.module.operation)[maybeKernelName]
                    res = processFunctionCall(otherKernel.type)
                    if res is not None:
                        self.pushValue(res)
                    return

            # Start by seeing if we have mod1.mod2.mod3...
            moduleNames = []
            value = node.func.value
            while isinstance(value, ast.Attribute):
                self.debug_msg(lambda: f'[(Inline) Visit Attribute]', value)
                moduleNames.append(value.attr)
                value = value.value
                if isinstance(value, ast.Name):
                    self.debug_msg(lambda: f'[(Inline) Visit Name]', value)
                    moduleNames.append(value.id)
                    break

            if all(x in moduleNames for x in ['cudaq', 'dbg', 'ast']):
                # Handle a debug print statement
                arg = self.__groupValues(node.args, [1])
                self.__insertDbgStmt(arg, node.func.attr)
                return

            # If we did have module names, then this is what we are looking for
            if len(moduleNames):
                name = node.func.attr
                # FIXME: We should be properly dealing with modules and submodules...
                if name in globalKernelRegistry:
                    # If it is in `globalKernelRegistry`, it has to be in this Module
                    otherKernel = SymbolTable(
                        self.module.operation)[nvqppPrefix + name]
                    
                    res = processFunctionCall(otherKernel.type)
                    if res is not None:
                        self.pushValue(res)
                    return

        if isinstance(node.func, ast.Name):

            namedArgs = {}
            for keyword in node.keywords:
                self.visit(keyword.value)
                # FIXME: ADD TEST TO MAKE SURE KW ARGS DON'T RESULT IN UNPROCESSED VALS
                namedArgs[keyword.arg] = self.popValue()

            self.debug_msg(lambda: f'[(Inline) Visit Name]', node.func)
            if node.func.id == 'len':
                listVal = self.__groupValues(node.args, [1])
                listVal = self.ifPointerThenLoad(listVal)

                # FIXME: split out into helper function
                if cc.StdvecType.isinstance(listVal.type):
                    self.pushValue(
                        cc.StdvecSizeOp(self.getIntegerType(), listVal).result)
                    return
                if quake.VeqType.isinstance(listVal.type):
                    self.pushValue(
                        quake.VeqSizeOp(self.getIntegerType(), listVal).result)
                    return

                self.emitFatalError(
                    "__len__ not supported on variables of this type.", node)

            if node.func.id == 'range':
                startVal, endVal, stepVal, isDecrementing = self.__processRangeLoopIterationBounds(
                    node.args)

                iTy = self.getIntegerType(64)
                zero = arith.ConstantOp(iTy, IntegerAttr.get(iTy, 0))
                one = arith.ConstantOp(iTy, IntegerAttr.get(iTy, 1))

                totalSize = arith.SubIOp(endVal, startVal).result
                if isDecrementing:
                    roundingOffset = arith.AddIOp(stepVal, one)
                else:
                    roundingOffset = arith.SubIOp(stepVal, one)
                totalSize = arith.AddIOp(totalSize, roundingOffset)
                totalSize = arith.MaxSIOp(zero, arith.DivSIOp(totalSize, stepVal).result).result

                # Create an array of i64 of the total size
                arrTy = cc.ArrayType.get(iTy)
                iterable = cc.AllocaOp(cc.PointerType.get(arrTy),
                                       TypeAttr.get(iTy),
                                       seqSize=totalSize).result

                # Logic here is as follows:
                # We are building an array like this
                # array = [start, start +- step, start +- 2*step, start +- 3*step, ...]
                # So we need to know the start and step (already have them),
                # but we also need to keep track of a counter
                counter = cc.AllocaOp(cc.PointerType.get(iTy),
                                      TypeAttr.get(iTy)).result
                cc.StoreOp(zero, counter)

                def bodyBuilder(iterVar):
                    loadedCounter = cc.LoadOp(counter).result
                    eleAddr = cc.ComputePtrOp(
                        cc.PointerType.get(iTy), iterable, [loadedCounter],
                        DenseI32ArrayAttr.get([kDynamicPtrIndex],
                                              context=self.ctx))
                    cc.StoreOp(iterVar, eleAddr)
                    incrementedCounter = arith.AddIOp(loadedCounter, one).result
                    cc.StoreOp(incrementedCounter, counter)

                self.createMonotonicForLoop(bodyBuilder, 
                                   startVal=startVal,
                                   stepVal=stepVal,
                                   endVal=endVal,
                                   isDecrementing=isDecrementing)

                vect = cc.StdvecInitOp(cc.StdvecType.get(iTy), iterable, length=totalSize).result
                self.pushValue(vect)
                return

            if node.func.id == 'enumerate':
                # We have to have something "iterable" on the stack,
                # could be coming from `range()` or an iterable like `qvector`
                iterable = self.__groupValues(node.args, [1])
                iterable = self.ifPointerThenLoad(iterable)

                # FIXME: leverage visit_for instead
                
                # Create a new iterable, `alloca cc.struct<i64, T>`
                if quake.VeqType.isinstance(iterable.type):
                    iterEleTy = self.getRefType()
                    totalSize = quake.VeqSizeOp(self.getIntegerType(),
                                                iterable).result

                    def extractFunctor(idxVal):
                        return quake.ExtractRefOp(iterEleTy,
                                                    iterable,
                                                    -1,
                                                    index=idxVal).result
                elif cc.StdvecType.isinstance(iterable.type):
                    iterEleTy = cc.StdvecType.getElementType(iterable.type)
                    totalSize = cc.StdvecSizeOp(self.getIntegerType(),
                                                iterable).result

                    def extractFunctor(idxVal):
                        arrEleTy = cc.ArrayType.get(iterEleTy)
                        elePtrTy = cc.PointerType.get(iterEleTy)
                        arrPtrTy = cc.PointerType.get(arrEleTy)
                        vecPtr = cc.StdvecDataOp(arrPtrTy, iterable).result
                        eleAddr = cc.ComputePtrOp(
                            elePtrTy, vecPtr, [idxVal],
                            DenseI32ArrayAttr.get([kDynamicPtrIndex],
                                                    context=self.ctx)).result
                        return cc.LoadOp(eleAddr).result
                else:
                    self.emitFatalError(
                        "could not infer enumerate tuple type ({})".format(
                            iterable.type), node)

                # Enumerate returns a iterable of tuple(i64, T) for type T
                # Allocate an array of struct<i64, T> == tuple (for us)
                structTy = cc.StructType.get([self.getIntegerType(), iterEleTy])
                enumIterable = cc.AllocaOp(cc.PointerType.get(cc.ArrayType.get(structTy)),
                                           TypeAttr.get(structTy),
                                           seqSize=totalSize).result

                # Now we need to loop through `enumIterable` and set the elements
                def bodyBuilder(iterVar):
                    # Create the struct
                    element = cc.UndefOp(structTy)
                    # Get the element from the iterable
                    extracted = extractFunctor(iterVar)
                    # Get the pointer to the enumeration iterable so we can set it
                    eleAddr = cc.ComputePtrOp(
                        cc.PointerType.get(structTy), enumIterable, [iterVar],
                        DenseI32ArrayAttr.get([kDynamicPtrIndex],
                                              context=self.ctx))
                    # Set the index value
                    element = cc.InsertValueOp(
                        structTy, element, iterVar,
                        DenseI64ArrayAttr.get([0], context=self.ctx)).result
                    # Set the extracted element value
                    element = cc.InsertValueOp(
                        structTy, element, extracted,
                        DenseI64ArrayAttr.get([1], context=self.ctx)).result
                    cc.StoreOp(element, eleAddr)

                self.createInvariantForLoop(bodyBuilder, totalSize)
                vect = cc.StdvecInitOp(cc.StdvecType.get(structTy), enumIterable, length=totalSize).result
                self.pushValue(vect)
                return

            if node.func.id == 'complex':
                if len(namedArgs) == 0:
                    real, imag = self.__groupValues(node.args, [2])
                else:
                    # FIXME: NEED TO CHECK AND POP SPARE VALUES...
                    imag = namedArgs['imag']
                    real = namedArgs['real']

                imag = self.changeOperandToType(self.getFloatType(), imag)
                real = self.changeOperandToType(self.getFloatType(), real)
                self.pushValue(
                    complex.CreateOp(self.getComplexType(), real, imag).result)
                return

            if self.__isSimpleGate(node.func.id):
                processQuakeCtor(node.func.id.title(), node.args, isCtrl=False, isAdj=False)
                return

            if self.__isAdjointSimpleGate(node.func.id):
                processQuakeCtor(node.func.id[0].title(), node.args, isCtrl=False, isAdj=True)
                return

            if self.__isControlledSimpleGate(node.func.id):
                # FIXME: ADD TESTS FOR MULTIPLE CONTROLS HERE
                processQuakeCtor(node.func.id[1:].title(), node.args, isCtrl=True, isAdj=False)
                return

            if self.__isRotationGate(node.func.id):
                processQuakeCtor(node.func.id.title(), node.args, isCtrl=False, isAdj=False, numParams = 1)
                return

            if self.__isControlledRotationGate(node.func.id):
                processQuakeCtor(node.func.id[1:].title(), node.args, isCtrl=True, isAdj=False, numParams = 1)
                return

            if self.__isMeasurementGate(node.func.id):
                registerName = self.currentAssignVariableName
                # If `registerName` is None, then we know that we
                # are not assigning this measure result to anything
                # so we therefore should not push it on the stack
                pushResultToStack = registerName != None or self.walkingReturnNode

                # By default we set the `register_name` for the measurement
                # to the assigned variable name (if there is one). But
                # the use could have manually specified `register_name='something'`
                # check for that here and use it there
                if len(node.keywords) == 1 and hasattr(node.keywords[0], 'arg'):
                    if node.keywords[0].arg == 'register_name':
                        userProvidedRegName = node.keywords[0]
                        if not isinstance(userProvidedRegName.value,
                                          ast.Constant):
                            self.emitFatalError(
                                "measurement register_name keyword must be a constant string literal.",
                                node)
                        self.debug_msg(lambda: f'[(Inline) Visit Constant]',
                                       userProvidedRegName.value)
                        registerName = userProvidedRegName.value.value

                qubits = self.__groupValues(node.args, [(1, -1)])
                label = registerName or None
                if len(qubits) == 1 and quake.RefType.isinstance(qubits[0].type):
                    measTy = quake.MeasureType.get()
                    resTy = self.getIntegerType(1)
                else:
                    measTy = cc.StdvecType.get(quake.MeasureType.get())
                    resTy = cc.StdvecType.get(self.getIntegerType(1))
                measureResult = processQuantumOperation(node.func.id.title(), [], qubits, measTy, broadcast=False, registerName=label).result

                # FIXME: needs to be revised when we properly distinguish measurement types
                if pushResultToStack:
                    self.pushValue(
                        quake.DiscriminateOp(resTy, measureResult).result)
                return

            if node.func.id == 'swap':
                processQuakeCtor(node.func.id.title(), node.args, isCtrl=False, isAdj=False, numTargets = 2)
                return

            if node.func.id == 'reset':
                targets = self.__groupValues(node.args, [(1, -1)])
                processQuantumOperation(node.func.id.title(), [], targets, broadcast= lambda q: q)
                return

            if node.func.id == 'u3':
                processQuakeCtor(node.func.id.title(), node.args, isCtrl=False, isAdj=False, numParams = 3)
                return

            if node.func.id == 'exp_pauli':
                # Note: C++ also has a constructor that takes an `f64`, `string`,
                # any any number of qubits. We don't support this here.
                theta, target, pauliWord =  self.__groupValues(node.args, [1, 1, 1])
                theta = self.changeOperandToType(self.getFloatType(), theta)
                processQuantumOperation("ExpPauli", [], [target], [], [theta], broadcast=False, pauli=pauliWord)
                return

            if node.func.id in globalRegisteredOperations:
                unitary = globalRegisteredOperations[node.func.id]
                numTargets = int(np.log2(np.sqrt(unitary.size)))
                targets = self.__groupValues(node.args, [(numTargets, numTargets)])

                for i, t in enumerate(targets):
                    if not quake.RefType.isinstance(t.type):
                        self.emitFatalError(
                            f'invalid target operand {i}, broadcasting is not supported on custom operations.'
                        )

                globalName = f'{nvqppPrefix}{node.func.id}_generator_{numTargets}.rodata'

                currentST = SymbolTable(self.module.operation)
                if not globalName in currentST:
                    with InsertionPoint(self.module.body):
                        gen_vector_of_complex_constant(self.loc, self.module,
                                                       globalName,
                                                       unitary.tolist())
                quake.CustomUnitarySymbolOp(
                    [],
                    generator=FlatSymbolRefAttr.get(globalName),
                    parameters=[],
                    controls=[],
                    targets=targets,
                    is_adj=False)
                return

            # Handle the case where we are capturing an opaque kernel
            # function. It has to be in the capture vars and it has to
            # be a PyKernelDecorator.
            if node.func.id in self.capturedVars and node.func.id not in globalKernelRegistry:
                from .kernel_decorator import PyKernelDecorator
                var = self.capturedVars[node.func.id]
                if isinstance(var, PyKernelDecorator):
                    # If we found it, then compile its ASTModule to MLIR so
                    # that it is in the proper registries, then give it
                    # the proper function alias
                    PyASTBridge(var.capturedDataStorage,
                                existingModule=self.module,
                                locationOffset=var.location).visit(
                                    var.astModule)
                    # If we have an alias, make sure we point back to the
                    # kernel registry correctly for the next conditional check
                    if var.name in globalKernelRegistry:
                        node.func.id = var.name

            if node.func.id in globalKernelRegistry:
                # If in `globalKernelRegistry`, it has to be in this Module
                otherKernel = SymbolTable(self.module.operation)[nvqppPrefix +
                                                                 node.func.id]

                res = processFunctionCall(otherKernel.type)
                if res is not None:
                    self.pushValue(res)
                return

            elif node.func.id in self.symbolTable:
                val = self.symbolTable[node.func.id]
                if cc.CallableType.isinstance(val.type):
                    callableTy = cc.CallableType.getFunctionType(val.type)
                    funcTy = FunctionType(callableTy)
                    numArgs = len(funcTy.inputs)
                    values = self.__groupValues(node.args, [(numArgs, numArgs)])
                    values = convertArguments(funcTy.inputs, values)
                    callable = cc.CallableFuncOp(callableTy, val).result
                    func.CallIndirectOp([], callable, values)
                    return

                self.emitFatalError(
                    f"`{node.func.id}` object is not callable, found symbol of type {val.type}",
                    node)

            elif node.func.id == 'int':
                # cast operation
                value = self.__groupValues(node.args, [1])
                casted = self.changeOperandToType(IntegerType.get_signless(64),
                                                  value,
                                                  allowDemotion=True)
                self.pushValue(casted)
                return

            elif node.func.id == 'list':
                value = self.__groupValues(node.args, [1])
                valueTy = value.type
                if cc.PointerType.isinstance(valueTy):
                    valueTy = cc.PointerType.getElementType(valueTy)
                if not cc.StdvecType.isinstance(valueTy):
                    self.emitFatalError('Invalid list() cast requested.', node)

                self.pushValue(value)
                return

            elif node.func.id in ['print_i64', 'print_f64']:
                value = self.__groupValues(node.args, [1])
                self.__insertDbgStmt(value, node.func.id)
                return

            elif node.func.id in globalRegisteredTypes.classes:
                # Handle User-Custom Struct Constructor
                cls, annotations = globalRegisteredTypes.getClassAttributes(
                    node.func.id)

                if '__slots__' not in cls.__dict__:
                    self.emitWarning(
                        f"Adding new fields in data classes is not yet supported. The dataclass must be declared with @dataclass(slots=True) or @dataclasses.dataclass(slots=True).",
                        node)

                # Alloca the struct
                structTys = [
                    mlirTypeFromPyType(v, self.ctx)
                    for _, v in annotations.items()
                ]

                structTy = mlirTryCreateStructType(structTys,
                                                   name=node.func.id,
                                                   context=self.ctx)
                if structTy is None:
                    self.emitFatalError(
                        "Hybrid quantum-classical data types and nested quantum structs are not allowed.",
                        node)

                # Disallow user specified methods on structs
                if len({
                        k: v
                        for k, v in cls.__dict__.items()
                        if not (k.startswith('__') and k.endswith('__')) and
                        isinstance(v, FunctionType)
                }) != 0:
                    self.emitFatalError(
                        'struct types with user specified methods are not allowed.',
                        node)

                if node.keywords:
                    self.emitFatalError("keyword arguments for data classes are not yet supported", node)

                numArgs = len(structTys)
                ctorArgs = self.__groupValues(node.args, [(numArgs, numArgs)])
                ctorArgs = convertArguments(structTys, ctorArgs)

                if quake.StruqType.isinstance(structTy):
                    # If we have a quantum struct. We cannot allocate classical
                    # memory and load / store quantum type values to that memory
                    # space, so use `quake.MakeStruqOp`.
                    self.pushValue(quake.MakeStruqOp(structTy, ctorArgs).result)
                    return

                stackSlot = cc.AllocaOp(cc.PointerType.get(structTy),
                                        TypeAttr.get(structTy)).result

                # loop over each type and `compute_ptr` / store
                for i, ty in enumerate(structTys):
                    eleAddr = cc.ComputePtrOp(
                        cc.PointerType.get(ty), stackSlot, [],
                        DenseI32ArrayAttr.get([i], context=self.ctx)).result
                    cc.StoreOp(ctorArgs[i], eleAddr)
                self.pushValue(stackSlot)
                return

            else:
                self.emitFatalError(
                    "unhandled function call - {}, known kernels are {}".format(
                        node.func.id, globalKernelRegistry.keys()), node)

        elif isinstance(node.func, ast.Attribute):
            self.debug_msg(lambda: f'[(Inline) Visit Attribute]', node.func)
            self.debug_msg(lambda: f'[(Inline) Visit Name]', node.func.value)

            if node.func.attr == 'size':
                if len(node.args) != 0:
                    self.emitFatalError("'size' does not support an argument",
                                        node)
                # Handled in the Attribute visit,
                # since `numpy` arrays have a size attribute
                self.visit(node.func)
                self.pushValue(self.popValue())
                return
            
            if node.func.attr == 'copy':
                self.visit(node.func.value)
                funcVal = self.ifPointerThenLoad(self.popValue())

                if cc.StructType.isinstance(funcVal.type):
                    slot = cc.AllocaOp(cc.PointerType.get(funcVal.type), TypeAttr.get(funcVal.type)).result
                    cc.StoreOp(funcVal, slot)                    
                    self.pushValue(slot)
                    return

                self.emitFatalError(f'unsupported function {node.func.attr}',
                                    node)

            if self.__isSupportedVectorFunction(node.func.attr):

                # This means we are visiting this node twice -
                # once in visit_Attribute, once here. But unless
                # we make the functions we support on values explicit
                # somewhere, there is no way around that.
                self.visit(node.func.value)
                funcVal = self.ifPointerThenLoad(self.popValue())

                # Just to be nice and give a dedicated error.
                if node.func.attr == 'append' and \
                    (quake.VeqType.isinstance(funcVal.type) or cc.StdvecType.isinstance(funcVal.type)):
                    self.emitFatalError(
                        "CUDA-Q does not allow dynamic resizing or lists, arrays, or qvectors.",
                        node)

                # Neither Python lists nor `numpy` arrays have a function
                # or attribute 'front'/'back'; hence we only support that
                # for `qvectors`.
                if not quake.VeqType.isinstance(funcVal.type):
                    self.emitFatalError(
                        f'function {node.func.attr} is not supported on a value of type {funcVal.type}',
                        node)

                funcArg = None
                args = self.__groupValues(node.args, [(0, 1)])
                if args:
                    funcArg = self.ifPointerThenLoad(args[0])
                    if not IntegerType.isinstance(funcArg.type):
                        self.emitFatalError(
                            f'expecting an integer argument for call to {node.func.attr}',
                            node)

                # `qreg` or `qview` method call
                if node.func.attr == 'back':
                    qrSize = quake.VeqSizeOp(self.getIntegerType(),
                                             funcVal).result
                    one = self.getConstantInt(1)
                    endOff = arith.SubIOp(qrSize, one)
                    if funcArg is None:
                        # extract the qubit...
                        self.pushValue(
                            quake.ExtractRefOp(self.getRefType(),
                                               funcVal,
                                               -1,
                                               index=endOff).result)
                    else:
                        # extract the `subveq`
                        startOff = arith.SubIOp(qrSize, funcArg)
                        dyna = IntegerAttr.get(self.getIntegerType(), -1)
                        self.pushValue(
                            quake.SubVeqOp(self.getVeqType(),
                                           funcVal,
                                           dyna,
                                           dyna,
                                           lower=startOff,
                                           upper=endOff).result)
                    return

                if node.func.attr == 'front':
                    zero = self.getConstantInt(0)
                    if funcArg is None:
                        # extract the qubit...
                        self.pushValue(
                            quake.ExtractRefOp(self.getRefType(),
                                               funcVal,
                                               -1,
                                               index=zero).result)
                    else:
                        # extract the `subveq`
                        one = self.getConstantInt(1)
                        offset = arith.SubIOp(funcArg, one)
                        dyna = IntegerAttr.get(self.getIntegerType(), -1)
                        self.pushValue(
                            quake.SubVeqOp(self.getVeqType(),
                                           funcVal,
                                           dyna,
                                           dyna,
                                           lower=zero,
                                           upper=offset).result)
                    return

                # To make sure we at least have a proper error if we have
                # any mismatch between what is implemented, vs what's listed in
                # __isSupportedVectorFunction.
                self.emitFatalError(f'unsupported function {node.func.attr}',
                                    node)

            if isinstance(node.func.value, ast.Name):

                if node.func.value.id in ['numpy', 'np']:

                    value = self.__groupValues(node.args, [1])
                    namedArgs = {}
                    for keyword in node.keywords:
                        # FIXME: CHECK THAT WE DON'T HAVE SPARE VALUES
                        # (OR REMOVE THIS, SINCE WE ONLY EVER ACCEPT A SINGLE ARGUMENT?)
                        self.visit(keyword.value)
                        namedArgs[keyword.arg] = self.popValue()

                    if node.func.attr == 'array':
                        # `np.array(vec, <dtype = ty>)`
                        valueTy = value.type
                        if cc.PointerType.isinstance(value.type):
                            valueTy = cc.PointerType.getElementType(
                                value.type)

                        if not cc.StdvecType.isinstance(valueTy):
                            raise self.emitFatalError(
                                f"unexpected numpy array initializer type: {valueTy}",
                                node)

                        eleTy = cc.StdvecType.getElementType(valueTy)
                        dTy = namedArgs['dtype'] if 'dtype' in namedArgs else eleTy

                        # Convert the vector to the provided data type if needed.
                        self.pushValue(
                            self.__copyVectorAndCastElements(
                                value, dTy, allowDemotion=True))
                        return

                    value = self.ifPointerThenLoad(value)

                    if node.func.attr in ['complex128', 'complex64']:
                        if node.func.attr == 'complex128':
                            ty = self.getComplexType()
                        if node.func.attr == 'complex64':
                            ty = self.getComplexType(width=32)

                        value = self.changeOperandToType(ty, value)
                        self.pushValue(value)
                        return

                    if node.func.attr in ['float64', 'float32']:
                        if node.func.attr == 'float64':
                            ty = self.getFloatType(width=64)
                        if node.func.attr == 'float32':
                            ty = self.getFloatType(width=32)

                        value = self.changeOperandToType(ty, value)
                        self.pushValue(value)
                        return

                    if node.func.attr in ['int64', 'int32']:
                        if node.func.attr == 'int64':
                            ty = self.getIntegerType(width=64)
                        if node.func.attr == 'int32':
                            ty = self.getIntegerType(width=32)

                        value = self.changeOperandToType(ty, value)
                        self.pushValue(value)
                        return

                    # Promote argument's types for `numpy.func` calls to match python's semantics
                    if self.__isSupportedNumpyFunction(node.func.attr):
                        if ComplexType.isinstance(value.type):
                            value = self.changeOperandToType(
                                self.getComplexType(), value)
                        elif IntegerType.isinstance(value.type):
                            value = self.changeOperandToType(
                                self.getFloatType(), value)
                        elif not F64Type.isinstance(value.type) and \
                            not F32Type.isinstance(value.type):
                            self.emitFatalError(
                                "invalid type {} for call to numpy function {}".
                                format(value.type, node.func.attr), node)

                    if node.func.attr == 'cos':
                        if ComplexType.isinstance(value.type):
                            self.pushValue(complex.CosOp(value).result)
                            return
                        self.pushValue(math.CosOp(value).result)
                        return
                    if node.func.attr == 'sin':
                        if ComplexType.isinstance(value.type):
                            self.pushValue(complex.SinOp(value).result)
                            return
                        self.pushValue(math.SinOp(value).result)
                        return
                    if node.func.attr == 'sqrt':
                        if ComplexType.isinstance(value.type):
                            self.pushValue(complex.SqrtOp(value).result)
                            return
                        self.pushValue(math.SqrtOp(value).result)
                        return
                    if node.func.attr == 'exp':
                        if ComplexType.isinstance(value.type):
                            # Note: using `complex.ExpOp` results in a
                            # "can't legalize `complex.exp`" error.
                            # Using Euler's' formula instead:
                            #
                            # "e^(x+i*y) = (e^x) * (cos(y)+i*sin(y))"
                            complexType = ComplexType(value.type)
                            floatType = complexType.element_type
                            real = complex.ReOp(value).result
                            imag = complex.ImOp(value).result
                            left = self.changeOperandToType(
                                complexType,
                                math.ExpOp(real).result)
                            re2 = math.CosOp(imag).result
                            im2 = math.SinOp(imag).result
                            right = complex.CreateOp(ComplexType.get(floatType),
                                                     re2, im2).result
                            res = complex.MulOp(left, right).result
                            self.pushValue(res)
                            return
                        self.pushValue(math.ExpOp(value).result)
                        return
                    if node.func.attr == 'ceil':
                        if ComplexType.isinstance(value.type):
                            self.emitFatalError(
                                f"numpy call ({node.func.attr}) is not supported for complex numbers",
                                node)
                            return
                        self.pushValue(math.CeilOp(value).result)
                        return

                    self.emitFatalError(
                        f"unsupported NumPy call ({node.func.attr})", node)

                if node.func.value.id == 'cudaq':
                    if node.func.attr == 'complex':
                        self.__groupValues(node.args, [0])
                        self.pushValue(self.simulationDType())
                        return

                    if node.func.attr == 'amplitudes':
                        value = self.__groupValues(node.args, [1])

                        valueTy = value.type
                        if cc.PointerType.isinstance(value.type):
                            valueTy = cc.PointerType.getElementType(
                                value.type)
                        if cc.StdvecType.isinstance(valueTy):
                            self.pushValue(value)
                            return

                        self.emitFatalError(
                            f"unsupported amplitudes argument type: {value.type}",
                            node)

                    if node.func.attr == 'qvector':
                        if len(node.args) == 0:
                            self.emitFatalError(
                                'qvector does not have default constructor. Init from size or existing state.',
                                node)

                        valueOrPtr = self.__groupValues(node.args, [1])
                        initializerTy = valueOrPtr.type

                        if cc.PointerType.isinstance(initializerTy):
                            initializerTy = cc.PointerType.getElementType(
                                initializerTy)

                        if (IntegerType.isinstance(initializerTy)):
                            # handle `cudaq.qvector(n)`
                            value = self.ifPointerThenLoad(valueOrPtr)
                            ty = self.getVeqType()
                            qubits = quake.AllocaOp(ty, size=value).result
                            self.pushValue(qubits)
                            return

                        if cc.StdvecType.isinstance(initializerTy):
                            # handle `cudaq.qvector(initState)`

                            # Validate the length in case of a constant initializer:
                            # `cudaq.qvector([1., 0., ...])`
                            # `cudaq.qvector(np.array([1., 0., ...]))`
                            value = self.ifPointerThenLoad(valueOrPtr)
                            listScalar = None
                            arrNode = node.args[0]
                            if isinstance(arrNode, ast.List):
                                listScalar = arrNode.elts

                            if isinstance(arrNode, ast.Call) and isinstance(
                                    arrNode.func, ast.Attribute):
                                if arrNode.func.value.id in [
                                        'numpy', 'np'
                                ] and arrNode.func.attr == 'array':
                                    lst = node.args[0].args[0]
                                    if isinstance(lst, ast.List):
                                        listScalar = lst.elts

                            if listScalar != None:
                                size = len(listScalar)
                                numQubits = np.log2(size)
                                if not numQubits.is_integer():
                                    self.emitFatalError(
                                        "Invalid input state size for qvector init (not a power of 2)",
                                        node)

                            eleTy = cc.StdvecType.getElementType(value.type)
                            size = cc.StdvecSizeOp(self.getIntegerType(),
                                                   value).result
                            numQubits = math.CountTrailingZerosOp(size).result

                            # TODO: Dynamically check if number of qubits is power of 2
                            # and if the state is normalized

                            arrTy = cc.ArrayType.get(eleTy)
                            ptrArrTy = cc.PointerType.get(arrTy)
                            veqTy = quake.VeqType.get()

                            qubits = quake.AllocaOp(veqTy,
                                                    size=numQubits).result
                            data = cc.StdvecDataOp(ptrArrTy, value).result
                            init = quake.InitializeStateOp(veqTy, qubits,
                                                           data).result
                            self.pushValue(init)
                            return

                        if cc.StateType.isinstance(initializerTy):
                            # handle `cudaq.qvector(state)`
                            statePtr = self.ifNotPointerThenStore(valueOrPtr)

                            i64Ty = self.getIntegerType()
                            numQubits = quake.GetNumberOfQubitsOp(
                                i64Ty, statePtr).result

                            veqTy = quake.VeqType.get()
                            qubits = quake.AllocaOp(veqTy,
                                                    size=numQubits).result
                            init = quake.InitializeStateOp(
                                veqTy, qubits, statePtr).result

                            self.pushValue(init)
                            return

                        self.emitFatalError(
                            f"unsupported qvector argument type: {value.type}",
                            node)

                    if node.func.attr == "qubit":
                        if len(node.args) != 0:
                            self.emitFatalError(
                                'cudaq.qubit() constructor does not take any arguments. To construct a vector of qubits, use `cudaq.qvector(N)`.'
                            )
                        self.pushValue(quake.AllocaOp(self.getRefType()).result)
                        return

                    if node.func.attr == 'adjoint' or node.func.attr == 'control':

                        # NOTE: We currently generally don't have the means in the
                        # compiler to handle composition of control and adjoint, since
                        # control and adjoint are not proper functors (i.e. there is
                        # no way to obtain a new callable object that is the adjoint
                        # or controlled version of another callable).
                        # Since we don't really treat callables as first-class values,
                        # the first argument to control and adjoint indeed has to be
                        # a Name object.
                        if not node.args or not isinstance(node.args[0], ast.Name):
                            self.emitFatalError(
                                f'unsupported argument in call to {node.func.attr} - first argument must be a symbol name',
                                node)
                        otherFuncName = node.args[0].id
                        kwargs = {"is_adj": node.func.attr == 'adjoint'}

                        if otherFuncName in self.symbolTable:
                            self.visit(node.args[0])
                            fctArg = self.popValue()
                            if not cc.CallableType.isinstance(fctArg.type):
                                self.emitFatalError(f"{otherFuncName} is not a quantum kernel", node)
                            functionTy = FunctionType(
                                cc.CallableType.getFunctionType(fctArg.type))
                            inputTys, outputTys = functionTy.inputs, functionTy.results
                            indirectCallee = [fctArg]
                        elif otherFuncName in globalKernelRegistry:
                            otherFunc = globalKernelRegistry[otherFuncName]
                            inputTys, outputTys = otherFunc.arguments.types, otherFunc.results.types
                            indirectCallee = []
                            kwargs["callee"] = FlatSymbolRefAttr.get(nvqppPrefix +
                                                                    otherFuncName)
                        elif otherFuncName in globalRegisteredOperations:
                            self.emitFatalError(
                                f"calling cudaq.control or cudaq.adjoint on a globally registered operation is not supported",
                                node)
                        elif self.__isUnitaryGate(
                                otherFuncName) or self.__isMeasurementGate(otherFuncName):
                            self.emitFatalError(
                                f"calling cudaq.control or cudaq.adjoint on a built-in gate is not supported",
                                node)
                        else:
                            self.emitFatalError(
                                f"{otherFuncName} is not a known quantum kernel - maybe a cudaq.kernel attribute is missing?.",
                                node)

                        numArgs = len(inputTys)
                        invert_controls = lambda: None
                        if node.func.attr == 'control':
                            # FIXME: CHECK MULTIPLE CONTROLS
                            controls, args = self.__groupValues(
                                node.args[1:], [(1, -1), (numArgs, numArgs)])
                            qvec_or_qubits = all((quake.RefType.isinstance(v.type) for v in controls)) or \
                                (len(controls) == 1 and quake.VeqType.isinstance(controls[0].type))
                            if not qvec_or_qubits:
                                self.emitFatalError(
                                    f'invalid argument type for control operand', node)
                            # TODO: it would be cleaner to add support for negated control
                            # qubits to `quake.ApplyOp`
                            negatedControlQubits = self.controlNegations.copy()
                            self.controlNegations.clear()
                            if negatedControlQubits:
                                invert_controls = lambda: processQuantumOperation('X', [], negatedControlQubits, [], [])                                
                        else:
                            controls, args = self.__groupValues(
                                node.args[1:], [(0, 0), (numArgs, numArgs)])

                        args = convertArguments(inputTys, args)
                        if len(outputTys) != 0:
                            self.emitFatalError(
                                f'cannot take {node.func.attr} of kernel {otherFuncName} that returns a value',
                                node)
                        invert_controls()
                        quake.ApplyOp([], indirectCallee, controls, args, **kwargs)
                        invert_controls()
                        return

                    if node.func.attr == 'apply_noise':

                        # The first argument must be the Kraus channel
                        # TODO: I AM NOT SURE CUSTOM REGISTERED CHANNELS WERE 
                        # EVER PROPERLY PROCESSED BY THE BRIDGE...
                        # FIXME check we have at least one arg

                        supportedChannels = [
                            'DepolarizationChannel', 'AmplitudeDampingChannel',
                            'PhaseFlipChannel', 'BitFlipChannel', 'PhaseDamping',
                            'ZError', 'XError', 'YError', 'Pauli1', 'Pauli2',
                            'Depolarization1', 'Depolarization2'
                        ]
                        if isinstance(node.args[0], ast.Attribute) and \
                            node.args[0].value.id == 'cudaq' and \
                            node.args[0].attr in supportedChannels:

                            cudaq_module = importlib.import_module('cudaq')
                            channel_class = getattr(cudaq_module, node.args[0].attr)
                            numParams = channel_class.num_parameters
                            key = self.getConstantInt(hash(channel_class))
                        else:
                            self.emitFatalError("currently, only built-in channels are supported for apply_noise", node)
                        
                        # This currently requires at least one qubit argument
                        params, values = self.__groupValues(node.args[1:], [(numParams, numParams), (1, -1)])
                        checkControlAndTargetTypes([], values)

                        for i, p in enumerate(params):
                            # If we have a F64 value, we want to
                            # store it to a pointer
                            if F64Type.isinstance(p.type):
                                alloca = cc.AllocaOp(cc.PointerType.get(p.type),
                                                     TypeAttr.get(
                                                         p.type)).result
                                cc.StoreOp(p, alloca)
                                params[i] = alloca

                        asVeq = quake.ConcatOp(self.getVeqType(), values).result
                        quake.ApplyNoiseOp(params, [asVeq], key=key)
                        return

                    if node.func.attr == 'compute_action':
                        compute, action = self.__groupValues(node.args, [2])
                        quake.ComputeActionOp(compute, action)
                        return

                    self.emitFatalError(
                        f'Invalid function or class type requested from the cudaq module ({node.func.attr})',
                        node)

                def maybeProposeOpAttrFix(opName, attrName):
                    """
                    Check the quantum operation attribute name and propose a smart
                    fix message if possible. For example, if we have
                    `x.control(...)` then remind the programmer the correct
                    attribute is `x.ctrl(...)`.
                    """
                    # TODO Add more possibilities in the future...
                    if attrName in [
                            'control'
                    ] or 'control' in attrName or 'ctrl' in attrName:
                        return f'Did you mean {opName}.ctrl(...)?'

                    if attrName in [
                            'adjoint'
                    ] or 'adjoint' in attrName or 'adj' in attrName:
                        return f'Did you mean {opName}.adj(...)?'

                    return ''

                # We have a `func_name.ctrl`
                if self.__isSimpleGate(node.func.value.id):
                    if node.func.attr == 'ctrl':
                        processQuakeCtor(node.func.value.id.title(), node.args, isCtrl=True, isAdj=False)
                        return
                    if node.func.attr == 'adj':
                        processQuakeCtor(node.func.value.id.title(), node.args, isCtrl=False, isAdj=True)
                        return
                    self.emitFatalError(
                        f'Unknown attribute on quantum operation {node.func.value.id} ({node.func.attr}). {maybeProposeOpAttrFix(node.func.value.id, node.func.attr)}'
                    )

                if self.__isRotationGate(node.func.value.id):
                    if node.func.attr == 'ctrl':
                        processQuakeCtor(node.func.value.id.title(), node.args, isCtrl=True, isAdj=False, numParams = 1)
                        return
                    if node.func.attr == 'adj':
                        processQuakeCtor(node.func.value.id.title(), node.args, isCtrl=False, isAdj=True, numParams = 1)
                        return
                    self.emitFatalError(
                        f'Unknown attribute on quantum operation {node.func.value.id} ({node.func.attr}). {maybeProposeOpAttrFix(node.func.value.id, node.func.attr)}'
                    )

                if node.func.value.id == 'swap' and node.func.attr == 'ctrl':
                    processQuakeCtor(node.func.value.id.title(), node.args, isCtrl=True, isAdj=False, numTargets = 2)
                    return

                if node.func.value.id == 'u3':
                    if node.func.attr == 'ctrl':
                        processQuakeCtor(node.func.value.id.title(), node.args, isCtrl=True, isAdj=False, numParams = 3)
                        return
                    if node.func.attr == 'adj':
                        processQuakeCtor(node.func.value.id.title(), node.args, isCtrl=False, isAdj=True, numParams = 3)
                        return
                    self.emitFatalError(
                        f'unknown attribute {node.func.attr} on u3', node)

                # custom `ctrl` and `adj`
                if node.func.value.id in globalRegisteredOperations:
                    if not node.func.attr == 'ctrl' and not node.func.attr == 'adj':
                        self.emitFatalError(
                            f'Unknown attribute on custom operation {node.func.value.id} ({node.func.attr}).'
                        )

                    unitary = globalRegisteredOperations[node.func.value.id]
                    numTargets = int(np.log2(np.sqrt(unitary.size)))
                    globalName = f'{nvqppPrefix}{node.func.value.id}_generator_{numTargets}.rodata'
                    currentST = SymbolTable(self.module.operation)
                    if not globalName in currentST:
                        with InsertionPoint(self.module.body):
                            gen_vector_of_complex_constant(
                                self.loc, self.module, globalName,
                                unitary.tolist())

                    if node.func.attr == 'ctrl':
                        controls, targets = self.__groupValues(node.args, [(1, -1), (numTargets, numTargets)])
                        negatedControlQubits = getNegatedControlQubits(controls)
                        is_adj = False
                    if node.func.attr == 'adj':
                        controls, targets = self.__groupValues(node.args, [0, (numTargets, numTargets)])
                        negatedControlQubits = None
                        is_adj = True

                    checkControlAndTargetTypes(controls, targets)
                    # The check above makes sure targets are either a list
                    # of individual qubits, or a single qvector. Since
                    # a qvector is not allowed, we check this here:
                    if not quake.RefType.isinstance(targets[0].type):
                        self.emitFatalError(f'invalid target operand - target must not be a qvector')

                    quake.CustomUnitarySymbolOp(
                        [],
                        generator=FlatSymbolRefAttr.get(globalName),
                        parameters=[],
                        controls=controls,
                        targets=targets,
                        is_adj=is_adj,
                        negated_qubit_controls=negatedControlQubits)
                    return

        self.emitFatalError(f"unknown function call", node)

    def visit_ListComp(self, node):
        """
        This method currently supports lowering simple list comprehensions to
        the MLIR. By simple, we mean expressions like
        `[expr(iter) for iter in iterable]` or
        `myList = [exprThatReturns(iter) for iter in iterable]`.
        """
        if len(node.generators) > 1:
            self.emitFatalError(
                "CUDA-Q only supports single generators for list comprehension.",
                node)

        # Let's handle the following `listVar` types
        # `   %9 = cc.alloca !cc.array<!cc.stdvec<T> x 2> -> ptr<array<stdvec<T> x N>`
        # or
        # `    %3 = cc.alloca T[%2 : i64] -> ptr<array<T>>`
        self.visit(node.generators[0].iter)
        iterable = self.ifPointerThenLoad(self.popValue())
        if cc.StdvecType.isinstance(iterable.type):
            iterableSize = cc.StdvecSizeOp(self.getIntegerType(), iterable).result
            iterTy = cc.StdvecType.getElementType(iterable.type)
            iterArrPtrTy = cc.PointerType.get(cc.ArrayType.get(iterTy))
            iterable = cc.StdvecDataOp(iterArrPtrTy, iterable).result
        elif quake.VeqType.isinstance(iterable.type):
            iterableSize = quake.VeqSizeOp(self.getIntegerType(),
                                            iterable).result
            iterTy = quake.RefType.get()
        else:
            self.emitFatalError(
                "CUDA-Q only supports list comprehension on ranges and arrays",
                node)

        def process_void_list():
            # NOTE: This does not actually create a valid value, and will fail is something
            # tries to use the value that this was supposed to create later on.
            # Keeping this to keep existing functionality, but this is a bit questionable.
            # Aside from no list being produced, this should work regardless of what we
            # iterate over or what expression we evaluate.
            self.emitWarning(
                "produced elements in list comprehension contain None - expression will be evaluated but no list is generated",
                node)
            forNode = ast.For()
            forNode.iter = node.generators[0].iter
            forNode.target = node.generators[0].target
            forNode.body = [node.elt]
            forNode.orelse = []
            forNode.lineno = node.lineno
            # FIXME: this loop is/should be invariant but visit_for can't know that
            self.visit_For(forNode)

        target_types = {}

        def get_target_type(target, targetType):
            if isinstance(target, ast.Name):
                if target.id in target_types:
                    self.emitFatalError(
                        "multiple definitions of target " + target.id, node)
                target_types[target.id] = targetType
            elif isinstance(target, ast.Tuple):
                if cc.PointerType.isinstance(targetType):
                    targetType = cc.PointerType.getElementType(targetType)
                if not cc.StructType.isinstance(targetType):
                    self.emitFatalError(
                        "shape mismatch in tuple deconstruction", node)
                types = cc.StructType.getTypes(targetType)
                if len(types) != len(target.elts):
                    self.emitFatalError(
                        "shape mismatch in tuple deconstruction", node)
                for i, ty in enumerate(types):
                    get_target_type(target.elts[i], ty)
            else:
                self.emitFatalError(
                    "unsupported target in tuple deconstruction", node)

        get_target_type(node.generators[0].target, iterTy)

        # We need to know the element type of the list we are creating.
        # Unfortunately, dynamic typing makes this a bit painful.
        # I didn't find a good way to fill in the type only once we
        # have processed the expression as part of the loop body,
        # but it would probably be nicer and cleaner to do that instead.
        def get_item_type(pyval):
            if isinstance(pyval, ast.Name):
                if pyval.id in target_types:
                    return target_types[pyval.id]
                self.visit(pyval)
                item = self.popValue()
                return item.type
            elif isinstance(pyval, ast.Constant):
                return mlirTypeFromPyType(type(pyval.value), self.ctx)
            elif isinstance(pyval, ast.Tuple):
                elts = [get_item_type(v) for v in pyval.elts]
                if None in elts:
                    return None
                return cc.PointerType.get(cc.StructType.getNamed("tuple", elts))
            elif isinstance(pyval, ast.Subscript) and \
                IntegerType.isinstance(get_item_type(pyval.slice)):
                parentType = get_item_type(pyval.value)
                if cc.PointerType.isinstance(parentType):
                    parentType = cc.PointerType.getElementType(parentType)
                if cc.StdvecType.isinstance(parentType):
                    return cc.StdvecType.getElementType(parentType)
                elif quake.VeqType.isinstance(parentType):
                    return quake.RefType.get()
                self.emitFatalError(
                    "unsupported data type for subscript in list comprehension",
                    node)
            elif isinstance(pyval, ast.List):
                elts = [get_item_type(v) for v in pyval.elts]
                if None in elts:
                    return None
                if len(elts) == 0:
                    self.emitFatalError(
                        "creating empty lists is not supported in CUDA-Q", node)
                base_elTy = elts[0]
                isHomogeneous = False not in [base_elTy == t for t in elts]
                if not isHomogeneous:
                    for t in elts[1:]:
                        base_elTy = self.__get_superior_type(base_elTy, t)
                        if base_elTy is None:
                            self.emitFatalError(
                                "non-homogenous list not allowed - must all be same type: {}"
                                .format(elts), node)
                return cc.StdvecType.get(base_elTy)
            elif isinstance(pyval, ast.Call):
                if isinstance(pyval.func, ast.Name):
                    # supported for calls but not here:
                    # 'range', 'enumerate', 'list'
                    if pyval.func.id == 'len' or pyval.func.id == 'int':
                        return IntegerType.get_signless(64)
                    elif pyval.func.id == 'complex':
                        return self.getComplexType()
                    elif self.__isUnitaryGate(
                            pyval.func.id) or pyval.func.id == 'reset':
                        process_void_list()
                        return None
                    elif self.__isMeasurementGate(pyval.func.id):
                        # It's tricky to know if we are calling a measurement on a single qubit,
                        # or on a vector of qubits, e.g. consider the case `[mz(qs[i:]) for i in range(n)]`,
                        # or `[mz(qs) for _ in range(n)], or [mz(qs) for _ in qs]`.
                        # We hence limit support to iterating over a vector of qubits, and check that the
                        # iteration variable is passed directly to the measurement.
                        iterSymName = None
                        if isinstance(node.generators[0].iter, ast.Name):
                            iterSymName = node.generators[0].iter.id
                        elif isinstance(node.generators[0].iter, ast.Subscript) and \
                            isinstance(node.generators[0].iter.slice, ast.Slice) and \
                            isinstance(node.generators[0].iter.value, ast.Name):
                            iterSymName = node.generators[0].iter.value.id
                        isIterOverVeq = iterSymName is not None and \
                                        iterSymName in self.symbolTable and \
                                        quake.VeqType.isinstance(self.symbolTable[iterSymName].type)
                        if not isIterOverVeq:
                            self.emitFatalError(
                                "performing measurements in list comprehension expressions is only supported when iterating over a vector of qubits",
                                node)
                        iterVarPassedAsArg = len(pyval.args) == 1 and \
                                isinstance(pyval.args[0], ast.Name) and \
                                isinstance(node.generators[0].target, ast.Name) and \
                                pyval.args[0].id == node.generators[0].target.id
                        if not iterVarPassedAsArg:
                            self.emitFatalError(
                                "unsupported argument to measurement in list comprehension",
                                node)
                        return IntegerType.get_signless(1)
                    elif pyval.func.id in globalKernelRegistry:
                        # Not necessarily unitary
                        resTypes = globalKernelRegistry[
                            pyval.func.id].type.results
                        if len(resTypes) == 0:
                            process_void_list()
                            return None
                        if len(resTypes) != 1:
                            self.emitFatalError(
                                "unsupported function call in list comprehension - function must return a single value",
                                node)
                        return resTypes[0]
                    elif pyval.func.id in globalRegisteredTypes.classes:
                        _, annotations = globalRegisteredTypes.getClassAttributes(
                            pyval.func.id)
                        structTys = [
                            mlirTypeFromPyType(v, self.ctx)
                            for _, v in annotations.items()
                        ]
                        # no need to do much verification on the validity of the type here -
                        # this will be handled when we build the body
                        isStruq = any(
                            (self.isQuantumType(t) for t in structTys))
                        if isStruq:
                            return quake.StruqType.getNamed(
                                pyval.func.id, structTys)
                        else:
                            return cc.PointerType.get(
                                cc.StructType.getNamed(pyval.func.id,
                                                       structTys))
                elif isinstance(pyval.func, ast.Attribute) and \
                    (pyval.func.attr == 'ctrl' or pyval.func.attr == 'adj'):
                    process_void_list()
                    return None
                self.emitFatalError("unsupported call in list comprehension",
                                    node)
            elif isinstance(pyval, ast.Compare):
                return IntegerType.get_signless(1)
            elif isinstance(pyval, ast.UnaryOp) and \
                isinstance(pyval.op, ast.Not):
                return IntegerType.get_signless(1)
            elif isinstance(pyval, ast.BinOp):
                # division and power are special, everything else
                # strictly creates a value of superior type
                if isinstance(pyval.op, ast.Pow):
                    # determining the correct type is messy, left as TODO for now...
                    self.emitFatalError(
                        "BinOp.Pow is not currently supported in list comprehension expressions",
                        node)
                leftTy = get_item_type(pyval.left)
                rightTy = get_item_type(pyval.right)
                superiorTy = self.__get_superior_type(leftTy, rightTy)
                # division converts integer type to `FP64` and preserves the superior type otherwise
                if isinstance(pyval.op,
                              ast.Div) and IntegerType.isinstance(superiorTy):
                    return F64Type.get()
                return superiorTy
            else:
                self.emitFatalError(
                    "Only variables, constants, and some calls can be used to populate values in list comprehension expressions",
                    node)

        listElemTy = get_item_type(node.elt)
        if listElemTy is None:
            return
        listTy = cc.ArrayType.get(listElemTy)
        listValue = cc.AllocaOp(cc.PointerType.get(listTy),
                                TypeAttr.get(listElemTy),
                                seqSize=iterableSize).result

        # General case of
        # `listVar = [expr(i) for i in iterable]`
        # Need to think of this as
        # `listVar = stdvec(iterable.size)`
        # `for i, r in enumerate(listVar):`
        # `   listVar[i] = expr(r)`
        def bodyBuilder(iterVar):
            self.symbolTable.pushScope()
            if quake.VeqType.isinstance(iterable.type):
                loadedEle = quake.ExtractRefOp(iterTy,
                                               iterable,
                                               -1,
                                               index=iterVar).result
            else:
                eleAddr = cc.ComputePtrOp(
                    cc.PointerType.get(iterTy), iterable, [iterVar],
                    DenseI32ArrayAttr.get([kDynamicPtrIndex], context=self.ctx))
                loadedEle = cc.LoadOp(eleAddr).result
            # We don't do support anything within list comprehensions that would
            # require being careful about assigning references, so simply
            # adding them to the symbol table is enough for list comprehension.
            self.__deconstructAssignment(node.generators[0].target, loadedEle)
            self.visit(node.elt)
            result = self.popValue()
            listValueAddr = cc.ComputePtrOp(
                cc.PointerType.get(listElemTy), listValue, [iterVar],
                DenseI32ArrayAttr.get([kDynamicPtrIndex], context=self.ctx))
            cc.StoreOp(result, listValueAddr)
            self.symbolTable.popScope()

        self.createInvariantForLoop(bodyBuilder, iterableSize)
        self.pushValue(
            cc.StdvecInitOp(cc.StdvecType.get(listElemTy),
                            listValue,
                            length=iterableSize).result)
        return

    def visit_List(self, node):
        """
        This method will visit the `ast.List` node and represent lists of 
        quantum typed values as a concatenated `quake.ConcatOp` producing a 
        single `veq` instances. 
        """

        # Prevent the creation of empty lists, since we don't support
        # inferring their type. To do so, we would need to look forward to
        # first use and determine the type based on that.
        if len(node.elts) == 0:
            self.emitFatalError(
                "creating empty lists is not supported in CUDA-Q", node)

        listElementValues = []
        for element in node.elts:
            if isinstance(element, ast.Starred):
                self.visit(element.value)
                evalElem = self.popValue()
                if not quake.VeqType.isinstance(evalElem.type):
                    self.emitFatalError(
                        "unpack operator `*` is only supported on qvectors",
                        node)
                listElementValues.append(evalElem)
            else:
                self.visit(element)
                # We do not store lists of pointers
                evalElem = self.ifPointerThenLoad(self.popValue())
                if self.isQuantumType(
                        evalElem.type) and not quake.RefType.isinstance(
                            evalElem.type):
                    self.emitFatalError(
                        "list must not contain a qvector or quantum struct - use `*` operator to unpack qvectors",
                        node)
                listElementValues.append(evalElem)

        numQuantumTs = sum(
            (self.isQuantumType(v.type) for v in listElementValues))
        if numQuantumTs != 0:
            if len(listElementValues) == 1:
                self.pushValue(listElementValues[0])
                return
            if numQuantumTs != len(listElementValues):
                self.emitFatalError("non-homogenous list not allowed", node)
            self.pushValue(
                quake.ConcatOp(self.getVeqType(), listElementValues).result)
            return

        # not a list of quantum types
        # Get the first element
        firstTy = listElementValues[0].type
        # Is this a list of homogenous types?
        isHomogeneous = False not in [
            firstTy == v.type for v in listElementValues
        ]

        # If not, see if the types are arithmetic and if so, find
        # the superior type and convert all to it.
        if not isHomogeneous:
            superiorType = firstTy
            for v in listElementValues[1:]:
                superiorType = self.__get_superior_type(superiorType, v.type)
                if superiorType is None:
                    self.emitFatalError(
                        "non-homogenous list not allowed - must all be same type: {}"
                        .format([v.type for v in listElementValues]), node)

            # Convert the values to the superior arithmetic type
            listElementValues = [
                self.changeOperandToType(superiorType, v, allowDemotion=False)
                for v in listElementValues
            ]

        # Turn this List into a StdVec<T>
        self.pushValue(
            self.__createStdvecWithKnownValues(listElementValues))

    def visit_Constant(self, node):
        """
        Convert constant values in the code to constant values in the MLIR. 
        """
        if isinstance(node.value, bool):
            boolValue = 0 if node.value == 0 else 1
            self.pushValue(self.getConstantInt(boolValue, 1))
            return

        if isinstance(node.value, int):
            self.pushValue(self.getConstantInt(node.value))
            return

        if isinstance(node.value, float):
            self.pushValue(self.getConstantFloat(node.value))
            return

        if isinstance(node.value, str):
            # Do not process the function doc string
            if self.docstring != None:
                if node.value.strip() == self.docstring.strip():
                    return

            strLitTy = cc.PointerType.get(
                cc.ArrayType.get(self.getIntegerType(8),
                                 len(node.value) + 1))
            self.pushValue(
                cc.CreateStringLiteralOp(strLitTy,
                                         StringAttr.get(node.value)).result)
            return

        if isinstance(node.value, type(1j)):
            self.pushValue(
                complex.CreateOp(ComplexType.get(self.getFloatType()),
                                 self.getConstantFloat(node.value.real),
                                 self.getConstantFloat(node.value.imag)).result)
            return

        self.emitFatalError("unhandled constant value", node)

    def visit_Subscript(self, node):
        """
        Convert element extractions (`__getitem__`, `operator[](idx)`, `q[1:3]`) to 
        corresponding extraction or slice code in the MLIR. This method handles 
        extraction for `veq` types and `stdvec` types. 
        """

        def get_size(val):
            if quake.VeqType.isinstance(val.type):
                return quake.VeqSizeOp(self.getIntegerType(), val).result
            elif cc.StdvecType.isinstance(val.type):
                return cc.StdvecSizeOp(self.getIntegerType(), val).result
            return None

        def fix_negative_idx(idx, get_size):
            if IntegerType.isinstance(idx.type) and \
                hasattr(idx.owner, 'opview') and \
                isinstance(idx.owner.opview, arith.ConstantOp) and \
                'value' in idx.owner.attributes:
                concreteIdx = IntegerAttr(idx.owner.attributes['value']).value
                if concreteIdx < 0:
                    size = get_size()
                    if size is not None:
                        return arith.AddIOp(
                            size, self.getConstantInt(concreteIdx)).result
            return idx

        # handle complex slice, VAR[lower:upper]
        if isinstance(node.slice, ast.Slice):
            self.debug_msg(lambda: f'[(Inline) Visit Slice]', node.slice)

            self.visit(node.value)
            var = self.ifPointerThenLoad(self.popValue())
            vectorSize = get_size(var)

            lowerVal, upperVal = None, None
            if node.slice.lower is not None:
                self.visit(node.slice.lower)
                lowerVal = fix_negative_idx(self.popValue(), lambda: vectorSize)
            else:
                lowerVal = self.getConstantInt(0)
            if node.slice.upper is not None:
                self.visit(node.slice.upper)
                upperVal = fix_negative_idx(self.popValue(), lambda: vectorSize)
            else:
                if not quake.VeqType.isinstance(
                        var.type) and not cc.StdvecType.isinstance(var.type):
                    self.emitFatalError(
                        f"unhandled upper slice == None, can't handle type {var.type}",
                        node)
                else:
                    upperVal = vectorSize

            if node.slice.step is not None:
                self.emitFatalError("step value in slice is not supported.",
                                    node)

            if quake.VeqType.isinstance(var.type):
                # Upper bound is exclusive
                upperVal = arith.SubIOp(upperVal, self.getConstantInt(1)).result
                dyna = IntegerAttr.get(self.getIntegerType(), -1)
                self.pushValue(
                    quake.SubVeqOp(self.getVeqType(),
                                   var,
                                   dyna,
                                   dyna,
                                   lower=lowerVal,
                                   upper=upperVal).result)
            elif cc.StdvecType.isinstance(var.type):
                eleTy = cc.StdvecType.getElementType(var.type)
                ptrTy = cc.PointerType.get(eleTy)
                arrTy = cc.ArrayType.get(eleTy)
                ptrArrTy = cc.PointerType.get(arrTy)
                nElementsVal = arith.SubIOp(upperVal, lowerVal).result
                # need to compute the distance between `upperVal` and `lowerVal`
                # then slice is `stdvecdataOp + computeptr[lower] + stdvecinit[ptr,distance]`
                vecPtr = cc.StdvecDataOp(ptrArrTy, var).result
                ptr = cc.ComputePtrOp(
                    ptrTy, vecPtr, [lowerVal],
                    DenseI32ArrayAttr.get([kDynamicPtrIndex],
                                          context=self.ctx)).result
                self.pushValue(
                    cc.StdvecInitOp(var.type, ptr, length=nElementsVal).result)
            else:
                self.emitFatalError(
                    f"unhandled slice operation, cannot handle type {var.type}",
                    node)

            return

        # FIXME: get rid of generic visit and replace assertion below
        self.generic_visit(node)
        assert self.valueStack.currentNumValues == 2

        # get the last name, should be name of var being subscripted
        var = self.ifPointerThenLoad(self.popValue())
        idx = self.popValue()

        if quake.VeqType.isinstance(var.type):
            if not IntegerType.isinstance(idx.type):
                self.emitFatalError(
                    f'invalid index variable type used for qvector extraction ({idx.type})',
                    node)
            idx = fix_negative_idx(idx, lambda: get_size(var))
            self.pushValue(
                quake.ExtractRefOp(self.getRefType(), var, -1,
                                   index=idx).result)
            return

        if cc.StdvecType.isinstance(var.type):
            idx = fix_negative_idx(idx, lambda: get_size(var))
            eleTy = cc.StdvecType.getElementType(var.type)
            elePtrTy = cc.PointerType.get(eleTy)
            arrTy = cc.ArrayType.get(eleTy)
            ptrArrTy = cc.PointerType.get(arrTy)
            vecPtr = cc.StdvecDataOp(ptrArrTy, var).result
            eleAddr = cc.ComputePtrOp(
                elePtrTy, vecPtr, [idx],
                DenseI32ArrayAttr.get([kDynamicPtrIndex],
                                      context=self.ctx)).result
            if self.pushPointerValue:
                self.pushValue(eleAddr)
                return
            self.pushValue(cc.LoadOp(eleAddr).result)
            return

        def get_idx_value(upper_bound):
            idxValue = None
            if hasattr(idx.owner, 'opview') and isinstance(
                    idx.owner.opview, arith.ConstantOp):
                if 'value' in idx.owner.attributes:
                    attr = IntegerAttr(idx.owner.attributes['value'])
                    idxValue = attr.value

            if idxValue == None:
                self.emitFatalError(
                    "non-constant subscript value on a tuple is not supported",
                    node)

            if idxValue < 0 or idxValue >= upper_bound:
                self.emitFatalError(f'tuple index is out of range: {idxValue}',
                                    node)
            return idxValue

        # We allow subscripts into `Structs`, but only if we don't need a pointer
        # (i.e. no updating of Tuples).
        if cc.StructType.isinstance(var.type):
            if self.pushPointerValue:
                self.emitFatalError(
                    "indexing into tuple or dataclass must not modify value",
                    node)

            memberTys = cc.StructType.getTypes(var.type)
            idxValue = get_idx_value(len(memberTys))

            member = cc.ExtractValueOp(
                memberTys[idxValue], var, [],
                DenseI32ArrayAttr.get([idxValue])).result

            self.pushValue(member)
            return

        # Let's allow subscripts into `Struqs`, but only if we don't need a pointer
        # (i.e. no updating of `Struqs`).
        if quake.StruqType.isinstance(var.type):
            if self.pushPointerValue:
                self.emitFatalError(
                    "indexing into quantum tuple or dataclass must not modify value",
                    node)

            memberTys = quake.StruqType.getTypes(var.type)
            idxValue = get_idx_value(len(memberTys))

            member = quake.GetMemberOp(
                memberTys[idxValue], var,
                IntegerAttr.get(self.getIntegerType(32), idxValue)).result
            self.pushValue(member)
            return

        self.emitFatalError("unhandled subscript", node)

    def visit_For(self, node):
        """
        Visit the For node. This node represents the typical Python for
        statement, `for VAR in ITERABLE`. Currently supported ITERABLEs are the
        `veq` type, the `stdvec` type, and the result of range() and
        enumerate().
        """

        getValues = None
        if isinstance(node.iter, ast.Call):
            self.debug_msg(lambda: f'[(Inline) Visit Call]', node.iter)

            # We can simplify `for i in range(N)` MLIR code immensely
            # by just building a for loop with N as the upper value,
            # no need to generate an array from the `range` call.
            if node.iter.func.id == 'range':
                iterable = None
                startVal, endVal, stepVal, isDecrementing = self.__processRangeLoopIterationBounds(
                    node.iter.args)
                getValues = lambda iterVar: iterVar

            # We can simplify `for i,j in enumerate(L)` MLIR code immensely
            # by just building a for loop over the iterable object L and using
            # the index into that iterable and the element.
            elif node.iter.func.id == 'enumerate':
                if len(node.iter.args) != 1:
                    self.emitFatalError("invalid number of arguments to enumerate - expecting 1 argument", node)
                
                self.visit(node.iter.args[0])
                iterable = self.ifPointerThenLoad(self.popValue())
                getValues = lambda iterVar, v: (iterVar, v)

        if not getValues:
            self.visit(node.iter)
            iterable = self.ifPointerThenLoad(self.popValue())

        if iterable:

            isDecrementing = False
            startVal = self.getConstantInt(0)
            stepVal = self.getConstantInt(1)
            relevantVals = getValues or (lambda iterVar, v: v)

            # we currently handle `veq` and `stdvec` types
            if quake.VeqType.isinstance(iterable.type):
                size = quake.VeqType.getSize(iterable.type)
                if quake.VeqType.hasSpecifiedSize(iterable.type):
                    endVal = self.getConstantInt(size)
                else:
                    endVal = quake.VeqSizeOp(self.getIntegerType(),
                                                iterable).result
                def loadElement(iterVar):
                    val = quake.ExtractRefOp(self.getRefType(),
                                                iterable,
                                                -1,
                                                index=iterVar).result
                    return relevantVals(iterVar, val)
                getValues = loadElement

            elif cc.StdvecType.isinstance(iterable.type):
                iterEleTy = cc.StdvecType.getElementType(iterable.type)
                endVal = cc.StdvecSizeOp(self.getIntegerType(),
                                            iterable).result
                def loadElement(iterVar):
                    elePtrTy = cc.PointerType.get(iterEleTy)
                    arrTy = cc.ArrayType.get(iterEleTy)
                    ptrArrTy = cc.PointerType.get(arrTy)
                    vecPtr = cc.StdvecDataOp(ptrArrTy, iterable).result
                    eleAddr = cc.ComputePtrOp(
                        elePtrTy, vecPtr, [iterVar],
                        DenseI32ArrayAttr.get([kDynamicPtrIndex],
                                                context=self.ctx)).result
                    val = cc.LoadOp(eleAddr).result
                    return relevantVals(iterVar, val)
                getValues = loadElement

            else:
                self.emitFatalError('{} iterable type not supported.', node)

        def blockBuilder(iterVar, stmts):
            self.symbolTable.pushScope()
            values = getValues(iterVar)
            # We need to create proper assignments to the loop
            # iteration variable(s) to have consistent behavior.
            assignNode = ast.Assign()
            assignNode.targets = [node.target]
            assignNode.value = values
            assignNode.lineno = node.lineno
            self.visit(assignNode)
            [self.visit(b) for b in stmts]
            self.symbolTable.popScope()

        self.createMonotonicForLoop(lambda iterVar: blockBuilder(iterVar, node.body),
                        startVal=startVal,
                        stepVal=stepVal,
                        endVal=endVal,
                        isDecrementing=isDecrementing,
                        orElseBuilder= None if not node.orelse else lambda iterVar: blockBuilder(iterVar, node.orelse))

    def visit_While(self, node):
        """
        Convert Python while statements into the equivalent CC `LoopOp`. 
        """
        def evalCond(args):
            # BUG you cannot print MLIR values while building the cc `LoopOp` while region.
            # verify will get called, no terminator yet, CCOps.cpp:520
            v = self.verbose
            self.verbose = False
            self.visit(node.test)
            condition = self.__arithmetic_to_bool(self.popValue())
            self.verbose = v
            return condition

        self.createForLoop([], lambda _: [self.visit(b) for b in node.body], [], evalCond, lambda _: [],
                          None if not node.orelse else lambda _: [self.visit(stmt) for stmt in node.orelse])

    def visit_BoolOp(self, node):
        """
        Convert boolean operations into equivalent MLIR operations using the
        Arith Dialect.
        """
        if isinstance(node.op, ast.And) or isinstance(node.op, ast.Or):

            # Visit the LHS and pop the value
            # Note we want any `mz(q)` calls to push their
            # result value to the stack, so we set a non-None
            # variable name here.
            self.currentAssignVariableName = ''
            self.visit(node.values[0])
            cond = self.__arithmetic_to_bool(self.popValue())

            def process_boolean_op(prior, values):

                if len(values) == 0: return prior

                if isinstance(node.op, ast.And):
                    prior = arith.XOrIOp(prior, self.getConstantInt(1, 1)).result

                ifOp = cc.IfOp([prior.type], prior, [])
                thenBlock = Block.create_at_start(ifOp.thenRegion, [])
                with InsertionPoint(thenBlock):
                    if isinstance(node.op, ast.And):
                        constantFalse = arith.ConstantOp(prior.type,
                                                            BoolAttr.get(False))
                        cc.ContinueOp([constantFalse])
                    else:
                        cc.ContinueOp([prior])

                elseBlock = Block.create_at_start(ifOp.elseRegion, [])
                with InsertionPoint(elseBlock):
                    self.symbolTable.pushScope()
                    self.pushIfStmtBlockStack()
                    self.visit(values[0])
                    rhs = process_boolean_op(
                        self.__arithmetic_to_bool(self.popValue()), 
                        values[1:])
                    cc.ContinueOp([rhs])
                    self.popIfStmtBlockStack()
                    self.symbolTable.popScope()
                
                return ifOp.result

            self.pushValue(process_boolean_op(cond, node.values[1:]))
            # Reset the assign variable name
            self.currentAssignVariableName = None
            return
    
        self.emitFatalError(f'unsupported boolean expression {node.op}', node)

    def visit_Compare(self, node):
        """
        Visit while loop compare operations and translate to equivalent MLIR. 
        Note, Python lets you construct expressions with multiple comparators, 
        here we limit ourselves to just a single comparator. 
        """
        if len(node.ops) > 1:
            self.emitFatalError("only single comparators are supported", node)

        iTy = self.getIntegerType()

        if isinstance(node.left, ast.Name):
            self.debug_msg(lambda: f'[(Inline) Visit Name]', node.left)
            if node.left.id not in self.symbolTable:
                self.emitFatalError(
                    f"{node.left.id} was not initialized before use in compare expression",
                    node)

        self.visit(node.left)
        left = self.popValue()
        self.visit(node.comparators[0])
        right = self.popValue()
        op = node.ops[0]

        def convert_arithmetic_types(item1, item2):
            superior_type = self.__get_superior_type(item1.type, item2.type)
            if superior_type is None:
                self.emitFatalError("invalid type in comparison", node)
            item1 = self.changeOperandToType(superior_type,
                                             item1,
                                             allowDemotion=False)
            item2 = self.changeOperandToType(superior_type,
                                             item2,
                                             allowDemotion=False)
            return item1, item2
        
        # To understand the integer attributes used here (the predicates)
        # see `arith::CmpIPredicate` and `arith::CmpFPredicate`.

        def compare_equality(item1, item2):
            # TODO: the In/NotIn case should be recursive such
            # that we can search for a list in a list of lists.
            item1, item2 = convert_arithmetic_types(item1, item2)
            iCondPred = self.getIntegerAttr(iTy, 0)
            fCondPred = self.getIntegerAttr(iTy, 1)

            if ComplexType.isinstance(item1.type):
                reComp = arith.CmpFOp(fCondPred,
                                      complex.ReOp(item1).result,
                                      complex.ReOp(item2).result)
                imComp = arith.CmpFOp(fCondPred,
                                      complex.ImOp(item1).result,
                                      complex.ImOp(item2).result)
                return arith.AndIOp(reComp, imComp).result
            elif IntegerType.isinstance(item1.type):
                return arith.CmpIOp(iCondPred, item1, item2).result
            else:
                return arith.CmpFOp(fCondPred, item1, item2).result

        if isinstance(op, ast.Gt):
            left, right = convert_arithmetic_types(left, right)
            if ComplexType.isinstance(left.type):
                self.emitFatalError("invalid type 'Complex' in comparison",
                                    node)
            elif IntegerType.isinstance(left.type):
                self.pushValue(
                    arith.CmpIOp(self.getIntegerAttr(iTy, 4), left,
                                 right).result)
            else:
                self.pushValue(
                    arith.CmpFOp(self.getIntegerAttr(iTy, 2), left,
                                 right).result)
            return

        if isinstance(op, ast.GtE):
            left, right = convert_arithmetic_types(left, right)
            if ComplexType.isinstance(left.type):
                self.emitFatalError("invalid type 'Complex' in comparison",
                                    node)
            elif IntegerType.isinstance(left.type):
                self.pushValue(
                    arith.CmpIOp(self.getIntegerAttr(iTy, 5), left,
                                 right).result)
            else:
                self.pushValue(
                    arith.CmpFOp(self.getIntegerAttr(iTy, 3), left,
                                 right).result)
            return

        if isinstance(op, ast.Lt):
            left, right = convert_arithmetic_types(left, right)
            if ComplexType.isinstance(left.type):
                self.emitFatalError("invalid type 'Complex' in comparison",
                                    node)
            elif IntegerType.isinstance(left.type):
                self.pushValue(
                    arith.CmpIOp(self.getIntegerAttr(iTy, 2), left,
                                 right).result)
            else:
                self.pushValue(
                    arith.CmpFOp(self.getIntegerAttr(iTy, 4), left,
                                 right).result)
            return

        if isinstance(op, ast.LtE):
            left, right = convert_arithmetic_types(left, right)
            if ComplexType.isinstance(left.type):
                self.emitFatalError("invalid type 'Complex' in comparison",
                                    node)
            elif IntegerType.isinstance(left.type):
                self.pushValue(
                    arith.CmpIOp(self.getIntegerAttr(iTy, 3), left,
                                 right).result)
            else:
                self.pushValue(
                    arith.CmpFOp(self.getIntegerAttr(iTy, 5), left,
                                 right).result)
            return

        if isinstance(op, ast.NotEq):
            eqComp = compare_equality(left, right)
            self.pushValue(
                arith.XOrIOp(eqComp, self.getConstantInt(1, 1)).result)
            return

        if isinstance(op, ast.Eq):
            self.pushValue(compare_equality(left, right))
            return

        if isinstance(op, (ast.In, ast.NotIn)):

            # Type validation and vector initialization
            if not cc.StdvecType.isinstance(right.type):
                self.emitFatalError(
                    "Right operand must be a list/vector for 'in' comparison")
            vectSize = cc.StdvecSizeOp(self.getIntegerType(), right).result

            # Loop setup
            i1_type = self.getIntegerType(1)
            trueVal = self.getConstantInt(1, 1)
            accumulator = cc.AllocaOp(cc.PointerType.get(i1_type),
                                      TypeAttr.get(i1_type)).result
            cc.StoreOp(trueVal, accumulator)

            # Element comparison loop
            def check_element(args):
                element = self.__load_vector_element(right, args[0])
                compRes = compare_equality(left, element)
                neqRes = arith.XOrIOp(compRes, trueVal).result
                current = cc.LoadOp(accumulator).result
                cc.StoreOp(arith.AndIOp(current, neqRes), accumulator)

            def check_condition(args):
                notListEnd = arith.CmpIOp(IntegerAttr.get(iTy, 2), args[0], vectSize).result
                notFound = cc.LoadOp(accumulator).result
                return arith.AndIOp(notListEnd, notFound).result

            # Break early if we found the item
            self.createForLoop([self.getIntegerType()],
                               check_element,
                               [self.getConstantInt(0)],
                               check_condition,
                               lambda args: [arith.AddIOp(args[0], self.getConstantInt(1)).result])

            final_result = cc.LoadOp(accumulator).result
            if isinstance(op, ast.In):
                final_result = arith.XOrIOp(final_result, trueVal).result
            self.pushValue(final_result)

            return

    def visit_If(self, node):
        """
        Map a Python `ast.If` node to an if statement operation in the CC
        dialect.
        """

        # Visit the conditional node, retain
        # measurement results by assigning a dummy variable name
        self.currentAssignVariableName = ''
        self.visit(node.test)
        self.currentAssignVariableName = None

        condition = self.ifPointerThenLoad(self.popValue())
        condition = self.__arithmetic_to_bool(condition)

        ifOp = cc.IfOp([], condition, [])
        thenBlock = Block.create_at_start(ifOp.thenRegion, [])
        with InsertionPoint(thenBlock):
            self.symbolTable.pushScope()
            self.pushIfStmtBlockStack()
            [self.visit(b) for b in node.body]
            if not self.hasTerminator(thenBlock):
                cc.ContinueOp([])
            self.popIfStmtBlockStack()
            self.symbolTable.popScope()

        if len(node.orelse) > 0:
            elseBlock = Block.create_at_start(ifOp.elseRegion, [])
            with InsertionPoint(elseBlock):
                self.symbolTable.pushScope()
                self.pushIfStmtBlockStack()
                [self.visit(b) for b in node.orelse]
                if not self.hasTerminator(elseBlock):
                    cc.ContinueOp([])
                self.popIfStmtBlockStack()
                self.symbolTable.popScope()

    def visit_Return(self, node):

        if node.value == None:
            return

        self.walkingReturnNode = True
        self.visit(node.value)
        self.walkingReturnNode = False

        if self.valueStack.currentNumValues == 0:
            return

        # FIXME: WHY TWO INDIRECTIONS? 
        # IT SEEMS WE EITHER SHOULD HAVE AT MOST ONE INDIRECTION,
        # OR WE CAN HAVE AN ARBITRARY NUMBER OF INDIRECTIONS
        result = self.ifPointerThenLoad(self.popValue())
        result = self.ifPointerThenLoad(result)
        result = self.changeOperandToType(self.knownResultType,
                                          result,
                                          allowDemotion=True)

        # FIXME: This logic needs to be updated.
        # Generally, anything that was allocated locally on the stack
        # needs to be copied to the heap to ensure it outlives the
        # the function. This holds recursively; if we have a struct
        # that contains a list, then the list data may need to be
        # copied if it was allocated inside the function.
        # However, if the return value or an item in the return value
        # is indeed a reference type passed as argument to the function,
        # then we need to make sure to keep that reference as is to
        # ensure correct behavior (i.e. behavior consistent with Python).
        if cc.StdvecType.isinstance(result.type):
            returnIsFunctionArg = BlockArgument.isinstance(result) and \
                isinstance(result.owner.owner, func.FuncOp)

            # FIXME: VECTORS OF STRUCTS AND STRUCTS OF VECTORS...
            # FIXME: add proper tests for that...
            # (currently not tested since device kernels are inlined
            # and entry point kernels can't return structs of lists)
            if not returnIsFunctionArg:
                symName = '__nvqpp_vectorCopyCtor'
                load_intrinsic(self.module, symName)
                eleTy = cc.StdvecType.getElementType(result.type)
                ptrTy = cc.PointerType.get(self.getIntegerType(8))
                arrTy = cc.ArrayType.get(self.getIntegerType(8))
                ptrArrTy = cc.PointerType.get(arrTy)
                resBuf = cc.StdvecDataOp(ptrArrTy, result).result
                # TODO Revisit this calculation
                byteWidth = 16 if ComplexType.isinstance(eleTy) else 8
                eleSize = self.getConstantInt(byteWidth)
                dynSize = cc.StdvecSizeOp(self.getIntegerType(), result).result
                resBuf = cc.CastOp(ptrTy, resBuf)
                heapCopy = func.CallOp([ptrTy], symName,
                                    [resBuf, dynSize, eleSize]).result
                result = cc.StdvecInitOp(result.type, heapCopy, length=dynSize).result

        if self.symbolTable.numLevels() > 1:
            # We are in an inner scope, release all scopes before returning
            cc.UnwindReturnOp([result])
            return

        func.ReturnOp([result])

    def visit_Tuple(self, node):
        """
        Map tuples in the Python AST to equivalents in MLIR.
        """
        # FIXME: The handling of tuples in Python likely needs to be examined carefully;
        # The corresponding issue to clarify the expected behavior is
        # https://github.com/NVIDIA/cuda-quantum/issues/3031
        # I re-enabled the tuple support in kernel signatures, given that we were already
        # allowing the use of data classes everywhere, and supporting tuple use within a
        # kernel. It hence seems that any issues with tuples also apply to named structs.

        self.generic_visit(node)

        elementValues = self.popAllValues(len(node.elts))
        elementValues.reverse()

        # We do not store structs of pointers
        elementValues = [
            cc.LoadOp(ele).result
            if cc.PointerType.isinstance(ele.type) else ele
            for ele in elementValues
        ]

        structTys = [v.type for v in elementValues]
        structTy = mlirTryCreateStructType(structTys, context=self.ctx)
        if structTy is None:
            self.emitFatalError(
                "hybrid quantum-classical data types and nested quantum structs are not allowed",
                node)

        if quake.StruqType.isinstance(structTy):
            self.pushValue(quake.MakeStruqOp(structTy, elementValues).result)
        else:
            stackSlot = cc.AllocaOp(cc.PointerType.get(structTy),
                                    TypeAttr.get(structTy)).result

            # loop over each type and `compute_ptr` / store
            for i, ty in enumerate(structTys):
                eleAddr = cc.ComputePtrOp(
                    cc.PointerType.get(ty), stackSlot, [],
                    DenseI32ArrayAttr.get([i], context=self.ctx)).result
                cc.StoreOp(elementValues[i], eleAddr)

            self.pushValue(stackSlot)
            return

    def visit_UnaryOp(self, node):
        """
        Map unary operations in the Python AST to equivalents in MLIR.
        """

        self.generic_visit(node)
        # FIXME: REPLACE THIS
        assert self.valueStack.currentNumValues == 1
        operand = self.popValue()

        # Handle qubit negations
        if isinstance(node.op, ast.Invert):
            if quake.RefType.isinstance(operand.type):
                self.controlNegations.append(operand)
                self.pushValue(operand)
                return
            else:
                self.emitFatalError(
                    "unary operator ~ is only supported for values of type qubit",
                    node)

        if isinstance(node.op, ast.USub):
            # Make our lives easier for -1 used in variable subscript extraction
            if isinstance(node.operand,
                          ast.Constant) and node.operand.value == 1:
                self.pushValue(self.getConstantInt(-1))
                return

            if F64Type.isinstance(operand.type):
                self.pushValue(arith.NegFOp(operand).result)
            elif ComplexType.isinstance(operand.type):
                # `complex.NegOp` does not seem to work
                self.pushValue(
                    complex.MulOp(
                        complex.CreateOp(operand.type,
                                         self.getConstantFloat(-1.),
                                         self.getConstantFloat(0.)).result,
                        operand).result)
            else:
                negOne = self.getConstantInt(-1)
                self.pushValue(arith.MulIOp(negOne, operand).result)
            return

        if isinstance(node.op, ast.Not):
            self.pushValue(arith.XOrIOp(
                self.__arithmetic_to_bool(operand),
                self.getConstantInt(1, 1)).result)
            return

        self.emitFatalError("unhandled UnaryOp.", node)

    def visit_Break(self, node):

        if not self.isInForBody():
            self.emitFatalError("break statement outside of for loop body.",
                                node)

        if self.isInIfStmtBlock():
            # Get the innermost enclosing `for` or `while` loop
            inArgs = [b for b in self.inForBodyStack[-1]]
            cc.UnwindBreakOp(inArgs)
        else:
            cc.BreakOp([])

        return

    def visit_Continue(self, node):

        if not self.isInForBody():
            self.emitFatalError("continue statement outside of for loop body.",
                                node)

        if self.isInIfStmtBlock():
            # Get the innermost enclosing `for` or `while` loop
            inArgs = [b for b in self.inForBodyStack[-1]]
            cc.UnwindContinueOp(inArgs)
        else:
            cc.ContinueOp([])

    def __process_binary_op(self, left, right, nodeType):
        """
        Process a binary operation in the AST and map them to equivalents in the 
        MLIR. This method handles arithmetic operations between values. 
        """

        left = self.ifPointerThenLoad(left)
        right = self.ifPointerThenLoad(right)

        # type promotion for anything except pow to match Python behavior
        if not issubclass(nodeType, ast.Pow):
            superiorTy = self.__get_superior_type(left.type, right.type)
            if superiorTy is not None:
                left = self.changeOperandToType(superiorTy,
                                                left,
                                                allowDemotion=False)
                right = self.changeOperandToType(superiorTy,
                                                 right,
                                                 allowDemotion=False)

        # Note: including support for any non-arithmetic types
        # (e.g. addition on lists) needs to be tested/implemented
        # also when used in list comprehension expressions.
        if not self.isArithmeticType(left.type) or not self.isArithmeticType(
                right.type):
            self.emitFatalError(f'Invalid type for {nodeType}',
                                self.currentNode)

        # Based on the op type and the leaf types, create the MLIR operator
        if issubclass(nodeType, ast.Add):
            if IntegerType.isinstance(left.type):
                self.pushValue(arith.AddIOp(left, right).result)
                return
            elif F64Type.isinstance(left.type) or \
                F32Type.isinstance(left.type):
                self.pushValue(arith.AddFOp(left, right).result)
                return
            elif ComplexType.isinstance(left.type):
                self.pushValue(complex.AddOp(left, right).result)
                return
            else:
                self.emitFatalError("unhandled BinOp.Add types",
                                    self.currentNode)

        if issubclass(nodeType, ast.Sub):
            if IntegerType.isinstance(left.type):
                self.pushValue(arith.SubIOp(left, right).result)
                return
            elif F64Type.isinstance(left.type) or \
                F32Type.isinstance(left.type):
                self.pushValue(arith.SubFOp(left, right).result)
                return
            if ComplexType.isinstance(left.type):
                self.pushValue(complex.SubOp(left, right).result)
                return
            else:
                self.emitFatalError("unhandled BinOp.Sub types",
                                    self.currentNode)

        if issubclass(nodeType, ast.FloorDiv):
            if IntegerType.isinstance(left.type):
                self.pushValue(arith.FloorDivSIOp(left, right).result)
                return
            else:
                self.emitFatalError("unhandled BinOp.FloorDiv types",
                                    self.currentNode)

        if issubclass(nodeType, ast.Div):
            if IntegerType.isinstance(left.type):
                left = arith.SIToFPOp(self.getFloatType(), left).result
                right = arith.SIToFPOp(self.getFloatType(), right).result
            if F64Type.isinstance(left.type) or \
                F32Type.isinstance(left.type):
                self.pushValue(arith.DivFOp(left, right).result)
                return
            elif ComplexType.isinstance(left.type):
                self.pushValue(complex.DivOp(left, right).result)
                return
            else:
                self.emitFatalError("unhandled BinOp.Div types",
                                    self.currentNode)

        if issubclass(nodeType, ast.Mult):
            if IntegerType.isinstance(left.type):
                self.pushValue(arith.MulIOp(left, right).result)
                return
            elif F64Type.isinstance(left.type) or \
                F32Type.isinstance(left.type):
                self.pushValue(arith.MulFOp(left, right).result)
                return
            if ComplexType.isinstance(left.type):
                self.pushValue(complex.MulOp(left, right).result)
                return
            else:
                self.emitFatalError("unhandled BinOp.Mult types",
                                    self.currentNode)

        if issubclass(nodeType, ast.Pow):
            if IntegerType.isinstance(left.type) and IntegerType.isinstance(
                    right.type):
                # `math.ipowi` does not lower to LLVM as is
                # workaround, use math to function conversion
                self.pushValue(math.IPowIOp(left, right).result)
                return
            if (F64Type.isinstance(left.type) or F32Type.isinstance(left.type)) and \
                IntegerType.isinstance(right.type):
                self.pushValue(math.FPowIOp(left, right).result)
                return
            if IntegerType.isinstance(left.type):
                left = arith.SIToFPOp(self.getFloatType(), left).result
            if IntegerType.isinstance(right.type):
                right = arith.SIToFPOp(self.getFloatType(), right).result
            if F64Type.isinstance(left.type):
                self.pushValue(math.PowFOp(left, right).result)
                return
            else:
                self.emitFatalError("unhandled BinOp.Pow types",
                                    self.currentNode)

        if issubclass(nodeType, ast.Mod):
            if IntegerType.isinstance(left.type):
                self.pushValue(arith.RemUIOp(left, right).result)
                return
            if F64Type.isinstance(left.type) or \
                F32Type.isinstance(left.type):
                self.pushValue(arith.RemFOp(left, right).result)
                return
            else:
                self.emitFatalError("unhandled BinOp.Mod types",
                                    self.currentNode)

        if issubclass(nodeType, ast.LShift):
            if IntegerType.isinstance(left.type):
                self.pushValue(arith.ShLIOp(left, right).result)
                return
            else:
                self.emitFatalError(
                    "unsupported operand type(s) for BinOp.LShift; only integers supported",
                    self.currentNode)

        if issubclass(nodeType, ast.RShift):
            if IntegerType.isinstance(left.type):
                self.pushValue(arith.ShRSIOp(left, right).result)
                return
            else:
                self.emitFatalError(
                    "unsupported operand type(s) for BinOp.RShift; only integers supported",
                    self.currentNode)

        if issubclass(nodeType, ast.BitAnd):
            if IntegerType.isinstance(left.type):
                self.pushValue(arith.AndIOp(left, right).result)
                return
            else:
                self.emitFatalError(
                    "unsupported operand type(s) for BinOp.BitAnd; only integers supported",
                    self.currentNode)

        if issubclass(nodeType, ast.BitOr):
            if IntegerType.isinstance(left.type):
                self.pushValue(arith.OrIOp(left, right).result)
                return
            else:
                self.emitFatalError(
                    "unsupported operand type(s) for BinOp.BitOr; only integers supported",
                    self.currentNode)

        if issubclass(nodeType, ast.BitXor):
            if IntegerType.isinstance(left.type):
                self.pushValue(arith.XOrIOp(left, right).result)
                return
            else:
                self.emitFatalError(
                    "unsupported operand type(s) for BinOp.BitXor; only integers supported.",
                    self.currentNode)

        self.emitFatalError("unhandled binary operator", self.currentNode)

    def visit_BinOp(self, node):
        """
        Visit binary operation nodes in the AST and map them to equivalents in the 
        MLIR. This method handles arithmetic operations between values. 
        """

        self.visit(node.left)
        left = self.popValue()
        self.visit(node.right)
        right = self.popValue()

        # pushes to the value stack
        self.__process_binary_op(left, right, type(node.op))

    def visit_AugAssign(self, node):
        """
        Visit augment-assign operations (e.g. +=). 
        """
        target = None
        if isinstance(node.target, ast.Name) and \
            node.target.id in self.symbolTable and \
            not node.target.id in self.capturedVars:
            self.debug_msg(lambda: f'[(Inline) Visit Name]', node.target)
            target = self.symbolTable[node.target.id]
        if not target or not cc.PointerType.isinstance(target.type):
            self.emitFatalError(
                "augment-assign target variable is not defined or cannot be assigned to.",
                node)

        self.visit(node.value)
        value = self.popValue()
        loaded = cc.LoadOp(target).result

        self.valueStack.pushFrame()
        self.__process_binary_op(loaded, value, type(node.op))
        self.valueStack.popFrame()
        res = self.popValue()
        # FIXME: aug assign is usually defined as producing a value, 
        # which we are not doing here. Now that we add proper stacks 
        # for values, we can/should probably push res back on the stack

        if res.type != loaded.type:
            self.emitFatalError(
                "augment-assign must not change the variable type", node)
        cc.StoreOp(res, target)

    def visit_Name(self, node):
        """
        Visit `ast.Name` nodes and extract the correct value from the symbol
        table.
        """

        if node.id in self.symbolTable:
            value = self.symbolTable[node.id]
            if cc.PointerType.isinstance(value.type):
                eleTy = cc.PointerType.getElementType(value.type)

                # Retain types that corresponds to Python reference
                # types as pointers
                if cc.StructType.isinstance(eleTy):
                    self.pushValue(value)
                    return
                if cc.StdvecType.isinstance(eleTy):
                    self.pushValue(value)
                    return

                # Always retain array types (used for strings)
                if cc.ArrayType.isinstance(eleTy):
                    self.pushValue(value)
                    return
                # Always retain `ptr<i8>` (used for `dbg` functions)
                if IntegerType.isinstance(eleTy) and IntegerType(
                        eleTy).width == 8:
                    self.pushValue(value)
                    return
                # Retain state types as pointers
                if cc.StateType.isinstance(eleTy):
                    self.pushValue(value)
                    return

                loaded = cc.LoadOp(value).result
                self.pushValue(loaded)
            else:
                self.pushValue(value)
            return

        if node.id in self.capturedVars:
            # Only support a small subset of types here
            complexType = type(1j)
            value = self.capturedVars[node.id]

            if isinstance(value, State):
                self.pushValue(self.capturedDataStorage.storeCudaqState(value))
                return

            if isinstance(value, (list, np.ndarray)) and isinstance(
                    value[0],
                (int, bool, float, np.int32, np.int64, np.float32, np.float64,
                 complexType, np.complex64, np.complex128)):

                elementValues = None
                if isinstance(value[0], bool):
                    elementValues = [self.getConstantInt(el, 1) for el in value]
                elif isinstance(value[0], np.int32):
                    elementValues = [
                        self.getConstantInt(el, width=32) for el in value
                    ]
                elif isinstance(value[0], (int, np.int64)):
                    elementValues = [self.getConstantInt(el) for el in value]
                elif isinstance(value[0], np.float32):
                    elementValues = [
                        self.getConstantFloat(el, width=32) for el in value
                    ]
                elif isinstance(value[0], (float, np.float64)):
                    elementValues = [self.getConstantFloat(el) for el in value]
                elif isinstance(value[0], np.complex64):
                    elementValues = [
                        self.getConstantComplex(el, width=32) for el in value
                    ]
                elif isinstance(value[0], complexType) or isinstance(
                        value[0], np.complex128):
                    elementValues = [
                        self.getConstantComplex(el, width=64) for el in value
                    ]

                if elementValues != None:
                    # Save the copy of the captured list so we can compare
                    # it to the scope to detect changes on recompilation.
                    self.dependentCaptureVars[node.id] = value.copy()
                    mlirVal = self.__createStdvecWithKnownValues(elementValues)
                    # This is just basically a form of caching to ensure that
                    # we only ever create one copy of a captured value.
                    self.symbolTable.add(node.id, mlirVal, 0)
                    self.pushValue(mlirVal)
                    return

            mlirVal = None
            self.dependentCaptureVars[node.id] = value
            if isinstance(value, bool):
                mlirVal = self.getConstantInt(value, 1)
            elif isinstance(value, np.int32):
                mlirVal = self.getConstantInt(value, width=32)
            elif isinstance(value, (int, np.int64)):
                mlirVal = self.getConstantInt(value)
            elif isinstance(value, np.float32):
                mlirVal = self.getConstantFloat(value, width=32)
            elif isinstance(value, (float, np.float64)):
                mlirVal = self.getConstantFloat(value)
            elif isinstance(value, np.complex64):
                mlirVal = self.getConstantComplex(value, width=32)
            elif isinstance(value, complexType) or isinstance(
                    value, np.complex128):
                mlirVal = self.getConstantComplex(value, width=64)

            if mlirVal != None:
                self.pushValue(mlirVal)
                return

            errorType = type(value).__name__
            if (isinstance(value, list)):
                errorType = f"{errorType}[{type(value[0]).__name__}]"

            try:
                if issubclass(value, cudaq_runtime.KrausChannel):
                    # Here we have a KrausChannel as part of the AST.  We want
                    # to create a hash value from it, and we then want to push
                    # the number of parameters and that hash value. This can
                    # only be used with apply_noise.
                    if not hasattr(value, 'num_parameters'):
                        self.emitFatalError(
                            'apply_noise kraus channels must have `num_parameters` constant class attribute specified.'
                        )

                    # FIXME: THIS IS PROBABLY HOW THAT WORKED WITH CUSTOM CHANNELS...
                    self.pushValue(self.getConstantInt(value.num_parameters))
                    self.pushValue(self.getConstantInt(hash(value)))
                    return
            except TypeError:
                pass

            if node.id not in globalKernelRegistry and \
                node.id not in globalRegisteredOperations:
                self.emitFatalError(
                    f"Invalid type for variable ({node.id}) captured from parent scope (only int, bool, float, complex, cudaq.State, and list/np.ndarray[int|bool|float|complex] accepted, type was {errorType}).",
                    node)

        if node.id in globalKernelRegistry or \
            node.id in globalRegisteredOperations:
            return

        if self.__isUnitaryGate(node.id) or \
            self.__isMeasurementGate(node.id):
            return

        if node.id == 'complex':
            self.pushValue(self.getComplexType())
            return

        if node.id == 'float':
            self.pushValue(self.getFloatType())
            return

        # Throw an exception for the case that the name is not
        # in the symbol table
        self.emitFatalError(
            f"Invalid variable name requested - '{node.id}' is not defined within the scope it is used in.",
            node)


def compile_to_mlir(astModule, capturedDataStorage: CapturedDataStorage,
                    **kwargs):
    """
    Compile the given Python AST Module for the CUDA-Q 
    kernel FunctionDef to an MLIR `ModuleOp`. 
    Return both the `ModuleOp` and the list of function 
    argument types as MLIR Types. 

    This function will first check to see if there are any dependent 
    kernels that are required by this function. If so, those kernels 
    will also be compiled into the `ModuleOp`. The AST will be stored 
    later for future potential dependent kernel lookups. 
    """

    global globalAstRegistry
    verbose = 'verbose' in kwargs and kwargs['verbose']
    returnType = kwargs['returnType'] if 'returnType' in kwargs else None
    lineNumberOffset = kwargs['location'] if 'location' in kwargs else ('', 0)
    parentVariables = kwargs[
        'parentVariables'] if 'parentVariables' in kwargs else {}

    # Create the AST Bridge
    bridge = PyASTBridge(capturedDataStorage,
                         verbose=verbose,
                         knownResultType=returnType,
                         returnTypeIsFromPython=True,
                         locationOffset=lineNumberOffset,
                         capturedVariables=parentVariables)

    ValidateArgumentAnnotations(bridge).visit(astModule)
    ValidateReturnStatements(bridge).visit(astModule)

    # First we need to find any dependent kernels, they have to be
    # built as part of this ModuleOp...
    vis = FindDepKernelsVisitor(bridge.ctx)
    vis.visit(astModule)
    depKernels = vis.depKernels

    # Keep track of a kernel call graph, we will
    # sort this later after we build up the graph
    callGraph = {vis.kernelName: {k for k, v in depKernels.items()}}

    # Visit dependent kernels recursively to
    # ensure we have all necessary kernels added to the
    # module
    transitiveDeps = depKernels
    while len(transitiveDeps):
        # For each found dependency, see if that kernel
        # has further dependencies
        for depKernelName, depKernelAst in transitiveDeps.items():
            localVis = FindDepKernelsVisitor(bridge.ctx)
            localVis.visit(depKernelAst[0])
            # Append the found dependencies to our running tally
            depKernels = {**depKernels, **localVis.depKernels}
            # Reset for the next go around
            transitiveDeps = localVis.depKernels
            # Update the call graph
            callGraph[localVis.kernelName] = {
                k for k, v in localVis.depKernels.items()
            }

    # Sort the call graph topologically
    callGraphSorter = graphlib.TopologicalSorter(callGraph)
    sortedOrder = callGraphSorter.static_order()

    # Add all dependent kernels to the MLIR Module,
    # Do not check any 'dependent' kernels that
    # have the same name as the main kernel here, i.e.
    # ignore kernels that have the same name as this one.
    for funcName in sortedOrder:
        if funcName != vis.kernelName and funcName in depKernels:
            # Build an AST Bridge and visit the dependent kernel
            # function. Provide the dependent kernel source location as well.
            PyASTBridge(capturedDataStorage,
                        existingModule=bridge.module,
                        locationOffset=depKernels[funcName][1]).visit(
                            depKernels[funcName][0])

    # Build the MLIR Module for this kernel
    bridge.visit(astModule)

    if verbose:
        print(bridge.module)

    # Canonicalize the code, check for measurement(s) readout
    pm = PassManager.parse(
        "builtin.module(func.func(unwind-lowering,canonicalize,cse,quake-add-metadata),quake-propagate-metadata)",
        context=bridge.ctx)

    try:
        pm.run(bridge.module)
    except:
        raise RuntimeError("could not compile code for '{}'.".format(
            bridge.name))

    extraMetaData = {}
    if len(bridge.dependentCaptureVars):
        extraMetaData['dependent_captures'] = bridge.dependentCaptureVars

    return bridge.module, bridge.argTypes, extraMetaData
