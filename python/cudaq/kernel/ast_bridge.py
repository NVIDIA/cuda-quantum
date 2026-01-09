# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import ast
import importlib
import inspect
import textwrap
import numpy as np
import os
import sys
import types
from collections import deque

from cudaq.mlir._mlir_libs._quakeDialects import (cudaq_runtime, load_intrinsic,
                                                  gen_vector_of_complex_constant
                                                 )
from cudaq.mlir.dialects import arith, cc, complex, func, math, quake
from cudaq.mlir.ir import (BoolAttr, Block, BlockArgument, Context, ComplexType,
                           DenseBoolArrayAttr, DenseI32ArrayAttr,
                           DenseI64ArrayAttr, DictAttr, F32Type, F64Type,
                           FlatSymbolRefAttr, FloatAttr, FunctionType,
                           InsertionPoint, IntegerAttr, IntegerType, Location,
                           Module, StringAttr, SymbolTable, TypeAttr, UnitAttr)
from cudaq.mlir.passmanager import PassManager
from .analysis import ValidateArgumentAnnotations, ValidateReturnStatements
from .captured_data import CapturedDataStorage
from .utils import (Color, globalAstRegistry, globalRegisteredOperations,
                    globalRegisteredTypes, nvqppPrefix, mlirTypeFromAnnotation,
                    mlirTypeFromPyType, mlirTypeToPyType, getMLIRContext,
                    recover_func_op, is_recovered_value_ok,
                    recover_value_of_or_none, cudaq__unique_attr_name,
                    mlirTryCreateStructType, resolve_qualified_symbol)

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

    # FIXME: make this similar to `pystack` to account for local functions
    def __init__(self):
        self.symbolTable = {}
        self.scopes = deque()
        self.scopeID = 0

    def pushScope(self):
        self.scopes.append(self.scopeID)
        self.scopeID += 1

    def popScope(self):
        self.scopes.pop()

    @property
    def depth(self):
        return len(self.scopes)

    def isDefined(self, symbol):
        return symbol in self.symbolTable

    def add(self, symbol, value, level=0):
        """
        Add a symbol to the scoped symbol table at any scope level.
        """
        self.symbolTable[symbol] = (value, level)

    def __contains__(self, symbol):
        if not symbol in self.symbolTable:
            return False

        # According to Python scoping rules, all variables within a function
        # are defined within the same scope. We hence make sure to insert all
        # `alloca` instructions to store variables in the main block of the
        # function (otherwise MLIR will fail with "operand does not dominate
        # this use"). However, some variables are stored as values in the
        # symbol table, meaning they are only defined if they are defined in
        # the current scope or in a parent scope (i.e. `sid` is not None).
        value, sid = self.symbolTable[symbol]
        return (sid in self.scopes or
                (isinstance(value.owner.opview, cc.AllocaOp) and
                 isinstance(value.owner.parent.opview, func.FuncOp)))

    def __setitem__(self, symbol, value):
        assert len(self.scopes) > 0
        self.add(symbol, value, self.scopes[-1])

    def __getitem__(self, symbol):
        if symbol in self:
            return self.symbolTable[symbol][0]
        if symbol in self.symbolTable:
            # We have a variable that is defined in an inner scope,
            # but not allocated in the main function body.
            # This case deviates from Python behavior, and we give
            # a hopefully comprehensive enough error.
            raise RuntimeError(
                f"variable of type {self.symbolTable[symbol][0].type} " +
                "is defined in a prior block and cannot be " +
                "accessed or changed outside that block" + os.linesep +
                f"(offending source -> {symbol})")
        raise RuntimeError(
            f"{symbol} is not a valid variable name in this scope.")

    def isInCurrentScope(self, symbol):
        return (symbol in self.symbolTable and len(self.scopes) > 0 and
                self.symbolTable[symbol][1] == self.scopes[-1])

    def clear(self):
        assert len(self.scopes) == 0
        self.symbolTable = {}
        self.scopes = {}
        self.scopeID = 0


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

        def __init__(self, parent=None):
            self.entries = None
            self.parent = parent

    def __init__(self, error_handler=None):

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
        self._frame = PyStack.Frame(parent=self._frame)

    def popFrame(self):
        '''
        A frame should be popped once a node in the AST has been processed.
        '''
        if not self._frame:
            self.emitError("stack has no frames to pop")
        elif self._frame.entries:
            self.emitError(
                "all values must be processed before popping a frame")
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
            # the bridge is doing its processing correctly.  We hence give a
            # somewhat general error.  For internal purposes, the error might be
            # better stated as something like: either this frame has not had a
            # child or the child did not produce any values
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


def recover_kernel_decorator(name):
    from .kernel_decorator import isa_kernel_decorator
    for frameinfo in inspect.stack():
        frame = frameinfo.frame
        if name in frame.f_locals:
            if isa_kernel_decorator(frame.f_locals[name]):
                return frame.f_locals[name]
            return None
        if name in frame.f_globals:
            if isa_kernel_decorator(frame.f_globals[name]):
                return frame.f_globals[name]
            return None
    return None


class PyASTBridge(ast.NodeVisitor):
    """
    The `PyASTBridge` class implements the `ast.NodeVisitor` type to convert a
    python function definition (annotated with cudaq.kernel) to an MLIR
    `ModuleOp` containing a `func.FuncOp` representative of the original python
    function but leveraging the Quake and CC dialects provided by CUDA-Q. This
    class keeps track of a MLIR Value stack that is pushed to and popped from
    during visitation of the function AST nodes. We leverage the auto-generated
    MLIR Python bindings for the internal C++ CUDA-Q dialects to build up the
    MLIR code.
    """

    def __init__(self, capturedDataStorage: CapturedDataStorage, **kwargs):
        """
        The constructor. Initializes the `mlir.Value` stack, the `mlir.Context`,
        and the `mlir.Module` that we will be building upon. This class keeps
        track of a symbol table, which maps variable names to constructed
        `mlir.Values`.
        """
        self.valueStack = PyStack(lambda msg: self.emitFatalError(
            f'processing error - {msg}', self.currentNode))
        self.knownResultType = kwargs[
            'knownResultType'] if 'knownResultType' in kwargs else None
        self.uniqueId = kwargs['uniqueId'] if 'uniqueId' in kwargs else None
        self.kernelModuleName = kwargs[
            'kernelModuleName'] if 'kernelModuleName' in kwargs else None
        if 'existingModule' in kwargs:
            self.module = kwargs['existingModule']
            self.ctx = self.module.context
            self.loc = Location.unknown(context=self.ctx)
        else:
            self.ctx = getMLIRContext()
            self.loc = Location.unknown(context=self.ctx)
            self.module = Module.create(self.loc)

        # Create a new captured data storage or use the existing one passed from
        # the current kernel decorator.
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

        # If the driver of this AST bridge instance has indicated that there is
        # a return type from analysis on the Python AST, then we want to set the
        # known result type so that the FuncOp can have it.
        if 'returnTypeIsFromPython' in kwargs and kwargs[
                'returnTypeIsFromPython'] and self.knownResultType is not None:
            self.knownResultType = mlirTypeFromPyType(self.knownResultType,
                                                      self.ctx)

        self.capturedVars = {}
        self.dependentCaptureVars = {}
        self.liftedArgs = []
        self.locationOffset = kwargs[
            'locationOffset'] if 'locationOffset' in kwargs else ('', 0)
        self.disableEntryPointTag = (kwargs['disableEntryPointTag']
                                     if 'disableEntryPointTag' in kwargs else
                                     False)
        self.disableNvqppPrefix = kwargs[
            'disableNvqppPrefix'] if 'disableNvqppPrefix' in kwargs else False
        self.symbolTable = PyScopedSymbolTable()
        self.indent_level = 0
        self.indent = 4 * " "
        self.buildingEntryPoint = False
        self.inForBodyStack = deque()
        self.inIfStmtBlockStack = 0
        self.currentAssignVariableName = None
        self.walkingReturnNode = False
        self.controlNegations = []
        self.pushPointerValue = False
        self.isSubscriptRoot = False
        self.verbose = 'verbose' in kwargs and kwargs['verbose']
        self.currentNode = None
        self.firstLiftedPos = None

    def debug_msg(self, msg, node=None):
        if self.verbose:
            print(f'{self.indent * self.indent_level}{msg()}')
            if node is not None:
                try:
                    print(
                        textwrap.indent(ast.unparse(node),
                                        (self.indent *
                                         (self.indent_level + 1))))
                except:
                    pass

    def emitWarning(self, msg, astNode=None):
        """
        Emit a warning, providing the user with source file information and
        the offending code.
        """
        codeFile = os.path.basename(self.locationOffset[0])
        if astNode == None:
            astNode = self.currentNode
        lineNumber = ('' if astNode == None else astNode.lineno +
                      self.locationOffset[1] - 1)

        print(Color.BOLD, end='')
        msg = (codeFile + ":" + str(lineNumber) + ": " + Color.YELLOW +
               "warning: " + Color.END + Color.BOLD + msg +
               ("\n\t (offending source -> " + ast.unparse(astNode) + ")"
                if hasattr(ast, 'unparse') and astNode is not None else '') +
               Color.END)
        print(msg)

    def emitFatalError(self, msg, astNode=None):
        """
        Emit a fatal error, providing the user with source file information and
        the offending code.
        """
        codeFile = os.path.basename(self.locationOffset[0])
        if astNode == None:
            astNode = self.currentNode
        lineNumber = '' if astNode == None or not hasattr(
            astNode, 'lineno') else astNode.lineno + self.locationOffset[1] - 1

        try:
            offending_source = "\n\t (offending source -> " + ast.unparse(
                astNode) + ")"
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

    def isFunctionArgument(self, value):
        return (BlockArgument.isinstance(value) and
                isinstance(value.owner.owner, func.FuncOp))

    def containsList(self, ty, innerListsOnly=False):
        """
        Returns true if the give type is a vector or contains
        items that are vectors.
        """
        if cc.StdvecType.isinstance(ty):
            return (not innerListsOnly or
                    self.containsList(cc.StdvecType.getElementType(ty)))
        if not cc.StructType.isinstance(ty):
            return False
        eleTys = cc.StructType.getTypes(ty)
        return any((self.containsList(t) for t in eleTys))

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
        # `numpy.complex128` is the same as `complex` type, with element width
        # of 64bit (`np.complex64` and `float`) `numpy.complex64` type has
        # element type of `np.float32`.
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
        Change the type of an operand to a specified type. This function
        primarily handles type conversions and promotions to higher types
        (complex > float > int).  Demotion of floating type to integer is not
        allowed by default.  Regardless of whether demotion is allowed, types
        will be cast to smaller widths.
        """
        if ty == operand.type:
            return operand
        if cc.CallableType.isinstance(ty):
            fctTy = cc.CallableType.getFunctionType(ty)
            if fctTy == operand.type:
                return operand
            self.emitFatalError(
                f'cannot convert value of type {operand.type} to '
                f'the requested type {fctTy}', self.currentNode)

        if ComplexType.isinstance(ty):
            complexType = ComplexType(ty)
            floatType = complexType.element_type
            if ComplexType.isinstance(operand.type):
                otherComplexType = ComplexType(operand.type)
                otherFloatType = otherComplexType.element_type
                if (floatType != otherFloatType):
                    real = self.changeOperandToType(
                        floatType,
                        complex.ReOp(operand).result,
                        allowDemotion=allowDemotion)
                    imag = self.changeOperandToType(
                        floatType,
                        complex.ImOp(operand).result,
                        allowDemotion=allowDemotion)
                    return complex.CreateOp(complexType, real, imag).result
            else:
                real = self.changeOperandToType(floatType,
                                                operand,
                                                allowDemotion=allowDemotion)
                imag = self.getConstantFloatWithType(0.0, floatType)
                return complex.CreateOp(complexType, real, imag).result

        if (cc.StdvecType.isinstance(ty)):
            if cc.StdvecType.isinstance(operand.type):
                eleTy = cc.StdvecType.getElementType(ty)
                return self.__copyVectorAndConvertElements(
                    operand,
                    eleTy,
                    allowDemotion=allowDemotion,
                    alwaysCopy=False)

        if (cc.StructType.isinstance(ty)):
            if cc.StructType.isinstance(operand.type):
                expectedEleTys = cc.StructType.getTypes(ty)
                currentEleTys = cc.StructType.getTypes(operand.type)
                if len(expectedEleTys) == len(currentEleTys):

                    def conversion(idx, value):
                        return self.changeOperandToType(
                            expectedEleTys[idx],
                            value,
                            allowDemotion=allowDemotion)

                    return self.__copyStructAndConvertElements(
                        operand,
                        expectedTy=ty,
                        allowDemotion=allowDemotion,
                        conversion=conversion)

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
            f'cannot convert value of type {operand.type} '
            f'to the requested type {ty}', self.currentNode)

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
        values = [
            self.popValue() for _ in range(self.valueStack.currentNumValues)
        ]
        if len(values) != expectedNumVals:
            self.emitFatalError(
                "processing error - expression did not produce a valid "
                "value in this context", self.currentNode)
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
        self.inIfStmtBlockStack += 1

    def popIfStmtBlockStack(self):
        """
        Indicate that we have just left an if statement then or else block.
        """
        assert self.inIfStmtBlockStack > 0
        self.inIfStmtBlockStack -= 1

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
        return self.inIfStmtBlockStack > 0

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
        return (self.__isSimpleGate(id) or self.__isRotationGate(id) or
                self.__isAdjointSimpleGate(id) or
                self.__isControlledSimpleGate(id) or
                self.__isControlledRotationGate(id) or
                id in ['swap', 'u3', 'exp_pauli'] or
                id in globalRegisteredOperations)

    def __createStdvecWithKnownValues(self, listElementValues):
        assert (len(set((v.type for v in listElementValues))) == 1)
        arrSize = self.getConstantInt(len(listElementValues))
        elemTy = listElementValues[0].type
        # If this is an `i1`, turns it into an `i8` array.
        isBool = elemTy == self.getIntegerType(1)
        if isBool:
            elemTy = self.getIntegerType(8)
        alloca = cc.AllocaOp(cc.PointerType.get(cc.ArrayType.get(elemTy)),
                             TypeAttr.get(elemTy),
                             seqSize=arrSize).result

        for i, v in enumerate(listElementValues):
            eleAddr = cc.ComputePtrOp(
                cc.PointerType.get(elemTy), alloca, [self.getConstantInt(i)],
                DenseI32ArrayAttr.get([kDynamicPtrIndex],
                                      context=self.ctx)).result
            if isBool:
                # Cast the list value before assigning
                v = self.changeOperandToType(self.getIntegerType(8), v)
            cc.StoreOp(v, eleAddr)

        # We still use `i1` as the vector element type for `cc.StdvecInitOp`.
        vecTy = cc.StdvecType.get(elemTy) if not isBool else cc.StdvecType.get(
            self.getIntegerType(1))
        return cc.StdvecInitOp(vecTy, alloca, length=arrSize).result

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
        if structName == 'tuple':
            self.emitFatalError('`tuple` does not support attribute access')
        if not globalRegisteredTypes.isRegisteredClass(structName):
            self.emitFatalError(f'Dataclass is not registered: {structName})')

        _, userType = globalRegisteredTypes.getClassAttributes(structName)
        for i, (k, _) in enumerate(userType.items()):
            if k == memberName:
                structIdx = i
                break
        if structIdx == None:
            self.emitFatalError(
                f'Invalid struct member: {structName}.{memberName} '
                f'(members={[k for k,_ in userType.items()]})')
        return structIdx, mlirTypeFromPyType(userType[memberName], self.ctx)

    def __copyStructAndConvertElements(self,
                                       struct,
                                       expectedTy=None,
                                       allowDemotion=False,
                                       conversion=None):
        """
        Creates a new struct on the stack. If a conversion is provided, applies
        the conversion on each element before changing its type to match the
        corresponding element type in `expectedTy`.
        """
        assert cc.StructType.isinstance(struct.type)
        if not expectedTy:
            expectedTy = struct.type
        assert cc.StructType.isinstance(expectedTy)
        eleTys = cc.StructType.getTypes(struct.type)
        expectedEleTys = cc.StructType.getTypes(expectedTy)
        assert len(eleTys) == len(expectedEleTys)

        returnVal = cc.UndefOp(expectedTy)
        for idx, eleTy in enumerate(eleTys):
            element = cc.ExtractValueOp(
                eleTy, struct, [],
                DenseI32ArrayAttr.get([idx], context=self.ctx)).result
            element = conversion(idx, element) if conversion else element
            element = self.changeOperandToType(expectedEleTys[idx],
                                               element,
                                               allowDemotion=allowDemotion)
            returnVal = cc.InsertValueOp(
                expectedTy, returnVal, element,
                DenseI64ArrayAttr.get([idx], context=self.ctx)).result
        return returnVal

    # Create a new vector with source elements converted to the target element
    # type if needed.
    def __copyVectorAndConvertElements(self,
                                       source,
                                       targetEleType=None,
                                       allowDemotion=False,
                                       alwaysCopy=False,
                                       conversion=None):
        '''
        Creates a new vector with the requested element type.  Returns the
        original vector if the requested element type already matches the
        current element type unless `alwaysCopy` is set to True.  If a
        conversion is provided, applies the conversion to each element before
        changing its type to match the `targetEleType`.  If `alwaysCopy` is set
        to True, return a shallow copy of the vector by default (conversion can
        be used to create a deep copy).
        '''

        assert cc.StdvecType.isinstance(source.type)
        sourceEleType = cc.StdvecType.getElementType(source.type)
        if not targetEleType:
            targetEleType = sourceEleType
        if not alwaysCopy and sourceEleType == targetEleType:
            return source
        isSourceBool = sourceEleType == self.getIntegerType(1)
        if isSourceBool:
            sourceEleType = self.getIntegerType(8)
        isTargetBool = targetEleType == self.getIntegerType(1)
        if isTargetBool:
            targetEleType = self.getIntegerType(8)

        sourceArrPtrTy = cc.PointerType.get(cc.ArrayType.get(sourceEleType))
        sourceDataPtr = cc.StdvecDataOp(sourceArrPtrTy, source).result
        sourceSize = cc.StdvecSizeOp(self.getIntegerType(), source).result
        targetPtr = cc.AllocaOp(cc.PointerType.get(
            cc.ArrayType.get(targetEleType)),
                                TypeAttr.get(targetEleType),
                                seqSize=sourceSize).result

        rawIndex = DenseI32ArrayAttr.get([kDynamicPtrIndex], context=self.ctx)

        def bodyBuilder(iterVar):
            eleAddr = cc.ComputePtrOp(cc.PointerType.get(sourceEleType),
                                      sourceDataPtr, [iterVar], rawIndex).result
            loadedEle = cc.LoadOp(eleAddr).result
            convertedEle = conversion(iterVar,
                                      loadedEle) if conversion else loadedEle
            convertedEle = self.changeOperandToType(targetEleType,
                                                    convertedEle,
                                                    allowDemotion=allowDemotion)
            targetEleAddr = cc.ComputePtrOp(cc.PointerType.get(targetEleType),
                                            targetPtr, [iterVar],
                                            rawIndex).result
            cc.StoreOp(convertedEle, targetEleAddr)

        self.createInvariantForLoop(bodyBuilder, sourceSize)

        # We still use `i1` as the vector element type for `cc.StdvecInitOp`.
        vecTy = cc.StdvecType.get(
            targetEleType) if not isTargetBool else cc.StdvecType.get(
                self.getIntegerType(1))
        return cc.StdvecInitOp(vecTy, targetPtr, length=sourceSize).result

    def __copyAndValidateContainer(self, value, pyVal, deepCopy, dataType=None):
        """
        Helper function to implement deep and shallow copies for structs and
        vectors.

        Arguments:
            `value`: The MLIR value to copy
            `pyVal`: The Python AST node to use for validation of the container
	             entries.
            `deepCopy`: Whether to perform a deep or shallow copy.
            `dataType`: Must be None unless the value to copy is a vector.
                If the value is a vector, then the element type of the new
		vector.
        """

        # NOTE: Creating a copy means we are creating a new container.  As such,
        # all elements in the container need to pass the validation in
        # `__validate_container_entry`.
        if deepCopy:

            def conversion(idx, structItem):
                if cc.StdvecType.isinstance(structItem.type):
                    structItem = self.__copyVectorAndConvertElements(
                        structItem, alwaysCopy=True, conversion=conversion)
                elif (cc.StructType.isinstance(structItem.type) and
                      self.containsList(structItem.type)):
                    structItem = self.__copyStructAndConvertElements(
                        structItem, conversion=conversion)
                self.__validate_container_entry(structItem, pyVal)
                return structItem
        else:

            def conversion(idx, structItem):
                self.__validate_container_entry(structItem, pyVal)
                return structItem

        if cc.StdvecType.isinstance(value.type):
            listVal = self.__copyVectorAndConvertElements(value,
                                                          dataType,
                                                          alwaysCopy=True,
                                                          conversion=conversion)
            return listVal

        if cc.StructType.isinstance(value.type):
            if dataType:
                self.emitFatalError("unsupported data type argument",
                                    self.currentNode)
            struct = self.__copyStructAndConvertElements(value,
                                                         conversion=conversion)
            return struct

        self.emitFatalError(
            f'copy is not supported on value of type {value.type}',
            self.currentNode)

    def __migrateLists(self, value, migrate):
        """
        Replaces all lists in the given value by the list returned by the
        `migrate` function, including inner lists. Does an in-place replacement
        for list elements.
        """
        if cc.StdvecType.isinstance(value.type):
            eleTy = cc.StdvecType.getElementType(value.type)
            if self.containsList(eleTy):
                size = cc.StdvecSizeOp(self.getIntegerType(), value).result
                ptrTy = cc.PointerType.get(cc.ArrayType.get(eleTy))
                iterable = cc.StdvecDataOp(ptrTy, value).result

                def bodyBuilder(iterVar):
                    eleAddr = cc.ComputePtrOp(
                        cc.PointerType.get(eleTy), iterable, [iterVar],
                        DenseI32ArrayAttr.get([kDynamicPtrIndex],
                                              context=self.ctx))
                    loadedEle = cc.LoadOp(eleAddr).result
                    element = self.__migrateLists(loadedEle, migrate)
                    cc.StoreOp(element, eleAddr)

                self.createInvariantForLoop(bodyBuilder, size)
            return migrate(value)
        if (cc.StructType.isinstance(value.type) and
                self.containsList(value.type)):
            return self.__copyStructAndConvertElements(
                value, conversion=lambda _, v: self.__migrateLists(v, migrate))
        assert not self.containsList(value.type)
        return value

    def __insertDbgStmt(self, value, dbgStmt):
        """
        Insert a debug print out statement if the programmer requested. Handles
        statements like `cudaq.dbg.ast.print_i64(i)`.
        """
        printFunc = None
        printStr = '[cudaq-ast-dbg] '
        argsTy = [cc.PointerType.get(self.getIntegerType(8))]
        if dbgStmt == 'print_i64':
            if not IntegerType.isinstance(value.type):
                self.emitFatalError(
                    f"print_i64 requested, but value is not of integer "
                    f"type (type was {value.type}).")

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
                    f"print_f64 requested, but value is not of float "
                    f"type (type was {value.type}).")

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
            elem_ty = cc.StdvecType.getElementType(vector.type)
            is_bool = elem_ty == self.getIntegerType(1)
            # std::vector<bool> is a special case in C++ where each element is
            # stored as a single bit, but the underlying array is actually an
            # array of `i8` values.
            if is_bool:
                # `i1` elements are stored as `i8` in the underlying array.
                elem_ty = self.getIntegerType(8)
            data_ptr = cc.StdvecDataOp(
                cc.PointerType.get(cc.ArrayType.get(elem_ty)), vector).result
            load_val = cc.LoadOp(
                cc.ComputePtrOp(cc.PointerType.get(elem_ty), data_ptr, [index],
                                DenseI32ArrayAttr.get([kDynamicPtrIndex
                                                      ]))).result
            if is_bool:
                # Cast back to `i1` if the original vector element type was `i1`.
                load_val = self.changeOperandToType(self.getIntegerType(1),
                                                    load_val)
            return load_val
        return cc.LoadOp(
            cc.ComputePtrOp(
                cc.PointerType.get(
                    cc.ArrayType.getElementType(
                        cc.PointerType.getElementType(vector.type))), vector,
                [index], DenseI32ArrayAttr.get([kDynamicPtrIndex]))).result

    def __get_superior_type(self, t1, t2):
        """
        Get the superior numeric type between two MLIR types.

        Complex > F64 > F32 > Integer, with integers and complex promoting to
        the wider width.  Returns None if no superior type can be determined.

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

    def createForLoop(self,
                      argTypes,
                      bodyBuilder,
                      inputs,
                      evalCond,
                      evalStep,
                      orElseBuilder=None):

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

    def createMonotonicForLoop(self,
                               bodyBuilder,
                               startVal,
                               stepVal,
                               endVal,
                               isDecrementing=False,
                               orElseBuilder=None):

        iTy = self.getIntegerType()
        assert startVal.type == iTy
        assert stepVal.type == iTy
        assert endVal.type == iTy

        condPred = IntegerAttr.get(
            iTy, 4) if isDecrementing else IntegerAttr.get(iTy, 2)
        return self.createForLoop(
            [iTy], lambda args: bodyBuilder(args[0]), [startVal],
            lambda args: arith.CmpIOp(condPred, args[0], endVal).result,
            lambda args: [arith.AddIOp(args[0], stepVal).result],
            None if orElseBuilder is None else
            (lambda args: orElseBuilder(args[0])))

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
            if (isinstance(value, ast.Tuple) or isinstance(value, ast.List)):
                nrArgs = len(value.elts)
                getItem = lambda idx: value.elts[idx]
            elif (isinstance(value, tuple) or isinstance(value, list)):
                nrArgs = len(value)
                getItem = lambda idx: value[idx]
            elif cc.StructType.isinstance(value.type):
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
            elif (isinstance(pyVals[2], ast.UnaryOp) and
                  isinstance(pyVals[2].op, ast.USub) and
                  isinstance(pyVals[2].operand, ast.Constant)):
                pyStepVal = -pyVals[2].operand.value
            else:
                self.emitFatalError('range step value must be a constant',
                                    self.currentNode)
            if pyStepVal == 0:
                self.emitFatalError("range step value must be non-zero",
                                    self.currentNode)
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

        for idx, v in enumerate([startVal, endVal, stepVal]):
            if not IntegerType.isinstance(v.type):
                # matching Python behavior to error on non-integer values
                self.emitFatalError("non-integer value in range expression",
                                    pyVals[idx if len(pyVals) > 1 else 0])
        return startVal, endVal, stepVal, isDecrementing

    def __groupValues(self, pyvals, groups: list[int | tuple[int, int]]):
        '''
        Helper function that visits the given AST nodes (`pyvals`), and groups
        them according to the specified list.  The list contains integers or
        tuples of two integers.  Integer values have to be positive or -1, where
        -1 indicates that any number of values is acceptable.  Tuples of two
        integers (min, max) indicate that any number of values in [min, max] is
        acceptable.  The list may only contain at most one negative integer or
        tuple (enforced via assert only).

        Emits a fatal error if any of the given `pyvals` did not generate a
        value. Emits a fatal error if there are too many or too few values to
        satisfy the requested grouping.

        Returns a tuple of value groups. Each value group is either a single
        value (if the corresponding entry in `groups` equals 1), or a list of
        values.
	'''

        def group_values(numExpected, values, reverse):
            groupedVals = []
            current_idx = 0
            for nArgs in numExpected:
                if (isinstance(nArgs, int) and nArgs == 1 and
                        current_idx < len(values)):
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
                groupedVals.append(values[current_idx:current_idx + nArgs])
                if reverse:
                    groupedVals[-1].reverse()
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
            assert len(groups) == 1  # ambiguous otherwise
            if isinstance(groups[0], tuple):
                minNumArgs, maxNumArgs = groups[0]
                assert 0 <= minNumArgs and (minNumArgs <= maxNumArgs or
                                            maxNumArgs < 0)
                if len(values) < minNumArgs:
                    self.emitFatalError("missing value", self.currentNode)
                if len(values) > maxNumArgs and maxNumArgs > 0:
                    self.emitFatalError("too many values", self.currentNode)
            groupedVals = *frontVals, values, *backVals
        return groupedVals[0] if len(groupedVals) == 1 else groupedVals

    def __get_root_value(self, pyVal):
        '''
        Strips any attribute and subscript expressions from the node to get the
        root node that the expression accesses.  Returns the symbol table entry
        for the root node, if such an entry exists, and return None otherwise.
        '''
        pyValRoot = pyVal
        while (isinstance(pyValRoot, ast.Subscript) or
               isinstance(pyValRoot, ast.Attribute)):
            pyValRoot = pyValRoot.value
        if (isinstance(pyValRoot, ast.Name) and
                pyValRoot.id in self.symbolTable):
            return self.symbolTable[pyValRoot.id]
        return None

    def __validate_container_entry(self, mlirVal, pyVal):
        '''
        Helper function that should be invoked for any elements that are stored
        in tuple, `dataclass`, or list. Note that the `pyVal` argument is only
        used to determine the root of `mlirVal` and as such could be either the
        Python AST node matching the container item (`mlirVal`) or the AST node
        for the container itself.
        '''

        rootVal = self.__get_root_value(pyVal)
        assert rootVal or not self.isFunctionArgument(mlirVal)

        if cc.PointerType.isinstance(mlirVal.type):
            # We do not allow to create container that contain pointers.
            valTy = cc.PointerType.getElementType(mlirVal.type)
            assert cc.StateType.isinstance(valTy)
            if cc.StateType.isinstance(valTy):
                self.emitFatalError(
                    "cannot use `cudaq.State` as element in lists, tuples, "
                    "or dataclasses", self.currentNode)
            self.emitFatalError(
                "lists, tuples, and dataclasses must not "
                "contain modifiable values", self.currentNode)

        if cc.StructType.isinstance(mlirVal.type):
            structName = cc.StructType.getName(mlirVal.type)
            # We need to give a proper error if we try to assign a mutable
            # `dataclass` to an item in another container. Allowing this would
            # lead to incorrect behavior (i.e. inconsistent with Python) unless
            # we change the representation of structs to be like `StdvecType`
            # where we have a container that is passed by value wrapping the
            # actual pointer, thus ensuring that the reference behavior actually
            # works across function boundaries.
            if structName != 'tuple' and rootVal:
                msg = ("only dataclass literals may be used as items in other "
                       "container values")
                self.emitFatalError(
                    f"{msg} - use `.copy(deep)` to create a new {structName}",
                    self.currentNode)

        if (self.knownResultType and self.containsList(self.knownResultType) and
                self.containsList(mlirVal.type)):
            # For lists that were created inside a kernel, we have to copy the
            # stack allocated array to the heap when we return such a list. In
            # the case where the list was created by the caller, this copy leads
            # to incorrect behavior (i.e. not matching Python behavior). We
            # hence want to make sure that we can know when a host allocated
            # list is returned. If we allow to assign lists passed as function
            # arguments to inner items of other lists and `dataclasses`, we
            # loose the information that this list was allocated by the parent.
            # We hence forbid this. All of this applies regardless of how the
            # list was passed (e.g. the list might be an inner item in a tuple
            # or `dataclass` that was passed) or how it is assigned (e.g. the
            # assigned value might be a tuple or `dataclass` that contains a
            # list).
            if rootVal and self.isFunctionArgument(rootVal):
                msg = ("lists passed as or contained in function arguments "
                       "cannot be inner items in other container values when a "
                       "list is returned")
                self.emitFatalError(
                    f"{msg} - use `.copy(deep)` to create a new list",
                    self.currentNode)

    def visit(self, node):
        self.debug_msg(lambda: f'[Visit {type(node).__name__}]', node)
        self.indent_level += 1
        parentNode = self.currentNode
        self.currentNode = node
        numVals = 0 if isinstance(
            node, ast.Module) else self.valueStack.currentNumValues
        self.valueStack.pushFrame()
        super().visit(node)
        self.valueStack.popFrame()
        if isinstance(node, ast.Module):
            if not self.valueStack.isEmpty:
                self.emitFatalError(
                    "processing error - unprocessed frame(s) in value stack",
                    node)
        elif self.valueStack.currentNumValues - numVals > 1:
            # Do **NOT** change this to be more permissive and allow multiple
            # values to be pushed without pushing proper frames for sub-nodes.
            # If visiting a single node potentially produces more than one
            # value, the bridge quickly will be a mess because we will easily
            # end up with values in the wrong places.
            self.emitFatalError(
                "must not generate more one value at a time in each frame",
                node)
        self.currentNode = parentNode
        self.indent_level -= 1

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
            # This is an inner function def, we will treat it as a cc.callable
            # (cc.create_lambda)
            self.debug_msg(lambda: f'Visiting inner FunctionDef {node.name}')

            arguments = node.args.args
            if len(arguments):
                self.emitFatalError(
                    "inner function definitions cannot have arguments.", node)

            ty = cc.CallableType.get(self.ctx, [], [])
            createLambda = cc.CreateLambdaOp(ty)
            initRegion = createLambda.initRegion
            initBlock = Block.create_at_start(initRegion, [])
            # TODO: process all captured variables in the main function
            # definition first to avoid reusing code not defined in the same or
            # parent scope of the produced MLIR.
            with InsertionPoint(initBlock):
                [self.visit(n) for n in node.body]
                cc.ReturnOp([])
            self.symbolTable[node.name] = createLambda.result
            return

        with self.ctx, InsertionPoint(self.module.body), self.loc:

            # Get the potential documentation string
            self.docstring = ast.get_docstring(node)

            # Get the argument types and argument names this will throw an error
            # if the types aren't annotated
            self.argTypes = [
                self.mlirTypeFromAnnotation(arg.annotation)
                for arg in node.args.args
            ]
            parentResultType = self.knownResultType
            if node.returns is not None and not (isinstance(
                    node.returns, ast.Constant) and
                                                 (node.returns.value is None)):
                self.knownResultType = self.mlirTypeFromAnnotation(node.returns)

            # Add uniqueness. In MLIR, we require unique symbols (`bijective`
            # function between symbols and artifacts) even if Python allows
            # hiding symbols and replacing symbols (dynamic `injective` function
            # between scoped symbols and artifacts).
            self.name = node.name + ".." + hex(self.uniqueId)
            self.capturedDataStorage.name = self.name

            # the full function name in MLIR is `__nvqpp__mlirgen__` + the
            # function name
            if self.disableNvqppPrefix:
                fullName = self.name
            else:
                fullName = nvqppPrefix + self.name

            # Create the FuncOp
            f = func.FuncOp(fullName, (self.argTypes, [] if self.knownResultType
                                       == None else [self.knownResultType]),
                            loc=self.loc)
            self.kernelFuncOp = f

            # Set this kernel as an entry point if the argument types are
            # classical only
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
                        assignNode.targets = [
                            ast.Tuple(
                                [ast.Name(arg.arg) for arg in node.args.args])
                        ]
                        assignNode.value = [
                            self.entry.arguments[idx]
                            for idx in range(len(self.entry.arguments.types))
                        ]
                    assignNode.lineno = node.lineno
                    self.visit_Assign(assignNode)

                # Intentionally set after we process the argument assignment,
                # since we currently treat value vs reference semantics slightly
                # differently when we have arguments vs when we have local
                # values.  To not make this distinction, we would need to add
                # support for having proper reference arguments, which we don't
                # want to.  Barring that, we at least try to be nice and give
                # errors on assignments that may lead to unexpected behavior
                # (i.e. behavior not following expected Python behavior).

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
            # An `ast.Expr` object is created when an expression is used as a
            # statement. This expression may produce a value, which is ignored
            # (not assigned) in the Python code. We hence need to pop that value
            # to match that behavior and ignore it.
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
            functor = lambda : (h(qubits), x(qubits), ry(np.pi, qubits))
	                                 # ^^ qubits captured from parent region
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

        ty = cc.CallableType.get(self.ctx, [], [])
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
        assignment in the MLIR. This method handles assignments, item updates,
        as well as deconstruction.

        For all assignments, the variable name will be used as a key for the
        symbol table, mapping to the corresponding MLIR Value. Quantum values,
        measurements results, `cc.callable`, and `cc.stdvec` will be stored as
        values in the symbol table.  For all other values, the variable will be
        allocated with a `cc.alloca` op, and the pointer will be stored in
        the symbol table.
        """

        # FIXME: Measurement results are stored as values
        # to preserve their origin from discriminate.
        # This should be revised when we introduce the proper
        # type distinction.
        def storedAsValue(val):
            varTy = val.type
            if cc.PointerType.isinstance(varTy):
                varTy = cc.PointerType.getElementType(varTy)
            # If `buildingEntryPoint` is not set we are processing function
            # arguments. Function arguments are always passed by value,
            # except states. We can treat non-container function arguments
            # like any local variable and create a stack slot for them.
            # For container types, on the the other hand, we need to preserve
            # them as values in the symbol table to make sure we can detect
            # any access to reference types that are function arguments, or
            # function argument items.
            containerFuncArg = (not self.buildingEntryPoint and
                                (cc.StructType.isinstance(varTy) or
                                 cc.StdvecType.isinstance(varTy)))
            # FIXME: Consider storing vectors and callables as pointers like
            # other variables.
            storeAsVal = (containerFuncArg or self.isQuantumType(varTy) or
                          cc.CallableType.isinstance(varTy) or
                          cc.StdvecType.isinstance(varTy) or
                          self.isMeasureResultType(varTy, val))
            # Nothing should ever produce a pointer to a type we store as value
            # in the symbol table.
            assert (not storeAsVal or not cc.PointerType.isinstance(val.type))
            return storeAsVal

        def process_assignment(target, value):

            if isinstance(target, ast.Tuple):

                if (isinstance(value, ast.Tuple) or
                        isinstance(value, ast.List)):
                    return target, value

                if isinstance(value, ast.AST):
                    # Measurements need to push their values to the stack, so we
                    # set a so we set a non-None variable name here.
                    self.currentAssignVariableName = ''
                    # NOTE: The way the assignment logic is processed, including
                    # that we load this value for the purpose of deconstruction,
                    # does not preserve any inner references. There are a bunch
                    # of issues that prevent us from properly dealing with any
                    # reference types stored as items in lists and
                    # `dataclasses`. We hence currently prevent the creation of
                    # such lists and `dataclasses`, and would need to change the
                    # representation for `dataclasses` to allow that.
                    self.visit(value)
                    value = self.popValue()
                    self.currentAssignVariableName = None
                    return target, value

                return target, value

            # Make sure we process arbitrary combinations
            # of subscript and attributes
            target_root = target
            while (isinstance(target_root, ast.Subscript) or
                   isinstance(target_root, ast.Attribute)):
                target_root = target_root.value
            if not isinstance(target_root, ast.Name):
                self.emitFatalError("invalid target for assignment", node)
            target_root_defined_in_parent_scope = (
                target_root.id in self.symbolTable and
                not self.symbolTable.isInCurrentScope(target_root.id))
            value_root = self.__get_root_value(value)

            def update_in_parent_scope(destination, value):
                assert not cc.PointerType.isinstance(value.type)
                if cc.StructType.isinstance(
                        value.type) and cc.StructType.getName(
                            value.type) != 'tuple':

                    # We can't properly deal with this case if the value we are
                    # assigning is not an `rvalue`. Consider the case were we
                    # have `v1` defined in the parent scope, `v2` in a child
                    # scope, and we are assigning v2 to v1 in the child
                    # scope. To do this assignment properly, we would need to
                    # make sure that the pointers for both v1 and v2 points to
                    # the same memory location such that any changes to v1 after
                    # the assignment are reflected in v2 and vice versa (v2
                    # could be changed in the child while v1 is still
                    # alive). Since we merely store the raw pointer in the
                    # symbol table for `dataclasses`, we have no way of updating
                    # that pointer conditionally on the child scope being
                    # executed.  To determine whether the value we assign is an
                    # `rvalue`, it is sufficient to check whether its root is a
                    # value in the symbol table (values returned from calls are
                    # never `lvalues`).

                    if value_root:

                        # Note that this check also makes sure that function
                        # arguments are not assigned to local variables, since
                        # function arguments are in the symbol table.

                        self.emitFatalError(
                            "only literals can be assigned to variables defined"
                            " in parent scope - use `.copy(deep)` to create a "
                            "new value that can be assigned", node)
                if cc.StdvecType.isinstance(destination.type):
                    # In this case, we are assigning a list to a variable in a
                    #  parent scope.
                    assert isinstance(target, ast.Name)
                    # If the value we are assigning is an `rvalue` then we can
                    # do an in-place update of the data in the parent; the
                    # restrictions for container items in
                    # `__validate_container_entry` ensure that the value we are
                    # assigning does not contain any references to `dataclass`
                    # values, and any lists contained in the value behave like
                    # proper references since they contain a data pointer
                    # (i.e. in-place update only does a shallow copy).  TODO:
                    # The only reason we cannot currently support this is
                    # because we have no way of updating the size of an existing
                    # vector...
                    self.emitFatalError(
                        "variable defined in parent scope cannot be modified",
                        node)
                # Allowing to assign vectors to container items in the parent
                # scope should be fine regardless of whether the assigned value
                # is an `rvalue` or not; replacing the item in the container
                # with the value leads to the correct behavior much like it does
                # for the case where the target is defined in the same scope.
                # NOTE: The assignment is subject to the usual restrictions for
                # container items - these should be validated before calling
                # update_in_parent_scope.
                if not cc.StdvecType.isinstance(
                        value.type) and storedAsValue(destination):
                    # We can't properly deal with this, since there is no way to ensure that
                    # the target in the symbol table is updated conditionally on the child
                    # scope executing.
                    self.emitFatalError(
                        "variable defined in parent scope cannot be modified",
                        node)
                assert cc.PointerType.isinstance(destination.type)
                expectedTy = cc.PointerType.getElementType(destination.type)
                value = self.changeOperandToType(expectedTy,
                                                 value,
                                                 allowDemotion=False)
                cc.StoreOp(value, destination)

            # Handle assignment `var = expr`
            if isinstance(target, ast.Name):

                # This is so that we properly preserve the references to local
                # variables. These variables can be of a reference type and
                # other values in the symbol table may be assigned to the same
                # reference. It is hence important to keep the reference as is,
                # since otherwise changes to it would not be reflected in other
                # values.  NOTE: we don't need to worry about any references in
                # values that are not `ast.Name` objects, since we don't allow
                # containers to contain references.

                value_is_name = False
                if (isinstance(value, ast.Name) and
                        value.id in self.symbolTable):
                    value_is_name = True
                    value = self.symbolTable[value.id]
                if isinstance(value, ast.AST):
                    # Retain the variable name for potential children (like
                    # `mz(q, registerName=...)`)
                    self.currentAssignVariableName = target.id
                    self.visit(value)
                    value = self.popValue()
                    self.currentAssignVariableName = None
                storeAsVal = storedAsValue(value)

                if value_root and self.isFunctionArgument(value_root):
                    # If we assign a function argument or argument item to a
                    # local variable, we need to be careful to not loose the
                    # information about contained lists that have been allocated
                    # by the caller, if the return value contains any
                    # lists. This is problematic for reasons commented in
                    # `__validate_container_entry`.
                    if (cc.StdvecType.isinstance(value.type) and
                            self.knownResultType and
                            self.containsList(self.knownResultType)):
                        # We loose this information if we assign an item of a
                        # function argument.
                        if not value_is_name:
                            self.emitFatalError(
                                "lists passed as or contained in function "
                                "arguments cannot be assigned to to a local "
                                "variable when a list is returned - use "
                                "`.copy(deep)` to create a new value that can"
                                " be assigned", node)
                        # We also loose this information if we assign to a value
                        # in the parent scope.
                        elif target_root_defined_in_parent_scope:
                            self.emitFatalError(
                                "lists passed as or contained in function "
                                "arguments cannot be assigned to variables in "
                                "the parent scope when a list is returned - use"
                                " `.copy(deep)` to create a new value that can "
                                "be assigned", node)
                    if cc.StructType.isinstance(value.type):
                        structName = cc.StructType.getName(value.type)

                        # For `dataclasses`, we have to do an additional check
                        # to ensure that their behavior (for cases that don't
                        # give an error) is consistent with Python; since we
                        # pass them by value across functions, we either have to
                        # force that an explicit copy is made when using them as
                        # call arguments, or we have to force that an explicit
                        # copy is made when a `dataclass` argument is assigned
                        # to a local variable (as long as it is not assigned, it
                        # will not be possible to make any modification to it
                        # since the argument itself is represented as an
                        # immutable value). The latter seems more comprehensive
                        # and also ensures that there is no unexpected behavior
                        # with regards to kernels not being able to modify
                        # `dataclass` values in host code.  NOTE: It is
                        # sufficient to check the value itself (not its root) is
                        # a function argument, (only!) since inner items are
                        # never references to `dataclasses` (enforced in
                        # `__validate_container_entry`).

                        if value_is_name and structName != 'tuple':
                            self.emitFatalError(
                                "cannot assign dataclass passed as function "
                                "argument to a local variable - use "
                                "`.copy(deep)` to create a new value that can "
                                "be assigned", node)
                        elif (self.knownResultType and
                              self.containsList(self.knownResultType) and
                              self.containsList(value.type)):
                            self.emitFatalError(
                                "cannot assign tuple or dataclass passed as "
                                "function argument to a local variable if it "
                                "contains a list when a list is returned - use"
                                " `.copy(deep)` to create a new value that can"
                                " be assigned", node)

                if target_root_defined_in_parent_scope:
                    if cc.PointerType.isinstance(value.type):
                        # This is fine since/as long as update_in_parent_scope
                        # validates that `lvalues` of reference types cannot be
                        # assigned. Note that tuples and states are value types.
                        value = cc.LoadOp(value).result
                    destination = self.symbolTable[target.id]
                    update_in_parent_scope(destination, value)
                    return target, None

                # The target variable has either not been defined or is defined
                # within the current scope; we can simply modify the symbol
                # table entry.

                if storeAsVal or cc.PointerType.isinstance(value.type):
                    return target, value

                with InsertionPoint.at_block_begin(self.entry):
                    address = cc.AllocaOp(cc.PointerType.get(value.type),
                                          TypeAttr.get(value.type)).result
                cc.StoreOp(value, address)
                return target, address

            # Handle updates of existing variables (target is a combination of
            # attribute and subscript)

            self.pushPointerValue = True
            self.visit(target)
            destination = self.popValue()
            self.pushPointerValue = False

            # We should have a pointer since we requested a pointer.
            assert cc.PointerType.isinstance(destination.type)
            expectedTy = cc.PointerType.getElementType(destination.type)

            # We prevent the creation of lists and structs that contain
            # pointers, and prevent obtaining pointers to quantum types.
            assert not cc.PointerType.isinstance(expectedTy)
            assert not self.isQuantumType(expectedTy)

            if not isinstance(value, ast.AST):
                # Can arise if have something like `l[0], l[1] = getTuple()`
                self.emitFatalError(
                    "updating lists or dataclasses as part of deconstruction is not supported",
                    node)

            # Measurements need to push their values to the stack, so we set a
            # so we set a non-None variable name here.

            self.currentAssignVariableName = ''
            self.visit(value)
            mlirVal = self.popValue()
            self.currentAssignVariableName = None
            assert not cc.PointerType.isinstance(mlirVal.type)

            # Must validate the container entry regardless of what scope the
            # target is defined in.
            self.__validate_container_entry(mlirVal, value)

            if target_root_defined_in_parent_scope:
                update_in_parent_scope(destination, mlirVal)
                return target_root, None

            mlirVal = self.changeOperandToType(expectedTy,
                                               mlirVal,
                                               allowDemotion=False)
            cc.StoreOp(mlirVal, destination)
            # The returned target root has no effect here since no value is
            # returns to push to he symbol table. We merely need to make sure
            # that it is an `ast.Name` object to break the recursion.
            return target_root, None

        if len(node.targets) > 1:
            # I am not entirely sure what kinds of Python language constructs
            # would result in having more than 1 target here, hence giving an
            # error on it for now.  (It would be easy to process this as target
            # tuple, but it may not be correct to do so.)
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
                    self.emitFatalError(
                        "noise channels may only be used as part of call expressions",
                        node)

                # must be handled by the parent
                return

            if node.attr == 'ctrl' or node.attr == 'adj':
                # to be processed by the caller
                return

        if node.attr == 'copy':
            if self.pushPointerValue:
                self.emitFatalError(
                    "function call does not produce a modifiable value", node)
            # needs to be handled by the caller
            return

        # Only variable names, subscripts and attributes can produce modifiable
        # values. Anything else produces an immutable value. We make sure the
        # visit gets processed such that the rest of the code can give a proper
        # error.
        value_root = node.value
        while (isinstance(value_root, ast.Subscript) or
               isinstance(value_root, ast.Attribute)):
            value_root = value_root.value
        if self.pushPointerValue and not isinstance(value_root, ast.Name):
            self.pushPointerValue = False
            self.visit(node.value)
            value = self.popValue()
            self.pushPointerValue = True
        else:
            self.visit(node.value)
            value = self.popValue()

        valType = value.type
        if cc.PointerType.isinstance(valType):
            valType = cc.PointerType.getElementType(valType)

        if quake.StruqType.isinstance(valType):
            if self.pushPointerValue:
                self.emitFatalError(
                    "accessing attribute of quantum tuple or dataclass does "
                    "not produce a modifiable value", node)
            # Need to extract value instead of load from compute pointer.
            structIdx, memberTy = self.getStructMemberIdx(node.attr, value.type)
            attr = IntegerAttr.get(self.getIntegerType(32), structIdx)
            self.pushValue(quake.GetMemberOp(memberTy, value, attr).result)
            return

        if (cc.PointerType.isinstance(value.type) and
                cc.StructType.isinstance(valType)):
            assert self.pushPointerValue
            structIdx, memberTy = self.getStructMemberIdx(node.attr, valType)
            eleAddr = cc.ComputePtrOp(cc.PointerType.get(memberTy), value, [],
                                      DenseI32ArrayAttr.get([structIdx])).result

            if self.pushPointerValue:
                self.pushValue(eleAddr)
                return

            eleAddr = cc.LoadOp(eleAddr).result
            self.pushValue(eleAddr)
            return

        if cc.StructType.isinstance(value.type):
            if self.pushPointerValue:
                self.emitFatalError(
                    "value cannot be modified - use `.copy(deep)` to create "
                    "a new value that can be modified", node)

            # Handle direct struct value - use ExtractValueOp (more efficient)
            structIdx, memberTy = self.getStructMemberIdx(node.attr, value.type)
            extractedValue = cc.ExtractValueOp(
                memberTy, value, [], DenseI32ArrayAttr.get([structIdx])).result

            self.pushValue(extractedValue)
            return

        if (quake.VeqType.isinstance(valType) or
                cc.StdvecType.isinstance(valType)):
            if self.__isSupportedVectorFunction(node.attr):
                if self.pushPointerValue:
                    self.emitFatalError(
                        "function call does not produce a modifiable value",
                        node)
                # needs to be handled by the caller
                return

        # everything else does not produce a modifiable value
        if self.pushPointerValue:
            self.emitFatalError(
                "attribute expression does not produce a modifiable value")

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
                return
            if cc.StdvecType.isinstance(value.type):
                self.pushValue(
                    cc.StdvecSizeOp(self.getIntegerType(), value).result)
                return

        self.emitFatalError("unrecognized attribute {}".format(node.attr), node)

    def find_unique_decorator_name(self, name):
        mod = sys.modules[self.kernelModuleName]
        if mod:
            if hasattr(mod, name):
                from .kernel_decorator import isa_kernel_decorator
                result = getattr(mod, name)
                if isa_kernel_decorator(result):
                    return name + ".." + hex(id(result))
        return None

    def visit_Call(self, node):
        """
        Map a Python Call operation to equivalent MLIR. This method handles
        functions that are `ast.Name` and `ast.Attribute` objects.

        This function handles all built-in unitary and measurement gates as well
        as all the ways to adjoint and control them.  General calls to
        previously seen CUDA-Q kernels or registered operations are supported.

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

        def copy_list_to_stack(value):
            symName = '__nvqpp_vectorCopyToStack'
            load_intrinsic(self.module, symName)
            elemTy = cc.StdvecType.getElementType(value.type)
            if elemTy == self.getIntegerType(1):
                elemTy = self.getIntegerType(8)
            ptrTy = cc.PointerType.get(self.getIntegerType(8))
            ptrArrTy = cc.PointerType.get(cc.ArrayType.get(elemTy))
            resBuf = cc.StdvecDataOp(ptrArrTy, value).result
            eleSize = cc.SizeOfOp(self.getIntegerType(),
                                  TypeAttr.get(elemTy)).result
            dynSize = cc.StdvecSizeOp(self.getIntegerType(), value).result
            resBuf = cc.CastOp(cc.PointerType.get(elemTy), resBuf)
            stackCopy = cc.AllocaOp(cc.PointerType.get(
                cc.ArrayType.get(elemTy)),
                                    TypeAttr.get(elemTy),
                                    seqSize=dynSize).result
            func.CallOp([], symName, [
                cc.CastOp(ptrTy, stackCopy).result,
                cc.CastOp(ptrTy, resBuf).result,
                arith.MulIOp(dynSize, eleSize).result
            ])
            return cc.StdvecInitOp(value.type, stackCopy, length=dynSize).result

        def convertArguments(expectedArgTypes, values):
            assert len(expectedArgTypes) == len(values)
            args = []
            for idx, expectedTy in enumerate(expectedArgTypes):
                arg = self.changeOperandToType(expectedTy,
                                               values[idx],
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

        def processFunctionCall(kernel):
            nrArgs = len(kernel.type.inputs)
            values = self.__groupValues(node.args, [(nrArgs, nrArgs)])
            values = convertArguments([t for t in kernel.type.inputs], values)
            if len(kernel.type.results) == 0:
                func.CallOp(kernel, values)
                return

            # The logic for calls that return values must match the logic in
            # `visit_Return`; anything copied to the heap during return must be
            # copied back to the stack. Compiler optimizations should take care
            # of eliminating unnecessary copies.
            result = func.CallOp(kernel, values).result
            return self.__migrateLists(result, copy_list_to_stack)

        def checkControlAndTargetTypes(controls, targets):
            """
            Check that the provided control and target operands are of an
            appropriate type. Emit a fatal error if not.
            """

            def is_qvec_or_qubits(vals):
                # We can either have a single item that is a vector of qubits,
                # or multiple single-qubit items.
                return (all((quake.RefType.isinstance(v.type) for v in vals)) or
                        (len(vals) == 1 and
                         quake.VeqType.isinstance(vals[0].type)))

            if len(controls) > 0 and not is_qvec_or_qubits(controls):
                self.emitFatalError(
                    f'invalid argument type for control operand', node)
            if len(targets) == 0:
                self.emitFatalError(f'missing argument for target operand',
                                    node)
            elif not is_qvec_or_qubits(targets):
                self.emitFatalError(f'invalid argument type for target operand',
                                    node)

        def processQuantumOperation(opName,
                                    controls,
                                    targets,
                                    *args,
                                    broadcast=lambda q: [q],
                                    **kwargs):
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

        def processQuakeCtor(opName,
                             pyArgs,
                             isCtrl,
                             isAdj,
                             numParams=0,
                             numTargets=1):
            kwargs = {}
            if isCtrl:
                argGroups = [(numParams, numParams), (1, -1),
                             (numTargets, numTargets)]
                # FIXME: we could allow this as long as we have 1 target
                kwargs['broadcast'] = False
            elif numTargets == 1:
                # when we have a single target and no controls, we generally
                # support any version of `x(qubit)`, `x(qvector)`, `x(q, r)`
                argGroups = [(numParams, numParams), 0, (1, -1)]
            else:
                argGroups = [(numParams, numParams), 0,
                             (numTargets, numTargets)]
                kwargs['broadcast'] = False

            params, controls, targets = self.__groupValues(pyArgs, argGroups)
            if isCtrl:
                negatedControlQubits = getNegatedControlQubits(controls)
                kwargs['negated_qubit_controls'] = negatedControlQubits
            if isAdj:
                kwargs['is_adj'] = True
            params = [
                self.changeOperandToType(self.getFloatType(), param)
                for param in params
            ]
            processQuantumOperation(opName, controls, targets, [], params,
                                    **kwargs)

        def processDecorator(name, path=None):
            if path:
                name = f"{path}.{name}"
                decorator = resolve_qualified_symbol(name)
            else:
                decorator = recover_kernel_decorator(name)

            if decorator and not name in self.symbolTable:
                self.appendToLiftedArgs(name)
                entryPoint = recover_func_op(decorator.qkeModule,
                                             nvqppPrefix + decorator.uniqName)
                funcTy = FunctionType(
                    TypeAttr(entryPoint.attributes['function_type']).value)
                callableTy = cc.CallableType.get(
                    self.ctx, funcTy.inputs[:decorator.firstLiftedPos],
                    funcTy.results)

                # `callee` will be a new `BlockArgument`
                callee = cudaq_runtime.appendKernelArgument(
                    self.kernelFuncOp, callableTy)
                self.argTypes.append(callableTy)
                self.symbolTable.add(name, callee, 0)

            return name if decorator else None

        # FIXME: unify with `processFunctionCall`?
        def processDecoratorCall(symName):
            assert symName in self.symbolTable
            self.visit(ast.Name(symName))
            kernel = self.popValue()
            if not cc.CallableType.isinstance(kernel.type):
                self.emitFatalError(
                    f"`{symName}` object is not callable, found symbol of type {kernel.type}",
                    node)
            functionTy = FunctionType(
                cc.CallableType.getFunctionType(kernel.type))
            nrArgs = len(functionTy.inputs)
            values = self.__groupValues(node.args, [(nrArgs, nrArgs)])
            values = convertArguments([t for t in functionTy.inputs], values)
            call = cc.CallCallableOp(functionTy.results, kernel, values)
            call.attributes.__setitem__('symbol', StringAttr.get(symName))

            if len(functionTy.results) == 0:
                return
            if len(functionTy.results) == 1:
                result = call.results[0]
            else:
                # FIXME: SPLIT OUT INTO HELPER FUNCTION
                for res in call.results:
                    self.__validate_container_entry(res, node)
                structTy = mlirTryCreateStructType(functionTy.results,
                                                   name='tuple',
                                                   context=self.ctx)
                if structTy is None:
                    self.emitFatalError(
                        "Hybrid quantum-classical data types and nested "
                        "quantum structs are not allowed.", node)
                if quake.StruqType.isinstance(structTy):
                    result = quake.MakeStruqOp(structTy, call.results).result
                else:
                    result = cc.UndefOp(structTy)
                    for idx, element in enumerate(call.results):
                        result = cc.InsertValueOp(
                            structTy, result, element,
                            DenseI64ArrayAttr.get([idx],
                                                  context=self.ctx)).result

            # The logic for calls that return values must match the logic in
            # `visit_Return`; anything copied to the heap during return must be
            # copied back to the stack. Compiler optimizations should take care
            # of eliminating unnecessary copies.
            return self.__migrateLists(result, copy_list_to_stack)

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
            moduleNames = []
            value = node.func.value
            while isinstance(value, ast.Attribute):
                self.debug_msg(lambda: f'[(Inline) Visit Attribute]', value)
                moduleNames.append(value.attr)
                value = value.value
            if isinstance(value, ast.Name):
                self.debug_msg(lambda: f'[(Inline) Visit Name]', value)
                moduleNames.append(value.id)
                moduleNames.reverse()

                devKey = '.'.join(moduleNames)
                for module_name, module in sys.modules.items():
                    if module_name.split('.')[-1] == moduleNames[0]:
                        try:
                            obj = module
                            for part in moduleNames[1:]:
                                obj = getattr(obj, part)
                            devKey = f"{module_name}.{'.'.join(moduleNames[1:])}" if len(
                                moduleNames) > 1 else module_name
                        except AttributeError:
                            continue

                # Handle registered C++ kernels
                if cudaq_runtime.isRegisteredDeviceModule(devKey):
                    maybeKernelName = cudaq_runtime.checkRegisteredCppDeviceKernel(
                        self.module, devKey + '.' + node.func.attr)
                    if maybeKernelName == None:
                        maybeKernelName = cudaq_runtime.checkRegisteredCppDeviceKernel(
                            self.module, devKey)
                    if maybeKernelName != None:
                        otherKernel = SymbolTable(
                            self.module.operation)[maybeKernelName]
                        res = processFunctionCall(otherKernel)
                        if res is not None:
                            self.pushValue(res)
                        return

                # Handle debug functions
                if devKey == 'cudaq.dbg.ast':
                    # Handle a debug print statement
                    arg = self.__groupValues(node.args, [1])
                    self.__insertDbgStmt(arg, node.func.attr)
                    return

                # Handle kernels defined in other modules
                symName = processDecorator(node.func.attr, path=devKey)
                if symName:
                    node.func = ast.Name(symName)

        if isinstance(node.func, ast.Name):
            symName = (node.func.id if node.func.id in self.symbolTable else
                       processDecorator(node.func.id))
            if symName:
                result = processDecoratorCall(symName)
                if result:
                    self.pushValue(result)
                return

            if node.func.id == 'complex':

                keywords = [kw.arg for kw in node.keywords]
                kwreal = 'real' in keywords
                kwimag = 'imag' in keywords
                real, imag = self.__groupValues(node.args,
                                                [not kwreal, not kwimag])
                for keyword in node.keywords:
                    self.visit(keyword.value)
                    kwval = self.popValue()
                    if keyword.arg == 'real':
                        real = kwval
                    elif keyword.arg == 'imag':
                        imag = kwval
                    else:
                        self.emitFatalError(f"unknown keyword `{keyword.arg}`",
                                            node)
                if not real or not imag:
                    self.emitFatalError("missing value", node)

                imag = self.changeOperandToType(self.getFloatType(), imag)
                real = self.changeOperandToType(self.getFloatType(), real)
                self.pushValue(
                    complex.CreateOp(self.getComplexType(), real, imag).result)
                return

            if node.func.id == 'len':
                listVal = self.__groupValues(node.args, [1])

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
                totalSize = arith.MaxSIOp(
                    zero,
                    arith.DivSIOp(totalSize, stepVal).result).result

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

                vect = cc.StdvecInitOp(cc.StdvecType.get(iTy),
                                       iterable,
                                       length=totalSize).result
                self.pushValue(vect)
                return

            if node.func.id == 'enumerate':
                # We have to have something "iterable" on the stack,
                # could be coming from `range()` or an iterable like `qvector`
                iterable = self.__groupValues(node.args, [1])

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
                enumIterable = cc.AllocaOp(cc.PointerType.get(
                    cc.ArrayType.get(structTy)),
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
                vect = cc.StdvecInitOp(cc.StdvecType.get(structTy),
                                       enumIterable,
                                       length=totalSize).result
                self.pushValue(vect)
                return

            if self.__isSimpleGate(node.func.id):
                processQuakeCtor(node.func.id.title(),
                                 node.args,
                                 isCtrl=False,
                                 isAdj=False)
                return

            if self.__isAdjointSimpleGate(node.func.id):
                processQuakeCtor(node.func.id[0].title(),
                                 node.args,
                                 isCtrl=False,
                                 isAdj=True)
                return

            if self.__isControlledSimpleGate(node.func.id):
                processQuakeCtor(node.func.id[1:].title(),
                                 node.args,
                                 isCtrl=True,
                                 isAdj=False)
                return

            if self.__isRotationGate(node.func.id):
                processQuakeCtor(node.func.id.title(),
                                 node.args,
                                 isCtrl=False,
                                 isAdj=False,
                                 numParams=1)
                return

            if self.__isControlledRotationGate(node.func.id):
                processQuakeCtor(node.func.id[1:].title(),
                                 node.args,
                                 isCtrl=True,
                                 isAdj=False,
                                 numParams=1)
                return

            if self.__isMeasurementGate(node.func.id):
                registerName = self.currentAssignVariableName
                # If `registerName` is None, then we know that we are not
                # assigning this measure result to anything so we therefore
                # should not push it on the stack
                pushResultToStack = registerName != None or self.walkingReturnNode

                # By default we set the `register_name` for the measurement to
                # the assigned variable name (if there is one). But the use
                # could have manually specified `register_name='something'`
                # check for that here and use it there
                if len(node.keywords) == 1 and hasattr(node.keywords[0], 'arg'):
                    if node.keywords[0].arg == 'register_name':
                        userProvidedRegName = node.keywords[0]
                        if not isinstance(userProvidedRegName.value,
                                          ast.Constant):
                            self.emitFatalError(
                                "measurement register_name keyword must be a "
                                "constant string literal.", node)
                        self.debug_msg(lambda: f'[(Inline) Visit Constant]',
                                       userProvidedRegName.value)
                        registerName = userProvidedRegName.value.value

                qubits = self.__groupValues(node.args, [(1, -1)])
                label = registerName or None
                if len(qubits) == 1 and quake.RefType.isinstance(
                        qubits[0].type):
                    measTy = quake.MeasureType.get()
                    resTy = self.getIntegerType(1)
                else:
                    measTy = cc.StdvecType.get(quake.MeasureType.get())
                    resTy = cc.StdvecType.get(self.getIntegerType(1))
                measureResult = processQuantumOperation(
                    node.func.id.title(), [],
                    qubits,
                    measTy,
                    broadcast=False,
                    registerName=label).result

                # FIXME: needs to be revised when we properly distinguish
                # measurement types
                if pushResultToStack:
                    self.pushValue(
                        quake.DiscriminateOp(resTy, measureResult).result)
                return

            if node.func.id == 'swap':
                processQuakeCtor(node.func.id.title(),
                                 node.args,
                                 isCtrl=False,
                                 isAdj=False,
                                 numTargets=2)
                return

            if node.func.id == 'reset':
                targets = self.__groupValues(node.args, [(1, -1)])
                processQuantumOperation(node.func.id.title(), [],
                                        targets,
                                        broadcast=lambda q: q)
                return

            if node.func.id == 'u3':
                processQuakeCtor(node.func.id.title(),
                                 node.args,
                                 isCtrl=False,
                                 isAdj=False,
                                 numParams=3)
                return

            if node.func.id == 'exp_pauli':
                # Note: C++ also has a constructor that takes an `f64`,
                # `string`, any any number of qubits. We don't support this
                # here.
                theta, target, pauliWord = self.__groupValues(
                    node.args, [1, 1, 1])
                theta = self.changeOperandToType(self.getFloatType(), theta)
                processQuantumOperation("ExpPauli", [], [target], [], [theta],
                                        broadcast=False,
                                        pauli=pauliWord)
                return

            if node.func.id in globalRegisteredOperations:
                unitary = globalRegisteredOperations[node.func.id]
                numTargets = int(np.log2(np.sqrt(unitary.size)))
                targets = self.__groupValues(node.args,
                                             [(numTargets, numTargets)])

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

            elif node.func.id == 'int':
                # cast operation
                value = self.__groupValues(node.args, [1])
                casted = self.changeOperandToType(IntegerType.get_signless(64),
                                                  value,
                                                  allowDemotion=True)
                self.pushValue(casted)
                return

            elif node.func.id == 'list':
                # The expected Python behavior is that a constructor call
                # to list creates a new list (a shallow copy).
                value = self.__groupValues(node.args, [1])
                copy = self.__copyAndValidateContainer(value, node.args[0],
                                                       False)
                self.pushValue(copy)
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
                        "Adding new fields in data classes is not yet "
                        "supported. The dataclass must be declared with "
                        "@dataclass(slots=True) or @dataclasses.dataclass"
                        "(slots=True).", node)

                if node.keywords:
                    self.emitFatalError(
                        "keyword arguments for data classes are not yet "
                        "supported", node)

                structTys = [
                    mlirTypeFromPyType(v, self.ctx)
                    for _, v in annotations.items()
                ]

                numArgs = len(structTys)
                ctorArgs = self.__groupValues(node.args, [(numArgs, numArgs)])
                ctorArgs = convertArguments(structTys, ctorArgs)
                for idx, arg in enumerate(ctorArgs):
                    self.__validate_container_entry(arg, node.args[idx])

                structTy = mlirTryCreateStructType(structTys,
                                                   name=node.func.id,
                                                   context=self.ctx)
                if structTy is None:
                    self.emitFatalError(
                        "Hybrid quantum-classical data types and nested "
                        "quantum structs are not allowed.", node)

                # Disallow user specified methods on structs
                if len({
                        k: v
                        for k, v in cls.__dict__.items()
                        if not (k.startswith('__') and k.endswith('__')) and
                        isinstance(v, types.FunctionType)
                }) != 0:
                    self.emitFatalError(
                        'struct types with user specified methods are not allowed.',
                        node)

                if quake.StruqType.isinstance(structTy):
                    # If we have a quantum struct. We cannot allocate classical
                    # memory and load / store quantum type values to that memory
                    # space, so use `quake.MakeStruqOp`.
                    self.pushValue(quake.MakeStruqOp(structTy, ctorArgs).result)
                    return

                struct = cc.UndefOp(structTy)
                for idx, element in enumerate(ctorArgs):
                    struct = cc.InsertValueOp(
                        structTy, struct, element,
                        DenseI64ArrayAttr.get([idx], context=self.ctx)).result
                self.pushValue(struct)
                return

            else:
                self.emitFatalError(f"unhandled function call - {node.func.id}",
                                    node)

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
                funcVal = self.popValue()
                deepCopy, dTy = None, None

                for keyword in node.keywords:
                    if keyword.arg == 'deep':
                        deepCopy = keyword.value
                    elif keyword.arg == 'dtype':
                        self.visit(keyword.value)
                        dTy = self.popValue()
                    else:
                        self.emitFatalError(f"unknown keyword `{keyword.arg}`",
                                            node)

                if len(node.args) == 1 and deepCopy is None:
                    deepCopy = node.args[0]
                else:
                    self.__groupValues(node.args, [0])
                if deepCopy:
                    if not isinstance(deepCopy, ast.Constant):
                        self.emitFatalError(
                            "argument to `copy` must be a constant", node)
                    deepCopy = deepCopy.value

                # If we created a deep copy, we can set the parent node
                # of the value to copy to be this node for validation purposes.
                pyVal = node if deepCopy else node.func.value
                copy = self.__copyAndValidateContainer(funcVal, pyVal, deepCopy,
                                                       dTy)
                self.pushValue(copy)
                return

            if self.__isSupportedVectorFunction(node.func.attr):
                # This means we are visiting this node twice - once in
                # visit_Attribute, once here. But unless we make the functions
                # we support on values explicit somewhere, there is no way
                # around that.
                self.visit(node.func.value)
                funcVal = self.popValue()

                # Just to be nice and give a dedicated error.
                if (node.func.attr == 'append' and
                    (quake.VeqType.isinstance(funcVal.type) or
                     cc.StdvecType.isinstance(funcVal.type))):
                    self.emitFatalError(
                        "CUDA-Q does not allow dynamic resizing or lists, "
                        "arrays, or qvectors.", node)

                # Neither Python lists nor `numpy` arrays have a function
                # or attribute 'front'/'back'; hence we only support that
                # for `qvectors`.
                if not quake.VeqType.isinstance(funcVal.type):
                    self.emitFatalError(
                        f'function {node.func.attr} is not supported '
                        f'on a value of type {funcVal.type}', node)

                funcArg = None
                args = self.__groupValues(node.args, [(0, 1)])
                if args:
                    funcArg = args[0]
                    if not IntegerType.isinstance(funcArg.type):
                        self.emitFatalError(
                            f'expecting an integer argument for call '
                            f'to {node.func.attr}', node)

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

                    if node.func.attr == 'array':
                        # The expected Python behavior is that a constructor
                        # call to array creates a new array (a shallow copy).
                        # Additionally, since a new value is created, we need to
                        # make sure container entries are properly validated. To
                        # not duplicate the logic, we simply call `copy` here.
                        self.visit_Call(
                            ast.Call(ast.Attribute(node.args[0], 'copy'), [],
                                     node.keywords))
                        return

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

                    # Promote argument's types for `numpy.func` calls to match
                    # python's semantics
                    if self.__isSupportedNumpyFunction(node.func.attr):
                        if ComplexType.isinstance(value.type):
                            value = self.changeOperandToType(
                                self.getComplexType(), value)
                        elif IntegerType.isinstance(value.type):
                            value = self.changeOperandToType(
                                self.getFloatType(), value)
                        elif (not F64Type.isinstance(value.type) and
                              not F32Type.isinstance(value.type)):
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
                            # Note: using `complex.ExpOp` results in a "can't
                            # legalize `complex.exp`" error. Using Euler's'
                            # formula instead:
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
                                f"numpy call ({node.func.attr}) is not "
                                f"supported for complex numbers", node)
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
                            valueTy = cc.PointerType.getElementType(value.type)
                        if cc.StdvecType.isinstance(valueTy):
                            self.pushValue(value)
                            return

                        self.emitFatalError(
                            f"unsupported amplitudes argument type: "
                            f"{value.type}", node)

                    if node.func.attr == 'qvector':
                        if len(node.args) == 0:
                            self.emitFatalError(
                                'qvector does not have default constructor. '
                                'Init from size or existing state.', node)

                        value = self.__groupValues(node.args, [1])

                        if (IntegerType.isinstance(value.type)):
                            # handle `cudaq.qvector(n)`
                            ty = self.getVeqType()
                            qubits = quake.AllocaOp(ty, size=value).result
                            self.pushValue(qubits)
                            return

                        if cc.StdvecType.isinstance(value.type):

                            # handle `cudaq.qvector(initState)`
                            def check_vector_init():
                                """
                                Run semantics checks. Validate the length in
                                case of a constant initializer:
                                  `cudaq.qvector([1., 0., ...])`
                                  `cudaq.qvector(np.array([1., 0., ...]))`
                                """
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
                                            "Invalid input state size for "
                                            "qvector init (not a power of 2)",
                                            node)

                            check_vector_init()
                            eleTy = cc.StdvecType.getElementType(value.type)
                            arrTy = cc.ArrayType.get(eleTy)
                            ptrArrTy = cc.PointerType.get(arrTy)
                            data = cc.StdvecDataOp(ptrArrTy, value).result
                            size = cc.StdvecSizeOp(self.getIntegerType(),
                                                   value).result

                            # Dynamic checking that the state is normalized is
                            # done at the library layer.
                            veqTy = quake.VeqType.get()
                            stateTy = cc.PointerType.get(cc.StateType.get())
                            statePtr = quake.CreateStateOp(stateTy, data, size)
                            numQubits = quake.GetNumberOfQubitsOp(
                                size.type, statePtr).result
                            qubits = quake.AllocaOp(veqTy,
                                                    size=numQubits).result
                            init = quake.InitializeStateOp(
                                veqTy, qubits, statePtr).result
                            quake.DeleteStateOp(statePtr)
                            self.pushValue(init)
                            return

                        if (cc.PointerType.isinstance(value.type) and
                                cc.StateType.isinstance(
                                    cc.PointerType.getElementType(value.type))):
                            # handle `cudaq.qvector(state)`

                            i64Ty = self.getIntegerType()
                            numQubits = quake.GetNumberOfQubitsOp(i64Ty,
                                                                  value).result

                            veqTy = quake.VeqType.get()
                            qubits = quake.AllocaOp(veqTy,
                                                    size=numQubits).result
                            init = quake.InitializeStateOp(
                                veqTy, qubits, value).result

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

                        # NOTE: We currently generally don't have the means in
                        # the compiler to handle composition of control and
                        # adjoint, since control and adjoint are not proper
                        # functors (i.e. there is no way to obtain a new
                        # callable object that is the adjoint or controlled
                        # version of another callable).  Since we don't really
                        # treat callables as first-class values, the first
                        # argument to control and adjoint indeed has to be a
                        # Name object.

                        # FIXME: WE SHOULD NOW BE ABLE TO DEAL WITH ADJOINT OF
                        # QUALIFIED NAME

                        if not node.args or not isinstance(
                                node.args[0], ast.Name):
                            self.emitFatalError(
                                f'unsupported argument in call to '
                                f'{node.func.attr} - first argument must be a'
                                f' symbol name', node)
                        otherFuncName = node.args[0].id
                        kwargs = {"is_adj": node.func.attr == 'adjoint'}
                        processDecorator(otherFuncName)

                        if otherFuncName in self.symbolTable:
                            self.visit(node.args[0])
                            fctArg = self.popValue()
                            if not cc.CallableType.isinstance(fctArg.type):
                                self.emitFatalError(
                                    f"{otherFuncName} is not a quantum kernel",
                                    node)
                            functionTy = FunctionType(
                                cc.CallableType.getFunctionType(fctArg.type))
                            inputTys = functionTy.inputs
                            outputTys = functionTy.results
                            indirectCallee = [fctArg]
                        elif otherFuncName in globalRegisteredOperations:
                            self.emitFatalError(
                                "calling cudaq.control or cudaq.adjoint on "
                                "a globally registered operation is not "
                                "supported", node)
                        elif self.__isUnitaryGate(
                                otherFuncName) or self.__isMeasurementGate(
                                    otherFuncName):
                            self.emitFatalError(
                                "calling cudaq.control or cudaq.adjoint on a "
                                "built-in gate is not supported", node)
                        else:
                            self.emitFatalError(
                                f"{otherFuncName} is not a known quantum "
                                f"kernel - maybe a cudaq.kernel attribute is"
                                f" missing?.", node)

                        numArgs = len(inputTys)
                        invert_controls = lambda: None
                        if node.func.attr == 'control':
                            controls, args = self.__groupValues(
                                node.args[1:], [(1, -1), (numArgs, numArgs)])
                            qvec_or_qubits = (
                                all((quake.RefType.isinstance(v.type)
                                     for v in controls)) or
                                (len(controls) == 1 and
                                 quake.VeqType.isinstance(controls[0].type)))
                            if not qvec_or_qubits:
                                self.emitFatalError(
                                    f'invalid argument type for control'
                                    f' operand', node)
                            # TODO: it would be cleaner to add support for
                            # negated control qubits to `quake.ApplyOp`
                            negatedControlQubits = self.controlNegations.copy()
                            self.controlNegations.clear()
                            if negatedControlQubits:
                                invert_controls = lambda: processQuantumOperation(
                                    'X', [], negatedControlQubits, [], [])
                        else:
                            controls, args = self.__groupValues(
                                node.args[1:], [(0, 0), (numArgs, numArgs)])

                        args = convertArguments(inputTys, args)
                        if len(outputTys) != 0:
                            self.emitFatalError(
                                f'cannot take {node.func.attr} of kernel '
                                f'{otherFuncName} that returns a value', node)
                        invert_controls()
                        quake.ApplyOp([], indirectCallee, controls, args,
                                      **kwargs)
                        invert_controls()
                        return

                    if node.func.attr == 'apply_noise':

                        supportedChannels = [
                            'DepolarizationChannel', 'AmplitudeDampingChannel',
                            'PhaseFlipChannel', 'BitFlipChannel',
                            'PhaseDamping', 'ZError', 'XError', 'YError',
                            'Pauli1', 'Pauli2', 'Depolarization1',
                            'Depolarization2'
                        ]

                        # The first argument must be the Kraus channel
                        numParams, key = 0, None
                        if (isinstance(node.args[0], ast.Attribute) and
                                node.args[0].value.id == 'cudaq' and
                                node.args[0].attr in supportedChannels):

                            cudaq_module = importlib.import_module('cudaq')
                            channel_class = getattr(cudaq_module,
                                                    node.args[0].attr)
                            numParams = channel_class.num_parameters
                            key = self.getConstantInt(hash(channel_class))
                        elif isinstance(node.args[0], ast.Name):
                            arg = recover_value_of_or_none(
                                node.args[0].id, None)
                            if (arg and isinstance(arg, type) and issubclass(
                                    arg, cudaq_runtime.KrausChannel)):
                                if not hasattr(arg, 'num_parameters'):
                                    self.emitFatalError(
                                        'apply_noise kraus channels must have '
                                        '`num_parameters` constant class '
                                        'attribute specified.')
                                numParams = arg.num_parameters
                                key = self.getConstantInt(hash(arg))
                        if key is None:
                            self.emitFatalError(
                                "unsupported argument for Kraus channel in "
                                "apply_noise", node)

                        # This currently requires at least one qubit argument
                        params, values = self.__groupValues(
                            node.args[1:], [(numParams, numParams), (1, -1)])
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

                    if node.func.attr == 'to_integer':
                        boolVec = self.__groupValues(node.args, [1])
                        args = convertArguments(
                            [cc.StdvecType.get(self.getIntegerType(1))],
                            [boolVec])
                        cudaqConvertToInteger = "__nvqpp_cudaqConvertToInteger"
                        load_intrinsic(self.module, cudaqConvertToInteger)
                        result = func.CallOp([self.getIntegerType(64)],
                                             cudaqConvertToInteger, args).result
                        self.pushValue(result)
                        return

                    self.emitFatalError(
                        f'Invalid function or class type requested from '
                        f'the cudaq module ({node.func.attr})', node)

                def maybeProposeOpAttrFix(opName, attrName):
                    """
                    Check the quantum operation attribute name and propose a
                    smart fix message if possible. For example, if we have
                    `x.control(...)` then remind the programmer the correct
                    attribute is `x.ctrl(...)`.
                    """
                    # TODO Add more possibilities in the future...
                    if (attrName == 'control') or ('control' in attrName) or (
                            'ctrl' in attrName):
                        return f'Did you mean {opName}.ctrl(...)?'

                    if (attrName == 'adjoint') or ('adjoint' in attrName) or (
                            'adj' in attrName):
                        return f'Did you mean {opName}.adj(...)?'

                    return ''

                # We have a `func_name.ctrl`
                if self.__isSimpleGate(node.func.value.id):
                    if node.func.attr == 'ctrl':
                        processQuakeCtor(node.func.value.id.title(),
                                         node.args,
                                         isCtrl=True,
                                         isAdj=False)
                        return
                    if node.func.attr == 'adj':
                        processQuakeCtor(node.func.value.id.title(),
                                         node.args,
                                         isCtrl=False,
                                         isAdj=True)
                        return
                    self.emitFatalError(
                        f'Unknown attribute on quantum operation '
                        f'{node.func.value.id} ({node.func.attr}). '
                        f'{maybeProposeOpAttrFix(node.func.value.id, node.func.attr)}'
                    )

                if self.__isRotationGate(node.func.value.id):
                    if node.func.attr == 'ctrl':
                        processQuakeCtor(node.func.value.id.title(),
                                         node.args,
                                         isCtrl=True,
                                         isAdj=False,
                                         numParams=1)
                        return
                    if node.func.attr == 'adj':
                        processQuakeCtor(node.func.value.id.title(),
                                         node.args,
                                         isCtrl=False,
                                         isAdj=True,
                                         numParams=1)
                        return
                    self.emitFatalError(
                        f'Unknown attribute on quantum operation '
                        f'{node.func.value.id} ({node.func.attr}). '
                        f'{maybeProposeOpAttrFix(node.func.value.id, node.func.attr)}'
                    )

                if node.func.value.id == 'swap' and node.func.attr == 'ctrl':
                    processQuakeCtor(node.func.value.id.title(),
                                     node.args,
                                     isCtrl=True,
                                     isAdj=False,
                                     numTargets=2)
                    return

                if node.func.value.id == 'u3':
                    if node.func.attr == 'ctrl':
                        processQuakeCtor(node.func.value.id.title(),
                                         node.args,
                                         isCtrl=True,
                                         isAdj=False,
                                         numParams=3)
                        return
                    if node.func.attr == 'adj':
                        processQuakeCtor(node.func.value.id.title(),
                                         node.args,
                                         isCtrl=False,
                                         isAdj=True,
                                         numParams=3)
                        return
                    self.emitFatalError(
                        f'unknown attribute {node.func.attr} on u3', node)

                # custom `ctrl` and `adj`
                if node.func.value.id in globalRegisteredOperations:
                    if not node.func.attr == 'ctrl' and not node.func.attr == 'adj':
                        self.emitFatalError(
                            f'Unknown attribute on custom operation '
                            f'{node.func.value.id} ({node.func.attr}).')

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
                        controls, targets = self.__groupValues(
                            node.args, [(1, -1), (numTargets, numTargets)])
                        negatedControlQubits = getNegatedControlQubits(controls)
                        is_adj = False
                    if node.func.attr == 'adj':
                        controls, targets = self.__groupValues(
                            node.args, [0, (numTargets, numTargets)])
                        negatedControlQubits = None
                        is_adj = True

                    checkControlAndTargetTypes(controls, targets)
                    # The check above makes sure targets are either a list
                    # of individual qubits, or a single `qvector`. Since
                    # a `qvector` is not allowed, we check this here:
                    if not quake.RefType.isinstance(targets[0].type):
                        self.emitFatalError(
                            'invalid target operand - target must not be '
                            'a qvector')

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

        self.visit(node.generators[0].iter)
        iterable = self.popValue()
        if cc.StdvecType.isinstance(iterable.type):
            iterableSize = cc.StdvecSizeOp(self.getIntegerType(),
                                           iterable).result
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
            # NOTE: This does not actually create a valid value, and will fail
            # if something tries to use the value that this was supposed to
            # create later on. Keeping this to keep existing functionality, but
            # this is a bit questionable. Aside from no list being produced,
            # this should work regardless of what we iterate over or what
            # expression we evaluate.
            self.emitWarning(
                "produced elements in list comprehension contain None - "
                "expression will be evaluated but no list is generated", node)
            forNode = ast.For()
            forNode.iter = node.generators[0].iter
            forNode.target = node.generators[0].target
            forNode.body = [node.elt]
            forNode.orelse = []
            forNode.lineno = node.lineno
            # This loop could be marked as invariant if we didn't use
            # `visit_For`, but that would be premature optimization.
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
        # Unfortunately, dynamic typing makes this a bit painful. I didn't find
        # a good way to fill in the type only once we have processed the
        # expression as part of the loop body, but it would probably be nicer
        # and cleaner to do that instead.
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
                structTy = mlirTryCreateStructType(elts, context=self.ctx)
                if not structTy:
                    # we return anything here since, or rather to make sure
                    # that, a comprehensive error is generated when `elt` is
                    # walked below.
                    return cc.StructType.getNamed("tuple", elts)
                return structTy
            elif (isinstance(pyval, ast.Subscript) and
                  IntegerType.isinstance(get_item_type(pyval.slice))):
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
                                f"non-homogenous list not allowed - must all be "
                                f"same type: {elts}", node)
                return cc.StdvecType.get(base_elTy)
            elif isinstance(pyval, ast.Call):
                if isinstance(pyval.func, ast.Name):
                    # supported for calls but not here: 'range', 'enumerate'
                    decorator = recover_kernel_decorator(pyval.func.id)
                    if decorator:
                        # Not necessarily unitary
                        resTy = decorator.handle_call_results()
                        if resTy == decorator.get_none_type():
                            process_void_list()
                            return None
                        return resTy
                    if self.__isUnitaryGate(
                            pyval.func.id) or pyval.func.id == 'reset':
                        process_void_list()
                        return None
                    if self.__isMeasurementGate(pyval.func.id):
                        # It's tricky to know if we are calling a measurement on
                        # a single qubit, or on a vector of qubits, e.g.
                        # consider the case `[mz(qs[i:]) for i in range(n)]`, or
                        # `[mz(qs) for _ in range(n)], or [mz(qs) for _ in qs]`.
                        # We hence limit support to iterating over a vector of
                        # qubits, and check that the iteration variable is
                        # passed directly to the measurement.
                        iterSymName = None
                        if isinstance(node.generators[0].iter, ast.Name):
                            iterSymName = node.generators[0].iter.id
                        elif (isinstance(node.generators[0].iter, ast.Subscript)
                              and isinstance(node.generators[0].iter.slice,
                                             ast.Slice) and
                              isinstance(node.generators[0].iter.value,
                                         ast.Name)):
                            iterSymName = node.generators[0].iter.value.id
                        isIterOverVeq = (
                            iterSymName is not None and
                            iterSymName in self.symbolTable and
                            quake.VeqType.isinstance(
                                self.symbolTable[iterSymName].type))
                        if not isIterOverVeq:
                            self.emitFatalError(
                                "performing measurements in list comprehension "
                                "expressions is only supported when iterating "
                                "over a vector of qubits", node)
                        iterVarPassedAsArg = (
                            len(pyval.args) == 1 and
                            isinstance(pyval.args[0], ast.Name) and
                            isinstance(node.generators[0].target, ast.Name) and
                            pyval.args[0].id == node.generators[0].target.id)
                        if not iterVarPassedAsArg:
                            self.emitFatalError(
                                "unsupported argument to measurement in list "
                                "comprehension", node)
                        return IntegerType.get_signless(1)
                    if pyval.func.id in globalRegisteredTypes.classes:
                        _, annotations = globalRegisteredTypes.getClassAttributes(
                            pyval.func.id)
                        elts = [
                            mlirTypeFromPyType(v, self.ctx)
                            for _, v in annotations.items()
                        ]
                        structTy = mlirTryCreateStructType(elts,
                                                           pyval.func.id,
                                                           context=self.ctx)
                        if not structTy:
                            # we return anything here since, or rather to make
                            # sure that, a comprehensive error is generated
                            # when `elt` is walked below.
                            return cc.StructType.getNamed(pyval.func.id, elts)
                        return structTy
                    elif pyval.func.id == 'len' or pyval.func.id == 'int':
                        return IntegerType.get_signless(64)
                    elif pyval.func.id == 'complex':
                        return self.getComplexType()
                    elif pyval.func.id == 'list' and len(pyval.args) == 1:
                        return get_item_type(pyval.args[0])
                elif isinstance(pyval.func, ast.Attribute):
                    if (pyval.func.attr == 'copy' and
                            'dtype' not in pyval.keywords):
                        return get_item_type(pyval.func.value)
                    if pyval.func.attr == 'ctrl' or pyval.func.attr == 'adj':
                        process_void_list()
                        return None
                self.emitFatalError("unsupported call in list comprehension",
                                    node)
            elif isinstance(pyval, ast.Compare):
                return IntegerType.get_signless(1)
            elif (isinstance(pyval, ast.UnaryOp) and
                  isinstance(pyval.op, ast.Not)):
                return IntegerType.get_signless(1)
            elif isinstance(pyval, ast.BinOp):
                # division and power are special, everything else
                # strictly creates a value of superior type
                if isinstance(pyval.op, ast.Pow):
                    # determining the correct type is messy, left as TODO for
                    # now...
                    self.emitFatalError(
                        "BinOp.Pow is not currently supported in list "
                        "comprehension expressions", node)
                leftTy = get_item_type(pyval.left)
                rightTy = get_item_type(pyval.right)
                superiorTy = self.__get_superior_type(leftTy, rightTy)
                # division converts integer type to `FP64` and preserves the
                # superior type otherwise
                if isinstance(pyval.op,
                              ast.Div) and IntegerType.isinstance(superiorTy):
                    return F64Type.get()
                return superiorTy
            else:
                self.emitFatalError(
                    "Only variables, constants, and some calls can be used to "
                    "populate values in list comprehension expressions", node)

        listElemTy = get_item_type(node.elt)
        if listElemTy is None:
            return

        resultVecTy = cc.StdvecType.get(listElemTy)
        if listElemTy == self.getIntegerType(1):
            listElemTy = self.getIntegerType(8)
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
                iterVal = quake.ExtractRefOp(iterTy,
                                             iterable,
                                             -1,
                                             index=iterVar).result
            else:
                eleAddr = cc.ComputePtrOp(
                    cc.PointerType.get(iterTy), iterable, [iterVar],
                    DenseI32ArrayAttr.get([kDynamicPtrIndex], context=self.ctx))
                iterVal = cc.LoadOp(eleAddr).result

            # We don't do support anything within list comprehensions that would
            # require being careful about assigning references, so simply
            # adding them to the symbol table is enough for list comprehension.
            self.__deconstructAssignment(node.generators[0].target, iterVal)
            self.visit(node.elt)
            element = self.popValue()
            # We do need to be careful, however, about validating the list
            # elements.
            self.__validate_container_entry(element, node.elt)

            listValueAddr = cc.ComputePtrOp(
                cc.PointerType.get(listElemTy), listValue, [iterVar],
                DenseI32ArrayAttr.get([kDynamicPtrIndex], context=self.ctx))
            element = self.changeOperandToType(listElemTy,
                                               element,
                                               allowDemotion=False)
            cc.StoreOp(element, listValueAddr)
            self.symbolTable.popScope()

        self.createInvariantForLoop(bodyBuilder, iterableSize)
        res = cc.StdvecInitOp(resultVecTy, listValue,
                              length=iterableSize).result
        self.pushValue(res)
        return

    def visit_List(self, node):
        """
        This method will visit the `ast.List` node and represent lists of
        quantum typed values as a concatenated `quake.ConcatOp` producing a
        single `veq` instances.
        """

        # Prevent the creation of empty lists, since we don't support inferring
        # their types. To do so, we would need to look forward to the first use
        # and determine the type based on that.
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
                evalElem = self.popValue()
                if self.isQuantumType(
                        evalElem.type) and not quake.RefType.isinstance(
                            evalElem.type):
                    self.emitFatalError(
                        "list must not contain a qvector or quantum"
                        " struct - use `*` operator to unpack qvectors", node)
                self.__validate_container_entry(evalElem, element)
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
        self.pushValue(self.__createStdvecWithKnownValues(listElementValues))

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
        Convert element extractions (`__getitem__`, `operator[](idx)`, `q[1:3]`)
        to corresponding extraction or slice code in the MLIR. This method
        handles extraction for `veq` types and `stdvec` types.
        """

        def get_size(val):
            if quake.VeqType.isinstance(val.type):
                return quake.VeqSizeOp(self.getIntegerType(), val).result
            elif cc.StdvecType.isinstance(val.type):
                return cc.StdvecSizeOp(self.getIntegerType(), val).result
            return None

        def fix_negative_idx(idx, get_size):
            if (IntegerType.isinstance(idx.type) and
                    hasattr(idx.owner, 'opview') and
                    isinstance(idx.owner.opview, arith.ConstantOp) and
                    'value' in idx.owner.attributes):
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
            if self.pushPointerValue:
                self.emitFatalError(
                    "slicing a list or qvector does not produce a "
                    "modifiable value", node)

            self.visit(node.value)
            var = self.popValue()
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
                        f"unhandled upper slice == None, can't handle "
                        f"type {var.type}", node)
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
                # Use `i8` for boolean elements
                if eleTy == self.getIntegerType(1):
                    eleTy = self.getIntegerType(8)
                ptrTy = cc.PointerType.get(eleTy)
                arrTy = cc.ArrayType.get(eleTy)
                ptrArrTy = cc.PointerType.get(arrTy)
                nElementsVal = arith.SubIOp(upperVal, lowerVal).result

                # need to compute the distance between `upperVal` and `lowerVal`
                # then slice is `stdvecdataOp + computeptr[lower] +`
                # `stdvecinit[ptr,distance]`

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

        # Only variable names, subscripts and attributes can produce modifiable
        # values. Anything else produces an immutable value. We make sure the
        # visit gets processed such that the rest of the code can give a proper
        # error.

        value_root = node.value
        while (isinstance(value_root, ast.Subscript) or
               isinstance(value_root, ast.Attribute)):
            value_root = value_root.value
        if self.pushPointerValue and not isinstance(value_root, ast.Name):
            self.pushPointerValue = False
            self.visit(node.value)
            var = self.popValue()
            self.pushPointerValue = True
        else:
            # `isSubscriptRoot` is only used/needed to enable modification of
            # items in lists and `dataclasses` contained in a tuple
            subscriptRoot = self.isSubscriptRoot
            self.isSubscriptRoot = True
            self.visit(node.value)
            var = self.popValue()
            self.isSubscriptRoot = subscriptRoot

        pushPtr = self.pushPointerValue
        self.pushPointerValue = False
        self.visit(node.slice)
        idx = self.popValue()
        self.pushPointerValue = pushPtr

        if quake.VeqType.isinstance(var.type):
            if self.pushPointerValue:
                self.emitFatalError(
                    "indexing into a qvector does not produce a "
                    "modifyable value", node)

            if not IntegerType.isinstance(idx.type):
                self.emitFatalError(
                    f'invalid index variable type used for '
                    f'qvector extraction ({idx.type})', node)
            idx = fix_negative_idx(idx, lambda: get_size(var))
            self.pushValue(
                quake.ExtractRefOp(self.getRefType(), var, -1,
                                   index=idx).result)
            return

        if cc.PointerType.isinstance(var.type):
            # We should only ever get a pointer if we explicitly asked for it.
            assert self.pushPointerValue
            varType = cc.PointerType.getElementType(var.type)
            if cc.StdvecType.isinstance(varType):
                # We can get a pointer to a vector (only) if we are updating a
                # struct item that is a pointer.
                if self.pushPointerValue:
                    # In this case, it should be save to load the vector, since
                    # the underlying data is not loaded.
                    var = cc.LoadOp(var).result

            if cc.StructType.isinstance(varType):
                structName = cc.StructType.getName(varType)
                if not self.isSubscriptRoot and structName == 'tuple':
                    self.emitFatalError("tuple value cannot be modified", node)
                if not isinstance(node.slice, ast.Constant):
                    if self.pushPointerValue:
                        if structName == 'tuple':
                            self.emitFatalError(
                                "tuple value cannot be modified via "
                                "non-constant subscript", node)
                        self.emitFatalError(
                            f"{structName} value cannot be modified "
                            f"via non-constant subscript - use "
                            f"attribute access instead", node)

                idxVal = node.slice.value
                structTys = cc.StructType.getTypes(varType)
                eleAddr = cc.ComputePtrOp(cc.PointerType.get(structTys[idxVal]),
                                          var, [],
                                          DenseI32ArrayAttr.get([idxVal
                                                                ])).result
                if self.pushPointerValue:
                    self.pushValue(eleAddr)
                    return

        if cc.StdvecType.isinstance(var.type):
            idx = fix_negative_idx(idx, lambda: get_size(var))
            eleTy = cc.StdvecType.getElementType(var.type)
            isBool = eleTy == self.getIntegerType(1)
            if isBool:
                eleTy = self.getIntegerType(8)
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
            val = cc.LoadOp(eleAddr).result
            if isBool:
                val = self.changeOperandToType(self.getIntegerType(1), val)
            self.pushValue(val)
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
                structName = cc.StructType.getName(var.type)
                if structName == 'tuple':
                    self.emitFatalError("tuple value cannot be modified", node)
                self.emitFatalError(
                    f"{structName} value cannot be modified - "
                    f"use `.copy(deep)` to create a new value that "
                    f"can be modified", node)

            memberTys = cc.StructType.getTypes(var.type)
            idxValue = get_idx_value(len(memberTys))

            member = cc.ExtractValueOp(memberTys[idxValue], var, [],
                                       DenseI32ArrayAttr.get([idxValue])).result

            self.pushValue(member)
            return

        # We allow subscripts into `Struqs`, but only if we don't need a pointer
        # (i.e. no updating of `Struqs`).
        if quake.StruqType.isinstance(var.type):
            if self.pushPointerValue:
                self.emitFatalError(
                    "indexing into quantum tuple or dataclass "
                    "does not produce a modifiable value", node)

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

            # We can simplify `for i in range(N)` MLIR code immensely by just
            # building a for loop with N as the upper value, no need to generate
            # an array from the `range` call.
            if node.iter.func.id == 'range':
                iterable = None
                startVal, endVal, stepVal, isDecrementing = self.__processRangeLoopIterationBounds(
                    node.iter.args)
                getValues = lambda iterVar: iterVar

            # We can simplify `for i,j in enumerate(L)` MLIR code immensely by
            # just building a for loop over the iterable object L and using the
            # index into that iterable and the element.
            elif node.iter.func.id == 'enumerate':
                if len(node.iter.args) != 1:
                    self.emitFatalError(
                        "invalid number of arguments to enumerate "
                        "- expecting 1 argument", node)

                self.visit(node.iter.args[0])
                iterable = self.popValue()
                getValues = lambda iterVar, v: (iterVar, v)

        if not getValues:
            self.visit(node.iter)
            iterable = self.popValue()

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
                isBool = iterEleTy == self.getIntegerType(1)
                if isBool:
                    iterEleTy = self.getIntegerType(8)
                endVal = cc.StdvecSizeOp(self.getIntegerType(), iterable).result

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
                    if isBool:
                        val = self.changeOperandToType(self.getIntegerType(1),
                                                       val)
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

        self.createMonotonicForLoop(
            lambda iterVar: blockBuilder(iterVar, node.body),
            startVal=startVal,
            stepVal=stepVal,
            endVal=endVal,
            isDecrementing=isDecrementing,
            orElseBuilder=None if not node.orelse else
            lambda iterVar: blockBuilder(iterVar, node.orelse))

    def visit_While(self, node):
        """
        Convert Python while statements into the equivalent CC `LoopOp`.
        """

        def evalCond(args):
            # Not a bug. MLIR printing requires IR to be in a coherent state.
            # When building the cc `LoopOp` regions, the IR is not coherent.
            # The IR verifier will fail.

            v = self.verbose
            self.verbose = False
            self.visit(node.test)
            condition = self.__arithmetic_to_bool(self.popValue())
            self.verbose = v
            return condition

        def blockBuilder(iterVar):
            self.symbolTable.pushScope()
            [self.visit(b) for b in node.body]
            self.symbolTable.popScope()

        self.createForLoop([], blockBuilder, [], evalCond, lambda _: [],
                           None if not node.orelse else
                           lambda _: [self.visit(stmt) for stmt in node.orelse])

    def visit_BoolOp(self, node):
        """
        Convert boolean operations into equivalent MLIR operations using the
        Arith Dialect.
        """
        if isinstance(node.op, ast.And) or isinstance(node.op, ast.Or):

            # Visit the LHS and pop the value Note we want any `mz(q)` calls to
            # push their result value to the stack, so we set a non-None
            # variable name here.
            self.currentAssignVariableName = ''
            self.visit(node.values[0])
            cond = self.__arithmetic_to_bool(self.popValue())

            def process_boolean_op(prior, values):

                if len(values) == 0:
                    return prior

                if isinstance(node.op, ast.And):
                    prior = arith.XOrIOp(prior, self.getConstantInt(1,
                                                                    1)).result

                ifOp = cc.IfOp([prior.type], prior, [])
                thenBlock = Block.create_at_start(ifOp.thenRegion, [])
                with InsertionPoint(thenBlock):
                    if isinstance(node.op, ast.And):
                        constantFalse = arith.ConstantOp(
                            prior.type, BoolAttr.get(False))
                        cc.ContinueOp([constantFalse])
                    else:
                        cc.ContinueOp([prior])

                elseBlock = Block.create_at_start(ifOp.elseRegion, [])
                with InsertionPoint(elseBlock):
                    self.symbolTable.pushScope()
                    self.pushIfStmtBlockStack()
                    self.visit(values[0])
                    rhs = process_boolean_op(
                        self.__arithmetic_to_bool(self.popValue()), values[1:])
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

        # To understand the integer attributes used here (the predicates) see
        # `arith::CmpIPredicate` and `arith::CmpFPredicate`.

        def compare_equality(item1, item2):

            # TODO: the In/NotIn case should be recursive such that we can
            # search for a list in a list of lists.
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
                notListEnd = arith.CmpIOp(IntegerAttr.get(iTy, 2), args[0],
                                          vectSize).result
                notFound = cc.LoadOp(accumulator).result
                return arith.AndIOp(notListEnd, notFound).result

            # Break early if we found the item
            self.createForLoop(
                [self.getIntegerType()], check_element,
                [self.getConstantInt(0)], check_condition, lambda args:
                [arith.AddIOp(args[0], self.getConstantInt(1)).result])

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
        condition = self.__arithmetic_to_bool(self.popValue())

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

        result = self.changeOperandToType(self.knownResultType,
                                          self.popValue(),
                                          allowDemotion=True)

        # Generally, anything that was allocated locally on the stack needs to
        # be copied to the heap to ensure it lives past the the function. This
        # holds recursively; if we have a struct that contains a list, then the
        # list data may need to be copied if it was allocated inside the
        # function.
        def copy_list_to_heap(value):
            symName = '__nvqpp_vectorCopyCtor'
            load_intrinsic(self.module, symName)
            elemTy = cc.StdvecType.getElementType(value.type)
            if elemTy == self.getIntegerType(1):
                elemTy = self.getIntegerType(8)
            ptrTy = cc.PointerType.get(self.getIntegerType(8))
            arrTy = cc.ArrayType.get(self.getIntegerType(8))
            ptrArrTy = cc.PointerType.get(arrTy)
            resBuf = cc.StdvecDataOp(ptrArrTy, value).result
            eleSize = cc.SizeOfOp(self.getIntegerType(),
                                  TypeAttr.get(elemTy)).result
            dynSize = cc.StdvecSizeOp(self.getIntegerType(), value).result
            resBuf = cc.CastOp(ptrTy, resBuf)
            heapCopy = func.CallOp([ptrTy], symName,
                                   [resBuf, dynSize, eleSize]).result
            return cc.StdvecInitOp(value.type, heapCopy, length=dynSize).result

        rootVal = self.__get_root_value(node.value)
        if rootVal and self.isFunctionArgument(rootVal):
            # If we allow assigning a value that contains a list to an item of a
            # function argument (which we do with the exceptions commented
            # below), then we necessarily need to make a copy when we return
            # function arguments, or function argument elements, that contain
            # lists, since we have to assume that their data may be allocated on
            # the stack. However, this leads to incorrect behavior if a returned
            # list was indeed caller-side allocated (and should correspondingly
            # have been returned by reference).  Rather than preventing that
            # lists in function arguments can be updated, we instead ensure that
            # lists contained in function arguments stay recognizable as such,
            # and prevent that function arguments that contain list are
            # returned.  NOTE: Why is seems straightforward in principle to fail
            # only for when we return *inner* lists of function arguments, this
            # is still not a good option for two reasons: 1) Even if we return
            # the reference to the outer list correctly, any caller-side
            # assignment of the return value would no longer be recognizable as
            # being the same reference given as argument, which is a problem if
            # the list was an argument to the caller.  I.e. while this works for
            # one function indirection, it does not work for two (see assignment
            # tests).  2) To ensure that we don't have any memory leaks, we copy
            # any lists returned from function calls to the stack. This copy (as
            # of the time of writing this) results in a segfault when the list
            # is not on the heap. As it is, we hence indeed have to copy every
            # returned list to the heap, followed by a copy to the stack in the
            # caller. Subsequent optimization passes should largely eliminate
            # unnecessary copies.
            if self.containsList(result.type):
                self.emitFatalError(
                    "return value must not contain a list that is a function "
                    "argument or an item in a function argument - for device "
                    "kernels, lists passed as arguments will be modified in "
                    "place; remove the return value or use .copy(deep) to "
                    "create a copy", node)
        else:
            result = self.__migrateLists(result, copy_list_to_heap)

        if self.symbolTable.depth > 1:
            # We are in an inner scope, release all scopes before returning
            cc.UnwindReturnOp([result])
            return
        func.ReturnOp([result])

    def visit_Tuple(self, node):
        """
        Map tuples in the Python AST to equivalents in MLIR.
        """

        [self.visit(el) for el in node.elts]
        elementValues = self.popAllValues(len(node.elts))
        elementValues.reverse()
        for idx, value in enumerate(elementValues):
            self.__validate_container_entry(value, node.elts[idx])

        structTys = [v.type for v in elementValues]
        structTy = mlirTryCreateStructType(structTys, context=self.ctx)
        if structTy is None:
            self.emitFatalError(
                "hybrid quantum-classical data types and nested quantum"
                " structs are not allowed", node)

        if quake.StruqType.isinstance(structTy):
            self.pushValue(quake.MakeStruqOp(structTy, elementValues).result)
            return

        struct = cc.UndefOp(structTy)
        for idx, element in enumerate(elementValues):
            struct = cc.InsertValueOp(
                structTy, struct, element,
                DenseI64ArrayAttr.get([idx], context=self.ctx)).result
        self.pushValue(struct)

    def visit_UnaryOp(self, node):
        """
        Map unary operations in the Python AST to equivalents in MLIR.
        """

        self.visit(node.operand)
        operand = self.popValue()

        # Handle qubit negations
        if isinstance(node.op, ast.Invert):
            if quake.RefType.isinstance(operand.type):
                self.controlNegations.append(operand)
                self.pushValue(operand)
                return
            else:
                self.emitFatalError(
                    "unary operator ~ is only supported for values of type "
                    "qubit", node)

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
            self.pushValue(
                arith.XOrIOp(self.__arithmetic_to_bool(operand),
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

        # Note: including support for any non-arithmetic types (e.g. addition on
        # lists) needs to be tested/implemented also when used in list
        # comprehension expressions.
        if not self.isArithmeticType(left.type) or not self.isArithmeticType(
                right.type):
            self.emitFatalError(f'Invalid type for {nodeType}',
                                self.currentNode)

        # Based on the op type and the leaf types, create the MLIR operator
        if issubclass(nodeType, ast.Add):
            if IntegerType.isinstance(left.type):
                self.pushValue(arith.AddIOp(left, right).result)
                return
            elif (F64Type.isinstance(left.type) or
                  F32Type.isinstance(left.type)):
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
            elif (F64Type.isinstance(left.type) or
                  F32Type.isinstance(left.type)):
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
            if (F64Type.isinstance(left.type) or F32Type.isinstance(left.type)):
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
            elif (F64Type.isinstance(left.type) or
                  F32Type.isinstance(left.type)):
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
            if ((F64Type.isinstance(left.type) or F32Type.isinstance(left.type))
                    and IntegerType.isinstance(right.type)):
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
            if (F64Type.isinstance(left.type) or F32Type.isinstance(left.type)):
                self.pushValue(arith.RemFOp(left, right).result)
                return
            else:
                self.emitFatalError("unhandled BinOp.Mod types",
                                    self.currentNode)

        # FIXME: Refactor. The following error messages are all the same with


# minor variation.
        if issubclass(nodeType, ast.LShift):
            if IntegerType.isinstance(left.type):
                self.pushValue(arith.ShLIOp(left, right).result)
                return
            else:
                self.emitFatalError(
                    "unsupported operand type(s) for BinOp.LShift; "
                    "only integers supported", self.currentNode)

        if issubclass(nodeType, ast.RShift):
            if IntegerType.isinstance(left.type):
                self.pushValue(arith.ShRSIOp(left, right).result)
                return
            else:
                self.emitFatalError(
                    "unsupported operand type(s) for BinOp.RShift; "
                    "only integers supported", self.currentNode)

        if issubclass(nodeType, ast.BitAnd):
            if IntegerType.isinstance(left.type):
                self.pushValue(arith.AndIOp(left, right).result)
                return
            else:
                self.emitFatalError(
                    "unsupported operand type(s) for BinOp.BitAnd; "
                    "only integers supported", self.currentNode)

        if issubclass(nodeType, ast.BitOr):
            if IntegerType.isinstance(left.type):
                self.pushValue(arith.OrIOp(left, right).result)
                return
            else:
                self.emitFatalError(
                    "unsupported operand type(s) for BinOp.BitOr; "
                    "only integers supported", self.currentNode)

        if issubclass(nodeType, ast.BitXor):
            if IntegerType.isinstance(left.type):
                self.pushValue(arith.XOrIOp(left, right).result)
                return
            else:
                self.emitFatalError(
                    "unsupported operand type(s) for BinOp.BitXor; "
                    "only integers supported.", self.currentNode)

        self.emitFatalError("unhandled binary operator", self.currentNode)

    def visit_BinOp(self, node):
        """
        Visit binary operation nodes in the AST and map them to equivalents in
        the MLIR. This method handles arithmetic operations between values.
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

        self.pushPointerValue = True
        self.visit(node.target)
        self.pushPointerValue = False
        target = self.popValue()

        if not cc.PointerType.isinstance(target.type):
            self.emitFatalError(
                "augment-assign target variable is not defined or "
                "cannot be assigned to.", node)

        self.visit(node.value)
        value = self.popValue()
        loaded = cc.LoadOp(target).result

        # NOTE: `aug-assign` is usually defined as producing a value, which we
        # are not doing here. However, if this produces a value, then we need to
        # start worrying that arbitrary expressions might contain assignments,
        # which would require updates to the bridge in a bunch of places and add
        # some complexity. We hence effectively disallow using any kind of
        # assignment as expression.
        self.valueStack.pushFrame()
        self.__process_binary_op(loaded, value, type(node.op))
        self.valueStack.popFrame()
        res = self.popValue()

        if res.type != loaded.type:
            self.emitFatalError(
                "augment-assign must not change the variable type", node)
        cc.StoreOp(res, target)

    def appendToLiftedArgs(self, symbol):
        """
        Append `symbol` to the list of lifted arguments.

        Any free symbols (which include all device kernel references) found in a
        kernel decorator are lambda lifted. This enables the implementation to
        correctly implement Python's dynamic scope rules.

        All device kernel references must be resolved at a kernel decorator call
        site, and the resolution must include resolution of all lifted arguments
        (recursively).
        """
        if not self.liftedArgs:
            self.firstLiftedPos = len(self.entry.arguments)
        if symbol not in self.liftedArgs:
            self.liftedArgs.append(symbol)

    def visit_Name(self, node):
        """
        Visit `ast.Name` nodes and extract the correct value from the symbol
        table.
        """

        if self.symbolTable.isDefined(node.id):
            value = self.symbolTable[node.id]

            if (self.pushPointerValue or
                    not cc.PointerType.isinstance(value.type)):
                self.pushValue(value)
                return

            eleTy = cc.PointerType.getElementType(value.type)

            # Retain state types as pointers
            # (function arguments of `StateType` are passed as pointers)
            if cc.StateType.isinstance(eleTy):
                self.pushValue(value)
                return

            loaded = cc.LoadOp(value).result
            self.pushValue(loaded)
            return

        # Check if a non-local symbol, and process it.
        value = recover_value_of_or_none(node.id, None)
        if is_recovered_value_ok(value):
            from .kernel_decorator import isa_kernel_decorator
            from .kernel_builder import isa_dynamic_kernel
            if isa_kernel_decorator(value) or isa_dynamic_kernel(value):
                # Not a data variable. Symbol bound to kernel object. This case
                # is handled elsewhere.
                return

            # node.id is a non-local symbol. Lift it to a formal argument.
            self.dependentCaptureVars[node.id] = value
            # If `node.id` is in `liftedArgs`, it should already be in the
            # symbol table and processed.
            assert not node.id in self.liftedArgs
            self.appendToLiftedArgs(node.id)

            # Append as a new argument
            argTy = mlirTypeFromPyType(type(value), self.ctx, argInstance=value)
            mlirVal = cudaq_runtime.appendKernelArgument(
                self.kernelFuncOp, argTy)
            self.argTypes.append(argTy)

            assignNode = ast.Assign()
            assignNode.targets = [node]
            assignNode.value = mlirVal
            assignNode.lineno = node.lineno
            self.visit_Assign(assignNode)

            self.visit(node)
            self.pushValue(
                self.popValue())  # propagating the pushed value through
            return

        if node.id in globalRegisteredOperations:
            return

        if (self.__isUnitaryGate(node.id) or self.__isMeasurementGate(node.id)):
            return

        if node.id == 'complex':
            self.pushValue(self.getComplexType())
            return

        if node.id == 'float':
            self.pushValue(self.getFloatType())
            return

        self.emitFatalError(
            f"Invalid variable name requested - '{node.id}' is not defined within the scope it is used in.",
            node)


def compile_to_mlir(uniqueId, astModule,
                    capturedDataStorage: CapturedDataStorage, **kwargs):
    """
    Compile the given Python AST Module for the CUDA-Q kernel FunctionDef to an
    MLIR `ModuleOp`. Return both the `ModuleOp` and the list of function
    argument types as MLIR Types.

    This function will first check to see if there are any dependent kernels
    that are required by this function. If so, those kernels will also be
    compiled into the `ModuleOp`. The AST will be stored later for future
    potential dependent kernel lookups.
    """

    global globalAstRegistry
    verbose = 'verbose' in kwargs and kwargs['verbose']
    returnType = kwargs['returnType'] if 'returnType' in kwargs else None
    lineNumberOffset = kwargs['location'] if 'location' in kwargs else ('', 0)
    parentVariables = kwargs[
        'parentVariables'] if 'parentVariables' in kwargs else {}
    preCompile = kwargs['preCompile'] if 'preCompile' in kwargs else False
    kernelModuleName = kwargs[
        'kernelModuleName'] if 'kernelModuleName' in kwargs else None

    # Create the AST Bridge
    bridge = PyASTBridge(capturedDataStorage,
                         uniqueId=uniqueId,
                         verbose=verbose,
                         knownResultType=returnType,
                         returnTypeIsFromPython=True,
                         locationOffset=lineNumberOffset,
                         capturedVariables=parentVariables,
                         kernelModuleName=kernelModuleName)

    ValidateArgumentAnnotations(bridge).visit(astModule)
    ValidateReturnStatements(bridge).visit(astModule)

    if not preCompile:
        raise RuntimeError("must be precompile mode")

    # Build the AOT Quake Module for this kernel.
    bridge.visit(astModule)

    # Precompile (simplify) the Module.
    pm = PassManager.parse("builtin.module(aot-prep-pipeline)",
                           context=bridge.ctx)
    try:
        pm.run(bridge.module)
    except:
        raise RuntimeError(f"could not compile code for '{bridge.name}'.")

    bridge.module.operation.attributes.__setitem__(
        cudaq__unique_attr_name, StringAttr.get(bridge.name,
                                                context=bridge.ctx))
    if verbose:
        print(bridge.module)
    extraMetaData = {}
    extraMetaData['dependent_captures'] = bridge.dependentCaptureVars
    # Clear the live operations cache. This avoids python crashing with
    # stale references being cached.
    bridge.module.context._clear_live_operations()
    # The only MLIR code object wrapped & tracked ought to be `newMod` now.
    cudaq_runtime.set_data_layout(bridge.module)
    return bridge.module, bridge.argTypes, extraMetaData, bridge.liftedArgs, bridge.firstLiftedPos
