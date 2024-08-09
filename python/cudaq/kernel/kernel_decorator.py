# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import ast, sys, traceback
import importlib
import inspect
import json
from typing import Callable
from ..mlir.ir import *
from ..mlir.passmanager import *
from ..mlir.dialects import quake, cc
from .ast_bridge import compile_to_mlir, PyASTBridge
from .utils import mlirTypeFromPyType, nvqppPrefix, mlirTypeToPyType, globalAstRegistry, emitFatalError, emitErrorIfInvalidPauli, globalRegisteredTypes
from .analysis import MidCircuitMeasurementAnalyzer, HasReturnNodeVisitor
from ..mlir._mlir_libs._quakeDialects import cudaq_runtime
from .captured_data import CapturedDataStorage

import numpy as np

# This file implements the decorator mechanism needed to
# JIT compile CUDA-Q kernels. It exposes the cudaq.kernel()
# decorator which hooks us into the JIT compilation infrastructure
# which maps the AST representation to an MLIR representation and ultimately
# executable code.


class PyKernelDecorator(object):
    """
    The `PyKernelDecorator` serves as a standard Python decorator that takes 
    the decorated function as input and optionally lowers its  AST 
    representation to executable code via MLIR. This decorator enables full JIT
    compilation mode, where the function is lowered to an MLIR representation.

    This decorator exposes a call overload that executes the code via the 
    MLIR `ExecutionEngine` for the MLIR mode. 
    """

    def __init__(self,
                 function,
                 verbose=False,
                 module=None,
                 kernelName=None,
                 funcSrc=None,
                 signature=None,
                 location=None,
                 overrideGlobalScopedVars=None):

        is_deserializing = isinstance(function, str)

        # When initializing with a provided `funcSrc`, we cannot use inspect
        # because we only have a string for the function source. That is - the
        # "function" isn't actually a concrete Python Function object in memory
        # that we can "inspect". Hence, use alternate approaches when
        # initializing from `funcSrc`.
        if is_deserializing:
            self.kernelFunction = None
            self.name = kernelName
            self.location = location
            self.signature = signature
        else:
            self.kernelFunction = function
            self.name = kernelName if kernelName != None else self.kernelFunction.__name__
            self.location = (inspect.getfile(self.kernelFunction),
                             inspect.getsourcelines(self.kernelFunction)[1]
                            ) if self.kernelFunction is not None else ('', 0)

        self.capturedDataStorage = None

        self.module = module
        self.verbose = verbose
        self.argTypes = None

        # Get any global variables from parent scope.
        # We filter only types we accept: integers and floats.
        # Note here we assume that the parent scope is 2 stack frames up
        self.parentFrame = inspect.stack()[2].frame
        if overrideGlobalScopedVars:
            self.globalScopedVars = {
                k: v for k, v in overrideGlobalScopedVars.items()
            }
        else:
            self.globalScopedVars = {
                k: v for k, v in dict(inspect.getmembers(self.parentFrame))
                ['f_locals'].items()
            }

        # Register any external class types that may be used
        # in the kernel definition
        for name, var in self.globalScopedVars.items():
            if isinstance(var, type):
                globalRegisteredTypes[name] = (var, var.__annotations__)

        # Once the kernel is compiled to MLIR, we
        # want to know what capture variables, if any, were
        # used in the kernel. We need to track these.
        self.dependentCaptures = None

        if self.kernelFunction is None and not is_deserializing:
            if self.module is not None:
                # Could be that we don't have a function
                # but someone has provided an external Module.
                # If we want this new decorator to be callable
                # we'll need to set the `argTypes`
                symbols = SymbolTable(self.module.operation)
                if nvqppPrefix + self.name in symbols:
                    function = symbols[nvqppPrefix + self.name]
                    entryBlock = function.entry_block
                    self.argTypes = [v.type for v in entryBlock.arguments]
                    self.signature = {
                        'arg{}'.format(i): mlirTypeToPyType(v.type)
                        for i, v in enumerate(self.argTypes)
                    }
                    self.returnType = self.signature[
                        'return'] if 'return' in self.signature else None
                return
            else:
                emitFatalError(
                    "Invalid kernel decorator. Module and function are both None."
                )

        if is_deserializing:
            self.funcSrc = funcSrc
        else:
            # Get the function source
            src = inspect.getsource(self.kernelFunction)

            # Strip off the extra tabs
            leadingSpaces = len(src) - len(src.lstrip())
            self.funcSrc = '\n'.join(
                [line[leadingSpaces:] for line in src.split('\n')])

        # Create the AST
        self.astModule = ast.parse(self.funcSrc)
        if verbose and importlib.util.find_spec('astpretty') is not None:
            import astpretty
            astpretty.pprint(self.astModule.body[0])

        # Assign the signature for use later and
        # keep a list of arguments (used for validation in the runtime)
        if not is_deserializing:
            self.signature = inspect.getfullargspec(
                self.kernelFunction).annotations
        self.arguments = [
            (k, v) for k, v in self.signature.items() if k != 'return'
        ]
        self.returnType = self.signature[
            'return'] if 'return' in self.signature else None

        # Validate that we have a return type annotation if necessary
        hasRetNodeVis = HasReturnNodeVisitor()
        hasRetNodeVis.visit(self.astModule)
        if hasRetNodeVis.hasReturnNode and 'return' not in self.signature:
            emitFatalError(
                'CUDA-Q kernel has return statement but no return type annotation.'
            )

        # Run analyzers and attach metadata (only have 1 right now)
        analyzer = MidCircuitMeasurementAnalyzer()
        analyzer.visit(self.astModule)
        self.metadata = {'conditionalOnMeasure': analyzer.hasMidCircuitMeasures}

        # Store the AST for this kernel, it is needed for
        # building up call graphs. We also must retain
        # the source code location for error diagnostics
        globalAstRegistry[self.name] = (self.astModule, self.location)

    def compile(self):
        """
        Compile the Python function AST to MLIR. This is a no-op 
        if the kernel is already compiled. 
        """

        # Before we can execute, we need to make sure
        # variables from the parent frame that we captured
        # have not changed. If they have changed, we need to
        # recompile with the new values.
        s = inspect.currentframe()
        while s:
            if s == self.parentFrame:
                # We found the parent frame, now
                # see if any of the variables we depend
                # on have changed.
                self.globalScopedVars = {
                    k: v
                    for k, v in dict(inspect.getmembers(s))['f_locals'].items()
                }
                if self.dependentCaptures != None:
                    for k, v in self.dependentCaptures.items():
                        if (isinstance(v, (list, np.ndarray))):
                            if not all(a == b for a, b in zip(
                                    self.globalScopedVars[k], v)):
                                # Recompile if values in the list have changed.
                                self.module = None
                                break
                        elif self.globalScopedVars[k] != v:
                            # Need to recompile
                            self.module = None
                            break
                break
            s = s.f_back

        if self.module != None:
            return

        self.module, self.argTypes, extraMetadata = compile_to_mlir(
            self.astModule,
            self.metadata,
            self.capturedDataStorage,
            verbose=self.verbose,
            returnType=self.returnType,
            location=self.location,
            parentVariables=self.globalScopedVars)

        # Grab the dependent capture variables, if any
        self.dependentCaptures = extraMetadata[
            'dependent_captures'] if 'dependent_captures' in extraMetadata else None

    def __str__(self):
        """
        Return the MLIR Module string representation for this kernel.
        """
        self.compile()
        return str(self.module)

    def _repr_svg_(self):
        """
        Return the SVG representation of `self` (:class:`PyKernelDecorator`).
        This assumes no arguments are required to execute the kernel,
        and `latex` (with `quantikz` package) and `dvisvgm` are installed,
        and the temporary directory is writable.
        If any of these assumptions fail, returns None.
        """
        self.compile()  # compile if not yet compiled
        if self.argTypes is None or len(self.argTypes) != 0:
            return None
        from cudaq import getSVGstring

        try:
            from subprocess import CalledProcessError

            try:
                return getSVGstring(self)
            except CalledProcessError:
                return None
        except ImportError:
            return None

    def isCastable(self, fromTy, toTy):
        if F64Type.isinstance(toTy):
            return F32Type.isinstance(fromTy) or IntegerType.isinstance(fromTy)

        if F32Type.isinstance(toTy):
            return F64Type.isinstance(fromTy) or IntegerType.isinstance(fromTy)

        if ComplexType.isinstance(toTy):
            floatToType = ComplexType(toTy).element_type
            if ComplexType.isinstance(fromTy):
                floatFromType = ComplexType(fromTy).element_type
                return self.isCastable(floatFromType, floatToType)

            return fromTy == floatToType or self.isCastable(fromTy, floatToType)

        return False

    def castPyList(self, fromEleTy, toEleTy, list):
        if self.isCastable(fromEleTy, toEleTy):
            if F64Type.isinstance(toEleTy):
                return [float(i) for i in list]

            if F32Type.isinstance(toEleTy):
                return [np.float32(i) for i in list]

            if ComplexType.isinstance(toEleTy):
                floatToType = ComplexType(toEleTy).element_type

                if F64Type.isinstance(floatToType):
                    return [complex(i) for i in list]

                return [np.complex64(i) for i in list]
        return list

    def createStorage(self):
        ctx = None if self.module == None else self.module.context
        return CapturedDataStorage(ctx=ctx,
                                   loc=self.location,
                                   name=self.name,
                                   module=self.module)

    @staticmethod
    def type_to_str(t):
        """
        This converts types to strings in a clean JSON-compatible way.
        int -> 'int'
        list[float] -> 'list[float]'
        List[float] -> 'list[float]'
        """
        if hasattr(t, '__origin__') and t.__origin__ is not None:
            # Handle generic types from typing
            origin = t.__origin__
            args = t.__args__
            args_str = ', '.join(
                PyKernelDecorator.type_to_str(arg) for arg in args)
            return f'{origin.__name__}[{args_str}]'
        elif hasattr(t, '__name__'):
            return t.__name__
        else:
            return str(t)

    def to_json(self):
        """
        Convert `self` to a JSON-serialized version of the kernel such that
        `from_json` can reconstruct it elsewhere.
        """
        obj = dict()
        obj['name'] = self.name
        obj['location'] = self.location
        obj['funcSrc'] = self.funcSrc
        obj['signature'] = {
            k: PyKernelDecorator.type_to_str(v)
            for k, v in self.signature.items()
        }
        return json.dumps(obj)

    @staticmethod
    def from_json(jStr, overrideDict=None):
        """
        Convert a JSON string into a new PyKernelDecorator object.
        """
        j = json.loads(jStr)
        return PyKernelDecorator(
            'kernel',  # just set to any string
            verbose=False,
            module=None,
            kernelName=j['name'],
            funcSrc=j['funcSrc'],
            signature=j['signature'],
            location=j['location'],
            overrideGlobalScopedVars=overrideDict)

    def __call__(self, *args):
        """
        Invoke the CUDA-Q kernel. JIT compilation of the 
        kernel AST to MLIR will occur here if it has not already occurred. 
        """
        # Prepare captured state storage for the run
        self.capturedDataStorage = self.createStorage()

        # Compile, no-op if the module is not None
        self.compile()

        if len(args) != len(self.argTypes):
            emitFatalError(
                f"Incorrect number of runtime arguments provided to kernel `{self.name}` ({len(self.argTypes)} required, {len(args)} provided)"
            )

        # validate the argument types
        processedArgs = []
        callableNames = []
        for i, arg in enumerate(args):
            if isinstance(arg, PyKernelDecorator):
                arg.compile()

            if isinstance(arg, str):
                # Only allow `pauli_word` as string input
                emitErrorIfInvalidPauli(arg)
                arg = cudaq_runtime.pauli_word(arg)

            if issubclass(type(arg), list):
                if all(isinstance(a, str) for a in arg):
                    [emitErrorIfInvalidPauli(a) for a in arg]
                    arg = [cudaq_runtime.pauli_word(a) for a in arg]

            mlirType = mlirTypeFromPyType(type(arg),
                                          self.module.context,
                                          argInstance=arg,
                                          argTypeToCompareTo=self.argTypes[i])

            # Support passing `list[int]` to a `list[float]` argument
            # Support passing `list[int]` or `list[float]` to a `list[complex]` argument
            if cc.StdvecType.isinstance(mlirType):
                if cc.StdvecType.isinstance(self.argTypes[i]):
                    argEleTy = cc.StdvecType.getElementType(mlirType)  # actual
                    eleTy = cc.StdvecType.getElementType(
                        self.argTypes[i])  # formal

                    if self.isCastable(argEleTy, eleTy):
                        processedArgs.append(
                            self.castPyList(argEleTy, eleTy, arg))
                        mlirType = self.argTypes[i]
                        continue

            if not cc.CallableType.isinstance(
                    mlirType) and mlirType != self.argTypes[i]:
                emitFatalError(
                    f"Invalid runtime argument type. Argument of type {mlirTypeToPyType(mlirType)} was provided, but {mlirTypeToPyType(self.argTypes[i])} was expected."
                )

            if cc.CallableType.isinstance(mlirType):
                # Assume this is a PyKernelDecorator
                callableNames.append(arg.name)
                # It may be that the provided input callable kernel
                # is not currently in the ModuleOp. Need to add it
                # if that is the case, we have to use the AST
                # so that it shares self.module's MLIR Context
                symbols = SymbolTable(self.module.operation)
                if nvqppPrefix + arg.name not in symbols:
                    tmpBridge = PyASTBridge(self.capturedDataStorage,
                                            existingModule=self.module,
                                            disableEntryPointTag=True)
                    tmpBridge.visit(globalAstRegistry[arg.name][0])

            # Convert `numpy` arrays to lists
            if cc.StdvecType.isinstance(mlirType) and hasattr(arg, "tolist"):
                if arg.ndim != 1:
                    emitFatalError(
                        f"CUDA-Q kernels only support array arguments from NumPy that are one dimensional (input argument {i} has shape = {arg.shape})."
                    )
                processedArgs.append(arg.tolist())
            else:
                processedArgs.append(arg)

        if self.returnType == None:
            cudaq_runtime.pyAltLaunchKernel(self.name,
                                            self.module,
                                            *processedArgs,
                                            callable_names=callableNames)
            self.capturedDataStorage.__del__()
            self.capturedDataStorage = None
        else:
            result = cudaq_runtime.pyAltLaunchKernelR(
                self.name,
                self.module,
                mlirTypeFromPyType(self.returnType, self.module.context),
                *processedArgs,
                callable_names=callableNames)

            self.capturedDataStorage.__del__()
            self.capturedDataStorage = None
            return result


def kernel(function=None, **kwargs):
    """
    The `cudaq.kernel` represents the CUDA-Q language function 
    attribute that programmers leverage to indicate the following function 
    is a CUDA-Q kernel and should be compile and executed on 
    an available quantum coprocessor.

    Verbose logging can be enabled via `verbose=True`. 
    """
    if function:
        return PyKernelDecorator(function)
    else:

        def wrapper(function):
            return PyKernelDecorator(function, **kwargs)

        return wrapper
