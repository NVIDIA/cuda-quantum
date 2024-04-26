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
from typing import Callable
from ..mlir.ir import *
from ..mlir.passmanager import *
from ..mlir.dialects import quake, cc
from .ast_bridge import compile_to_mlir, PyASTBridge
from .utils import mlirTypeFromPyType, nvqppPrefix, mlirTypeToPyType, globalAstRegistry, emitFatalError, emitErrorIfInvalidPauli
from .analysis import MidCircuitMeasurementAnalyzer, RewriteMeasures, HasReturnNodeVisitor
from ..mlir._mlir_libs._quakeDialects import cudaq_runtime

# This file implements the decorator mechanism needed to
# JIT compile CUDA Quantum kernels. It exposes the cudaq.kernel()
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

    def __init__(self, function, verbose=False, module=None, kernelName=None):
        self.kernelFunction = function
        self.module = None if module == None else module
        self.verbose = verbose
        self.name = kernelName if kernelName != None else self.kernelFunction.__name__
        self.argTypes = None
        self.location = (inspect.getfile(self.kernelFunction),
                         inspect.getsourcelines(self.kernelFunction)[1]
                        ) if self.kernelFunction is not None else ('', 0)

        # Get any global variables from parent scope.
        # We filter only types we accept: integers and floats.
        # Note here we assume that the parent scope is 2 stack frames up
        self.parentFrame = inspect.stack()[2].frame
        self.globalScopedVars = {
            k: v for k, v in dict(inspect.getmembers(self.parentFrame))
            ['f_locals'].items()
        }

        # Once the kernel is compiled to MLIR, we
        # want to know what capture variables, if any, were
        # used in the kernel. We need to track these.
        self.dependentCaptures = None

        if self.kernelFunction is None:
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
        self.signature = inspect.getfullargspec(self.kernelFunction).annotations
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
                'CUDA Quantum kernel has return statement but no return type annotation.'
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
                        if self.globalScopedVars[k] != v:
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

    def __call__(self, *args):
        """
        Invoke the CUDA Quantum kernel. JIT compilation of the 
        kernel AST to MLIR will occur here if it has not already occurred. 
        """

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
            if cc.StdvecType.isinstance(mlirType):
                if cc.StdvecType.isinstance(self.argTypes[i]):
                    argEleTy = cc.StdvecType.getElementType(mlirType)
                    eleTy = cc.StdvecType.getElementType(self.argTypes[i])
                    if F64Type.isinstance(eleTy) and IntegerType.isinstance(
                            argEleTy):
                        processedArgs.append([float(i) for i in arg])
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
                    tmpBridge = PyASTBridge(existingModule=self.module,
                                            disableEntryPointTag=True)
                    tmpBridge.visit(globalAstRegistry[arg.name][0])

            # Convert `numpy` arrays to lists
            if cc.StdvecType.isinstance(mlirType) and hasattr(arg, "tolist"):
                if arg.ndim != 1:
                    emitFatalError(
                        f"CUDA Quantum kernels only support array arguments from NumPy that are one dimensional (input argument {i} has shape = {arg.shape})."
                    )
                processedArgs.append(arg.tolist())
            else:
                processedArgs.append(arg)

        if self.returnType == None:
            cudaq_runtime.pyAltLaunchKernel(self.name,
                                            self.module,
                                            *processedArgs,
                                            callable_names=callableNames)
        else:
            return cudaq_runtime.pyAltLaunchKernelR(
                self.name,
                self.module,
                mlirTypeFromPyType(self.returnType, self.module.context),
                *processedArgs,
                callable_names=callableNames)


def kernel(function=None, **kwargs):
    """
    The `cudaq.kernel` represents the CUDA Quantum language function 
    attribute that programmers leverage to indicate the following function 
    is a CUDA Quantum kernel and should be compile and executed on 
    an available quantum coprocessor.

    Verbose logging can be enabled via `verbose=True`. 
    """
    if function:
        return PyKernelDecorator(function)
    else:

        def wrapper(function):
            return PyKernelDecorator(function, **kwargs)

        return wrapper
