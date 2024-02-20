# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import ast
import importlib
import inspect
from typing import Callable
from ..mlir.ir import *
from ..mlir.passmanager import *
from ..mlir.dialects import quake, cc
from .ast_bridge import compile_to_mlir, PyASTBridge
from .utils import mlirTypeFromPyType, nvqppPrefix, mlirTypeToPyType, globalAstRegistry
from .qubit_qis import h, x, y, z, s, t, rx, ry, rz, r1, swap, exp_pauli, mx, my, mz, adjoint, control, compute_action
from .analysis import MidCircuitMeasurementAnalyzer, RewriteMeasures
from ..mlir._mlir_libs._quakeDialects import cudaq_runtime

# This file implements the decorator mechanism needed to
# JIT compile CUDA Quantum kernels. It exposes the cudaq.kernel()
# decorator which hooks us into the JIT compilation infrastructure
# which maps the AST representation to an MLIR representation and ultimately
# executable code.

globalImportedKernels = {}


class PyKernelDecorator(object):
    """
    The `PyKernelDecorator` serves as a standard Python decorator that takes 
    the decorated function as input and optionally lowers its  AST 
    representation to executable code via MLIR. This decorator enables full JIT
    compilation mode, where the function is lowered to an MLIR representation.

    This decorator exposes a call overload that executes the code via the 
    MLIR `ExecutionEngine` if not in library mode. 
    """

    def __init__(self, function, verbose=False, module=None, kernelName=None):
        global globalImportedKernels
        self.kernelFunction = function
        self.module = None if module == None else module
        self.verbose = verbose
        self.name = kernelName if kernelName != None else self.kernelFunction.__name__
        self.argTypes = None

        if self.kernelFunction is None:
            if self.module is not None:
                ## ASKME: Is the following still needed?
                # Could be that we don't have a function
                # but someone has provided an external Module.
                # But if we want this new decorator to be callable
                # we'll need to turn library_mode off and set the `argTypes`
                symbols = SymbolTable(self.module.operation)
                if nvqppPrefix + self.name in symbols:
                    function = symbols[nvqppPrefix + self.name]
                    entryBlock = function.entry_block
                    self.argTypes = [v.type for v in entryBlock.arguments]
                    self.signature = {
                        'arg{}'.format(i): mlirTypeToPyType(v.type)
                        for i, v in enumerate(self.argTypes)
                    }
                return
            else:
                raise RuntimeError(
                    "invalid kernel decorator. module and function are both None."
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
        self.arguments = [(k, v) for k, v in self.signature.items()]

        # Run analyzers and attach metadata (only have 1 right now)
        analyzer = MidCircuitMeasurementAnalyzer()
        analyzer.visit(self.astModule)
        self.metadata = {'conditionalOnMeasure': analyzer.hasMidCircuitMeasures}

        # JIT compile to MLIR
        self.module, self.argTypes = compile_to_mlir(self.astModule,
                                                     verbose=self.verbose)
        if self.metadata['conditionalOnMeasure']:
            SymbolTable(
                self.module.operation)[nvqppPrefix +
                                       self.name].attributes.__setitem__(
                                           'qubitMeasurementFeedback',
                                           BoolAttr.get(
                                               True,
                                               context=self.module.context))

    def __str__(self):
        if not self.module == None:
            return str(self.module)
        return "{cannot print this kernel}"

    def __call__(self, *args):

        if self.kernelFunction is None:
            if self.module is None:
                raise RuntimeError(
                    "this kernel is not callable (no function or MLIR Module found)"
                )
            if self.argTypes is None:
                raise RuntimeError(
                    "this kernel is not callable (no function and no MLIR argument types found)"
                )

        if len(args) != len(self.argTypes):
            raise RuntimeError(
                "Incorrect number of runtime arguments provided to kernel {} ({} required, {} provided)"
                .format(self.name, len(self.argTypes), len(args)))

        # validate the argument types
        processedArgs = []
        callableNames = []
        for i, arg in enumerate(args):
            mlirType = mlirTypeFromPyType(type(arg),
                                          self.module.context,
                                          argInstance=arg,
                                          argTypeToCompareTo=self.argTypes[i])
            if not cc.CallableType.isinstance(
                    mlirType) and mlirType != self.argTypes[i]:
                raise RuntimeError("invalid runtime arg type ({} vs {})".format(
                    mlirType, self.argTypes[i]))
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
                    tmpBridge.visit(globalAstRegistry[arg.name])

            # Convert `numpy` arrays to lists
            if cc.StdvecType.isinstance(mlirType) and hasattr(arg, "tolist"):
                if arg.ndim != 1:
                    raise RuntimeError(
                        'CUDA Quantum kernels only support 1D numpy array arguments.'
                    )
                processedArgs.append(arg.tolist())
            else:
                processedArgs.append(arg)

        cudaq_runtime.pyAltLaunchKernel(self.name,
                                        self.module,
                                        *processedArgs,
                                        callable_names=callableNames)


def kernel(function=None, **kwargs):
    """
    The `cudaq.kernel` represents the CUDA Quantum language function 
    attribute that programmers leverage to indicate the following function 
    is a CUDA Quantum kernel and should be compile and executed on 
    an available quantum coprocessor.

    One can indicate JIT compilation to MLIR via the `jit=True` flag. Verbose 
    logging can also be enabled via verbose=True. 
    """
    if function:
        return PyKernelDecorator(function)
    else:

        def wrapper(function):
            return PyKernelDecorator(function, **kwargs)

        return wrapper
