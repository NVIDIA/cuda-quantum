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

class PyObjectiveFunctionDecorator(object):
    """
    The `PyObjectiveFunctionDecorator` serves as a standard Python decorator that takes 
    the decorated function as input and compiles it to the executable code. 
    """

    def __init__(self, function, verbose=False, module=None, kernelName=None) -> None:
        self.function = function
        self.module = None if module is None else module
        self.verbose = verbose
        self.name = kernelName if kernelName is not None else self.function.__name__
        self.argTypes = None
        self.location = (
            inspect.getfile(self.function),
            inspect.getsourcelines(self.function)[1]
        ) if self.function is not None else ('', 0)

        # Get any global variables from parent scope.
        self.parentFrame = inspect.stack()[2].frame
        self.globalScopedVars = {
            k: v for k, v in dict(inspect.getmembers(self.parentFrame))
            ['f_locals'].items()
        }
        self.dependentCaptures = None

        # Get the function source
        src = inspect.getsource(self.function)

        # Strip off the extra tabs
        leadingSpaces = len(src) - len(src.lstrip())
        self.funcSrc = '\n'.join(
            [line[leadingSpaces:] for line in src.split('\n')])

        # Create the AST
        self.parsed_ast = ast.parse(self.funcSrc)
        if verbose and importlib.util.find_spec('astpretty') is not None:
            import astpretty
            astpretty.pprint(self.parsed_ast.body[0])

        # Assign the signature for use later and
        # keep a list of arguments (used for validation in the runtime)
        self.signature = inspect.getfullargspec(self.function).annotations
        self.arguments = [
            (k, v) for k, v in self.signature.items() if k != 'return'
        ]
        self.returnType = self.signature[
            'return'] if 'return' in self.signature else None

        # Validate that we have a return type annotation if necessary
        hasRetNodeVis = HasReturnNodeVisitor()
        hasRetNodeVis.visit(self.parsed_ast)
        if hasRetNodeVis.hasReturnNode and 'return' not in self.signature:
            emitFatalError(
                'CUDA-Q objective function has return statement but no return type annotation.'
            )

        # Store the AST for this objective function, it is needed for
        # building up call graphs. We also must retain
        # the source code location for error diagnostics
        globalAstRegistry[self.name] = (self.parsed_ast, self.location)
    
    def compile(self):
        """
        Compile the Python objective function AST. This is a no-op 
        if the objective function is already compiled. 
        """

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

        self.compiled_bytecode = compile(
            self.parsed_ast,
            '',
            mode = 'exec'
        )

    def __call__(self, *args):
        """
        Invoke the CUDA-Q objective function. 
        """

        # Compile, no-op if the module is not None
        self.compile()

        exec(self.compiled_bytecode)


def objective_function(function=None, **kwargs):
    """
    The `cudaq.objective_function` represents the CUDA-Q language function 
    attribute that programmers leverage to indicate the following function 
    is a CUDA-Q objective function and should be compile and executed on 
    an available quantum coprocessor.

    Verbose logging can be enabled via `verbose=True`. 
    """
    if function:
        return PyObjectiveFunctionDecorator(function)
    else:

        def wrapper(function):
            return PyObjectiveFunctionDecorator(function, **kwargs)

        return wrapper