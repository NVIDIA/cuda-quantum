# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import ast
import inspect
import importlib
import textwrap

from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime
from cudaq.mlir.dialects import cc
from .utils import globalAstRegistry, globalKernelRegistry, mlirTypeFromAnnotation


class FindDepKernelsVisitor(ast.NodeVisitor):

    def __init__(self, ctx):
        self.depKernels = {}
        self.context = ctx
        self.kernelName = ''

    def visit_FunctionDef(self, node):
        """
        Here we will look at this Functions arguments, if 
        there is a Callable, we will add any seen kernel/AST with the same 
        signature to the dependent kernels map. This enables the creation 
        of `ModuleOps` that contain all the functions necessary to inline and 
        synthesize callable block arguments.
        """
        self.kernelName = node.name
        for arg in node.args.args:
            annotation = arg.annotation
            if annotation == None:
                raise RuntimeError(
                    'cudaq.kernel functions must have argument type annotations.'
                )
            if isinstance(annotation, ast.Subscript) and hasattr(
                    annotation.value,
                    "id") and annotation.value.id == 'Callable':
                if not hasattr(annotation, 'slice'):
                    raise RuntimeError(
                        'Callable type must have signature specified.')

                # This is callable, let's add all in scope kernels with
                # the same signature
                callableTy = mlirTypeFromAnnotation(annotation, self.context)
                for k, v in globalKernelRegistry.items():
                    if str(v.type) == str(
                            cc.CallableType.getFunctionType(callableTy)):
                        self.depKernels[k] = globalAstRegistry[k]

        [self.visit(stm) for stm in node.body]

    def visit_Attribute(self, node):
        if not self.kernelName:
            return
        if node.attr in globalAstRegistry:
            self.depKernels[node.attr] = globalAstRegistry[node.attr]
        self.visit(node.value)

    def visit_Name(self, node):
        if not self.kernelName:
            return
        if node.id in globalAstRegistry:
            self.depKernels[node.id] = globalAstRegistry[node.id]


class HasReturnNodeVisitor(ast.NodeVisitor):
    """
    This visitor will visit the function definition and report 
    true if that function has a return statement.
    """

    def __init__(self):
        self.hasReturnNode = False

    def visit_FunctionDef(self, node):
        for n in node.body:
            if isinstance(n, ast.Return) and n.value != None:
                self.hasReturnNode = True


class FindDepFuncsVisitor(ast.NodeVisitor):
    """
    Populate a list of function names that have `ast.Call` nodes in them. This
    only populates functions, not attributes (like `np.sum()`).
    """

    def __init__(self):
        self.func_names = set()

    def visit_Call(self, node):
        if hasattr(node, 'func'):
            if isinstance(node.func, ast.Name):
                self.func_names.add(node.func.id)


class FetchDepFuncsSourceCode:
    """
    For a given function (or lambda), fetch the source code of the function,
    along with the source code of all the of the recursively nested functions
    invoked in that function. The main public function is `fetch`.
    """

    def __init__(self):
        pass

    @staticmethod
    def _isLambda(obj):
        return hasattr(obj, '__name__') and obj.__name__ == '<lambda>'

    @staticmethod
    def _getFuncObj(name: str, calling_frame: object):
        currFrame = calling_frame
        while currFrame:
            if name in currFrame.f_locals:
                if inspect.isfunction(currFrame.f_locals[name]
                                     ) or FetchDepFuncsSourceCode._isLambda(
                                         currFrame.f_locals[name]):
                    return currFrame.f_locals[name]
            currFrame = currFrame.f_back
        return None

    @staticmethod
    def _getChildFuncNames(func_obj: object,
                           calling_frame: object,
                           name: str = None,
                           full_list: list = None,
                           visit_set: set = None,
                           nest_level: int = 0) -> list:
        """
        Recursively populate a list of function names that are called by a parent
        `func_obj`. Set all parameters except `func_obj` to `None` for the top-level
        call to this function.
        """
        if full_list is None:
            full_list = []
        if visit_set is None:
            visit_set = set()
        if not inspect.isfunction(
                func_obj) and not FetchDepFuncsSourceCode._isLambda(func_obj):
            return full_list
        if name is None:
            name = func_obj.__name__

        tree = ast.parse(textwrap.dedent(inspect.getsource(func_obj)))
        vis = FindDepFuncsVisitor()
        visit_set.add(name)
        vis.visit(tree)
        for f in vis.func_names:
            if f not in visit_set:
                childFuncObj = FetchDepFuncsSourceCode._getFuncObj(
                    f, calling_frame)
                if childFuncObj:
                    FetchDepFuncsSourceCode._getChildFuncNames(
                        childFuncObj, calling_frame, f, full_list, visit_set,
                        nest_level + 1)
        full_list.append(name)
        return full_list

    @staticmethod
    def fetch(func_obj: object):
        """
        Given an input `func_obj`, fetch the source code of that function, and
        all the required child functions called by that function. This does not
        support fetching class attributes/methods.
        """
        callingFrame = inspect.currentframe().f_back
        func_name_list = FetchDepFuncsSourceCode._getChildFuncNames(
            func_obj, callingFrame)
        code = ''
        for funcName in func_name_list:
            # Get the function source
            if funcName == func_obj.__name__:
                this_func_obj = func_obj
            else:
                this_func_obj = FetchDepFuncsSourceCode._getFuncObj(
                    funcName, callingFrame)
            src = textwrap.dedent(inspect.getsource(this_func_obj))

            code += src + '\n'

        return code


class ValidateArgumentAnnotations(ast.NodeVisitor):
    """
    Utility visitor for finding argument annotations
    """

    def __init__(self, bridge):
        self.bridge = bridge

    def visit_FunctionDef(self, node):
        for arg in node.args.args:
            if arg.annotation == None:
                self.bridge.emitFatalError(
                    'cudaq.kernel functions must have argument type annotations.',
                    arg)


class ValidateReturnStatements(ast.NodeVisitor):
    """
    Analyze the AST and ensure that functions with a return-type annotation
    actually have a return statement in all paths.
    """

    def __init__(self, bridge):
        self.bridge = bridge

    def visit_FunctionDef(self, node):
        # skip if un-annotated or explicitly marked as None
        is_none_ret = (isinstance(node.returns, ast.Constant) and
                       node.returns.value
                       is None) or (isinstance(node.returns, ast.Name) and
                                    node.returns.id == 'None')

        if node.returns is None or is_none_ret:
            return self.generic_visit(node)

        def all_paths_return(stmts):
            for stmt in stmts:
                if isinstance(stmt, ast.Return):
                    return True

                if isinstance(stmt, ast.If):
                    if all_paths_return(stmt.body) and all_paths_return(
                            stmt.orelse):
                        return True

                if isinstance(stmt, (ast.For, ast.While)):
                    if all_paths_return(stmt.body) or all_paths_return(
                            stmt.orelse):
                        return True

            return False

        if not all_paths_return(node.body):
            self.bridge.emitFatalError(
                'cudaq.kernel functions with return type annotations must have a return statement.',
                node)

        self.generic_visit(node)
