# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import ast, inspect, importlib, textwrap
from .utils import globalAstRegistry, globalKernelRegistry, mlirTypeFromAnnotation
from ..mlir.dialects import cc
from ..mlir.ir import *


class MidCircuitMeasurementAnalyzer(ast.NodeVisitor):
    """
    The `MidCircuitMeasurementAnalyzer` is a utility class searches for 
    common measurement - conditional patterns to indicate to the runtime 
    that we have a circuit with mid-circuit measurement and subsequent conditional 
    quantum operation application.
    """

    def __init__(self):
        self.measureResultsVars = []
        self.hasMidCircuitMeasures = False

    def isMeasureCallOp(self, node):
        return isinstance(
            node, ast.Call) and node.__dict__['func'].id in ['mx', 'my', 'mz']

    def visit_Assign(self, node):
        target = node.targets[0]
        # Check if a variable is assigned from result(s) of measurement
        if hasattr(node, 'value') and hasattr(
                node.value, 'id') and node.value.id in self.measureResultsVars:
            self.measureResultsVars.append(target.id)
            return
        if not 'func' in node.value.__dict__:
            return
        creatorFunc = node.value.func
        if 'id' in creatorFunc.__dict__ and creatorFunc.id in [
                'mx', 'my', 'mz'
        ]:
            self.measureResultsVars.append(target.id)

    # Get the variable name from a variable node.
    # Returns an empty string if not something we know how to get a variable name from.
    def getVariableName(self, node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Subscript):
            return self.getVariableName(node.value)
        return ''

    def checkForMeasureResult(self, value):
        return self.isMeasureCallOp(value) or self.getVariableName(
            value) in self.measureResultsVars

    def visit_If(self, node):
        condition = node.test

        # Catch `if mz(q)`, `if val`, where `val = mz(q)` or `if var[k]`, where `var = mz(qvec)`
        if self.checkForMeasureResult(condition):
            self.hasMidCircuitMeasures = True
            return

        # Compare: look at left expression.
        # Catch `if var == True/False` and `if var[k] == True/False:` or `if mz(q) == True/False`
        if isinstance(condition, ast.Compare) and self.checkForMeasureResult(
                condition.left):
            self.hasMidCircuitMeasures = True
            return

        # Catch `if UnaryOp mz(q)` or `if UnaryOp var` (`var = mz(q)`)
        if isinstance(condition, ast.UnaryOp) and self.checkForMeasureResult(
                condition.operand):
            self.hasMidCircuitMeasures = True
            return

        # Catch `if something BoolOp mz(q)` or `something BoolOp var` (`var = mz(q)`)
        if isinstance(condition, ast.BoolOp) and 'values' in condition.__dict__:

            for value in condition.__dict__['values']:
                if self.checkForMeasureResult(value):
                    self.hasMidCircuitMeasures = True
                    return
                if isinstance(value,
                              ast.Compare) and self.checkForMeasureResult(
                                  value.left):
                    self.hasMidCircuitMeasures = True
                    return


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

        self.generic_visit(node)

    def visit_Call(self, node):
        """
        Here we look for function calls within this kernel. We will 
        add these to dependent kernels dictionary. We will also look for 
        kernels that are passed to control and adjoint.
        """
        if hasattr(node, 'func'):
            if isinstance(node.func,
                          ast.Name) and node.func.id in globalAstRegistry:
                self.depKernels[node.func.id] = globalAstRegistry[node.func.id]
            elif isinstance(node.func, ast.Attribute):
                if hasattr(
                        node.func.value, 'id'
                ) and node.func.value.id == 'cudaq' and node.func.attr == 'kernel':
                    return
                # May need to somehow import a library kernel, find
                # all module names in a mod1.mod2.mod3.function type call
                moduleNames = []
                value = node.func.value
                while isinstance(value, ast.Attribute):
                    moduleNames.append(value.attr)
                    value = value.value
                    if isinstance(value, ast.Name):
                        moduleNames.append(value.id)
                        break

                if all(x in moduleNames for x in ['cudaq', 'dbg', 'ast']):
                    return

                if len(moduleNames):
                    moduleNames.reverse()
                    # This will throw if the function / module is invalid
                    m = importlib.import_module('.'.join(moduleNames))
                    getattr(m, node.func.attr)
                    name = node.func.attr
                    if name not in globalAstRegistry:
                        raise RuntimeError(
                            f"{name} is not a valid kernel to call.")

                    self.depKernels[name] = globalAstRegistry[name]

                elif hasattr(node.func,
                             'attr') and node.func.attr in globalAstRegistry:
                    self.depKernels[node.func.attr] = globalAstRegistry[
                        node.func.attr]
                elif node.func.value.id == 'cudaq' and node.func.attr in [
                        'control', 'adjoint'
                ] and node.args[0].id in globalAstRegistry:
                    self.depKernels[node.args[0].id] = globalAstRegistry[
                        node.args[0].id]


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
