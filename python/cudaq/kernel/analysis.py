# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import ast
import inspect
import textwrap
from typing import Optional, Type

from .utils import get_function_source_or_raise


class FunctionDefVisitor(ast.NodeVisitor):
    """
    This visitor will visit the function definition of `kernel_name` and report 
    type annotations and whether the function has a return statement.
    """

    arg_annotations: list[(str, Type)]
    return_annotation: Optional[Type] = None
    has_return_statement: bool = False
    found: bool = False

    def __init__(self, kernel_name: str):
        self.kernel_name: str = kernel_name
        self.arg_annotations = []

    def visit_FunctionDef(self, node):
        if node.name == self.kernel_name:
            self.found = True
            self.arg_annotations = [
                (arg.arg, arg.annotation) for arg in node.args.args
            ]
            self.return_annotation = node.returns
            self.has_return_statement = any(
                isinstance(n, ast.Return) and n.value != None
                for n in node.body)

    def generic_visit(self, node):
        if self.found:
            # skip traversing the rest of the AST once found
            return
        super().generic_visit(node)


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

        src, _ = get_function_source_or_raise(func_obj)
        tree = ast.parse(src)
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
            if this_func_obj is None:
                continue
            src, _ = get_function_source_or_raise(this_func_obj)

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

        def literal_int(node, known):
            if isinstance(node, ast.Constant) and isinstance(
                    node.value, int) and not isinstance(node.value, bool):
                return node.value
            if isinstance(node, ast.UnaryOp) and isinstance(
                    node.op, ast.USub):
                v = literal_int(node.operand, known)
                return None if v is None else -v
            if isinstance(node, ast.Name):
                return known.get(node.id)
            return None

        compare_ops = {
            ast.Lt: lambda a, b: a < b,
            ast.LtE: lambda a, b: a <= b,
            ast.Gt: lambda a, b: a > b,
            ast.GtE: lambda a, b: a >= b,
            ast.Eq: lambda a, b: a == b,
            ast.NotEq: lambda a, b: a != b,
        }

        def while_test_is_statically_true(test, known):
            if isinstance(test, ast.Constant):
                return bool(test.value)
            if isinstance(test, ast.Compare) and len(test.ops) == 1 and len(
                    test.comparators) == 1 and type(
                        test.ops[0]) in compare_ops:
                left = literal_int(test.left, known)
                right = literal_int(test.comparators[0], known)
                if left is None or right is None:
                    return False
                return compare_ops[type(test.ops[0])](left, right)
            return False

        def loop_provably_runs(stmt, known):
            """A `for`/`while` loop's `orelse` always runs when the loop
            finishes without `break`, including zero iterations, so it needs
            no special handling. The body is only trustworthy for "all paths
            return" when the loop is statically guaranteed to execute at
            least once -- otherwise control can fall through the loop
            without ever taking it, reaching the end of the function with no
            return and no error.

            This check is intentionally narrow. It only tightens the two
            shapes that can be proven unsound from the syntax alone: a
            `while` whose condition is not a statically-true comparison, and
            a `for` over `range(...)` whose bound is not a compile-time
            constant. Iteration over any other iterable (a plain variable, a
            list literal, `enumerate(...)`, a qvector, ...) keeps the
            pre-existing lenient treatment, since it is a compile-time
            unknown either way and this file's own test suite relies on
            that leniency for unrelated reasons in exactly that case.
            """
            if isinstance(stmt, ast.While):
                return while_test_is_statically_true(stmt.test, known)
            it = stmt.iter
            if not (isinstance(it, ast.Call) and
                    isinstance(it.func, ast.Name) and
                    it.func.id == 'range' and 1 <= len(it.args) <= 3):
                return True
            bounds = [literal_int(a, known) for a in it.args]
            if any(b is None for b in bounds):
                return False
            if len(bounds) == 1:
                start, stop, step = 0, bounds[0], 1
            elif len(bounds) == 2:
                start, stop, step = bounds[0], bounds[1], 1
            else:
                start, stop, step = bounds
            if step == 0:
                return False
            span = stop - start
            return span > 0 if step > 0 else span < 0

        def all_paths_return(stmts):
            # Tracks variables assigned a compile-time-constant int earlier in
            # this same statement list, so a following `while <var> <op>
            # <literal>:` can be recognized as provably entered -- the
            # established idiom throughout this file's own test suite (e.g.
            # `i = 0` then `while i < 6:`).
            known_ints = {}
            for stmt in stmts:
                if isinstance(stmt, ast.Assign) and len(
                        stmt.targets) == 1 and isinstance(
                            stmt.targets[0], ast.Name):
                    v = literal_int(stmt.value, known_ints)
                    if v is None:
                        known_ints.pop(stmt.targets[0].id, None)
                    else:
                        known_ints[stmt.targets[0].id] = v

                if isinstance(stmt, ast.Return):
                    return True

                if isinstance(stmt, ast.If):
                    if all_paths_return(stmt.body) and all_paths_return(
                            stmt.orelse):
                        return True

                if isinstance(stmt, (ast.For, ast.While)):
                    if (loop_provably_runs(stmt, known_ints) and
                            all_paths_return(stmt.body)) or all_paths_return(
                                stmt.orelse):
                        return True

            return False

        if not all_paths_return(node.body):
            self.bridge.emitFatalError(
                'cudaq.kernel functions with return type annotations must have a return statement.',
                node)

        self.generic_visit(node)
