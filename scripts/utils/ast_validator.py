# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import ast

class ASTValidator(ast.NodeVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.allowed_nodes = (
            # This list was fetched from here
            # https://docs.python.org/3/library/ast.html
            ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
            ast.Return, ast.Delete, ast.Assign, ast.AugAssign, ast.AnnAssign,
            ast.For, ast.AsyncFor, ast.While, ast.If, ast.With, ast.AsyncWith,
            ast.Raise, ast.Try, ast.Assert, ast.Import, ast.ImportFrom,
            ast.Global, ast.Nonlocal, ast.Expr, ast.Pass, ast.Break,
            ast.Continue, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Lambda,
            ast.IfExp, ast.Dict, ast.Set, ast.ListComp, ast.SetComp,
            ast.DictComp, ast.GeneratorExp, ast.Await, ast.Yield, ast.YieldFrom,
            ast.Compare, ast.Call, ast.FormattedValue, ast.JoinedStr, ast.Constant,
            ast.Attribute, ast.Subscript, ast.Starred, ast.Name, ast.List,
            ast.Tuple, ast.Slice, ast.ExtSlice, ast.Index, ast.keyword, ast.arg,
            ast.arguments, ast.Lambda, ast.alias, ast.withitem, ast.comprehension,
            ast.ExceptHandler
        )
        self.errors = []

    def visit(self, node):
        if isinstance(node, self.allowed_nodes):
            self.generic_visit(node)
        else:
            node_info = ast.dump(node)
            self.errors.append(
                f"Disallowed node type: {type(node).__name__}. Node details: {node_info}."
            )

    def validate(self, node):
        self.visit(node)
        return self.errors == [], self.errors
