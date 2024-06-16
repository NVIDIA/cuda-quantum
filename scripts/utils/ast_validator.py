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
        self.user_defined_functions = set()
        self.user_defined_variables = set()
        # List of allowed modules for function calls
        self.allowed_modules = {
            'cudaq', 'numpy', 'optimizer', 'gradient_strategy', 'List', 'Tuple'
        }
        self.errors = []

    def visit_FunctionDef(self, node):
        # Collect user-defined function names
        self.user_defined_functions.add(node.name)
        self.fetch_user_defined_variables(node)
        self.generic_visit(node)

    def visit_Lambda(self, node):
        self.fetch_user_defined_variables(node)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        # Validate attribute access
        attr_name = self.get_full_name(node)
        if not self.is_allowed_function(attr_name):
            node_info = ast.dump(node)
            self.errors.append(
                f"Disallowed attribute access: {attr_name}. Node details: {node_info}."
            )
        self.generic_visit(node)

    def visit_Assign(self, node):
        # Capture variables instantiated from allowed modules
        if isinstance(node.value, ast.Lambda):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.user_defined_functions.add(target.id)
        if isinstance(node.value, ast.Call):
            func_name = self.get_full_name(node.value.func)
            if any(
                    func_name.startswith(f"{module}.")
                    for module in self.allowed_modules):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.user_defined_variables.add(target.id)
        self.generic_visit(node)

    def visit_Call(self, node):
        # Validate function calls
        func_name = self.get_full_name(node.func)
        if not self.is_allowed_function(func_name):
            node_info = ast.dump(node)
            self.errors.append(
                f"Disallowed function call: {func_name}. Node details: {node_info}."
            )
        self.generic_visit(node)

    def _check_annotation(self, arg):
        # Check if the annotation is a type from allowed modules
        if arg.annotation:
            annotation = self._get_annotation_name(arg.annotation)
            if annotation in self.allowed_modules:
                self.user_defined_variables.add(arg.arg)

    def _get_annotation_name(self, annotation):
        # Extract the full name from an annotation node
        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Attribute):
            return f"{self._get_annotation_name(annotation.value)}.{annotation.attr}"
        elif isinstance(annotation, ast.Subscript):
            return self._get_annotation_name(annotation.value)
        elif isinstance(annotation, ast.Call):
            return self._get_annotation_name(annotation.func)
        return None

    def fetch_user_defined_variables(self, node):
        # Check parameters (lambda) and keyword-only arguments
        for arg in node.args.args + node.args.kwonlyargs:
            if isinstance(arg, ast.arg) and arg.annotation:
                self._check_annotation(arg)

    def get_full_name(self, node):
        # Get the full name of the function being called
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value = self.get_full_name(node.value)
            return f"{value}.{node.attr}"
        elif isinstance(node, ast.Call):
            return self.get_full_name(node.func)
        else:
            return ""

    def is_allowed_function(self, func_name):
        # Check if the function is allowed
        if func_name in self.user_defined_functions:
            return True
        if func_name.split('.')[0] in self.user_defined_variables:
            return True
        return any(
            func_name.startswith(f"{module}.")
            for module in self.allowed_modules)

    def validate(self, node):
        # Validate AST node
        self.visit(node)
        return self.errors == [], self.errors
