# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from utils.ast_validator import ASTValidator
import ast
import re


class RequestValidator:

    def __init__(self) -> None:
        self.ast_validator = ASTValidator()

    def validate_ast(self, source_code: str):
        try:
            parsed_ast = ast.parse(source_code)
            is_valid, errors = self.ast_validator.validate(parsed_ast)
            if not is_valid:
                return False, errors
            return True, None
        except SyntaxError as e:
            return False, [f"Syntax error in code: {e.msg}"]
        except Exception as e:
            return False, [str(e)]

    def validate_json_value(self, value):
        if isinstance(value, str):
            return self.validate_ast(value)
        elif isinstance(value, dict):
            return self.validate_namespace(value)
        elif isinstance(value, list):
            for item in value:
                is_valid, match = self.validate_json_value(item)
                if not is_valid:
                    return False, match
        return True, None

    def validate_namespace(self, namespace_dict: dict):
        for key, value in namespace_dict.items():
            is_valid, match = self.validate_json_value(key)
            if not is_valid:
                return False, match
            is_valid, match = self.validate_json_value(value)
            if not is_valid:
                return False, match
        return True, None

    def validate_request(
            self, serialized_code_execution_context: dict) -> tuple[bool, str]:
        try:
            source_code = serialized_code_execution_context['source_code']
            global_namespace = serialized_code_execution_context[
                'scoped_var_dict']

            is_valid, errors = self.validate_ast(source_code)
            if not is_valid:
                return False, f"Invalid source code: '{errors}'"
            # Commenting out as I don't have a much solid use case to validate namespace
            # as we are already validating the source code
            # is_valid, match = self.validate_namespace(global_namespace)
            # if not is_valid:
            #     return False, f"Invalid namespace: '{match}'"
        except KeyError as e:
            return False, f"Missing key in request: {str(e)}"
        except Exception as e:
            return False, str(e)

        return True, ""
