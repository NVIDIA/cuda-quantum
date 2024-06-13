# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import re

class RequestValidator:
    def __init__(self) -> None:
        self.validation_checks = re.compile(
            r"\b(eval|exec|compile|open|input|os\.|sys\.|subprocess\.|shutil\.|Popen|system|getattr|setattr|delattr|globals|locals|vars|exit|quit|file|open|read|write|close|unlink|remove|rmdir|mkdir|chmod|chown|chdir|pathlib|tempfile|signal|threading|multiprocessing|socket|ctypes|ffi|pickle|marshal|xml|json|yaml|base64|webbrowser|urllib|requests|http|ftplib|poplib|smtplib|telnetlib|imaplib|nntplib|requests|cgi|random|secrets|hashlib|inspect|ast|imp|resource|crypt|pwd|grp)\b"
        )

    def validate_string(self, source_code: str):
        match = re.search(self.validation_checks, source_code)
        if match:
            print(source_code)
            print(match.group())
            return False, match.group()
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
    
    def validate_json_value(self, value):
        if isinstance(value, str):
            return self.validate_string(value)
        elif isinstance(value, dict):
            return self.validate_namespace(value)
        elif isinstance(value, list):
            for item in value:
                is_valid, match = self.validate_json_value(item)
                if not is_valid:
                    return False, match
        return True, None
    
    def validate_request(self, serialized_code_execution_context: dict) -> tuple[bool, str]:
        try:
            source_code = serialized_code_execution_context['source_code']
            globals_namespace = serialized_code_execution_context['globals']

            is_valid, match = self.validate_string(source_code)
            if not is_valid:
                return False, f"Invalid source code: '{match}'"
            is_valid, match = self.validate_namespace(globals_namespace)
            if not is_valid:
                return False, f"Invalid namespace: '{match}'"
        except Exception as e:
            return False, "Invalid request field format."
        
        return True, ""