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
        self.unsafe_patterns = re.compile(
            # r"\b(__import__|eval|exec|compile|open|input|os\.|sys\.|subprocess\.|shutil\.|Popen|system|getattr|setattr|delattr|globals|locals|vars|exit|quit|file|open|read|write|close|unlink|remove|rmdir|mkdir|chmod|chown|chdir|pathlib|tempfile|signal|threading|multiprocessing|socket|ctypes|ffi|pickle|marshal|builtins|xml|json|yaml|base64|webbrowser|urllib|requests|http|ftplib|poplib|smtplib|telnetlib|imaplib|nntplib|requests|cgi|random|secrets|hashlib|inspect|ast|imp|resource|crypt|pwd|grp)\b"
            # Check with the security team to verify the list of unsafe patterns
            r"\b(exec)"
        )

    def validate_string(self, source_code: str) -> bool:
        return not re.search(self.unsafe_patterns, source_code)
    
    def validate_namespace(self, namespace_dict: dict) -> bool:
        for key, value in namespace_dict.items():
            if not self.validate_json_value(key) or not self.validate_json_value(value):
                return False
        return True
    
    def validate_json_value(self, value):
        if isinstance(value, str):
            return self.validate_string(value)
        elif isinstance(value, dict):
            return self.validate_namespace(value)
        elif isinstance(value, list):
            return all(self.validate_json_value(item) for item in value)
        return True
    
    def validate_request(self, serialized_code_execution_context: dict) -> tuple[bool, str]:
        try:
            source_code = serialized_code_execution_context['source_code']
            globals_namespace = serialized_code_execution_context['globals']

            if not self.validate_string(source_code):
                return False, "Invalid source code."
            if not self.validate_namespace(globals_namespace):
                return False, "Invalid namespace."
        except Exception as e:
            return False, "Invalid request field format."
        
        return True, ""