# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import json
import sys
from datetime import datetime


def write_a_mock_tokens_file(tokens_file_path):
    tokens_data = {
        "access_token": "good_access_token",
    }

    json_str = json.dumps(tokens_data)
    with open(tokens_file_path, "w") as f:
        f.write(json_str)


if __name__ == "__main__":
    tokens_file_path = sys.argv[1]
    write_a_mock_tokens_file(tokens_file_path)
