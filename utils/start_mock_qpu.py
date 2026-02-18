#!/usr/bin/env python3

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import argparse

from mock_qpu import start_server, all_backend_names

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Start one of the available mock QPU backend server.")
    parser.add_argument("backend",
                        type=str,
                        help="backend name",
                        choices=all_backend_names())
    args = parser.parse_args()
    backend = args.backend.lower()

    start_server(backend)
