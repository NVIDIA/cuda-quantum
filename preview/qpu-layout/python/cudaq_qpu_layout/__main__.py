# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Trace a Quake payload straight from the command line, with no target:

    python3 -m cudaq_qpu_layout payload.mlir --regions 2 --region-size 2
"""

from .sim import main

main()
