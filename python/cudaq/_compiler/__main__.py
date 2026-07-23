# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Thin ``python3 -m cudaq._compiler`` entry point.

The command line is a convenience wrapper over the public API. The
API is the product. All the logic lives in :mod:`optimization_cli`.
"""

import sys

from .optimization_cli import main

if __name__ == "__main__":
    sys.exit(main())
