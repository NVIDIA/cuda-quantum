# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""EDEM bindings, in development toward a stable C ABI.

The interfaces and binary compatibility are subject to change during development.
"""

from pathlib import Path

from . import _core

__version__ = "@CUDAQ_EDEM_VERSION@"


def get_cmake_dir() -> str:
    """Return the installed cudaq-edem CMake package directory."""
    return str((Path(__file__).resolve().parent /
                "@CUDAQ_EDEM_PYTHON_CMAKE_DIR@").resolve())
