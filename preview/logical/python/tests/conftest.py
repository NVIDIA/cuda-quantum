# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Load the build-tree cudaq.logical package when the wheel is not installed."""

from __future__ import annotations

try:
    import _cudaq_logical_devpath  # noqa: F401
except ImportError:
    pass
import cudaq.logical  # noqa: F401
