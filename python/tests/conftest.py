# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""
Shared pytest fixtures and markers for CUDA-Q Python tests.
"""

import sys
import platform
import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "skip_macos_arm64_jit: skip on macOS ARM64 due to JIT exception handling bug (llvm-project#49036)"
    )


def pytest_collection_modifyitems(config, items):
    """Apply skip marker to tests marked with skip_macos_arm64_jit on macOS ARM64."""
    if sys.platform == 'darwin' and platform.machine() == 'arm64':
        skip_marker = pytest.mark.skip(
            reason=
            "JIT exception handling broken on macOS ARM64 (llvm-project#49036)")
        for item in items:
            if item.get_closest_marker('skip_macos_arm64_jit'):
                item.add_marker(skip_marker)
