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

import os
import sys
import platform
import pytest

_tests_dir = os.path.dirname(os.path.abspath(__file__))
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "skip_macos_arm64_jit: skip on macOS ARM64 due to JIT exception handling bug (llvm-project#49036)"
    )
    config.addinivalue_line(
        "markers",
        "skip_arm64_jit: skip on any ARM64 platform (macOS or Linux aarch64) due to "
        "JIT exception handling bug where C++ exceptions thrown through LLVM-compiled "
        "frames terminate the process instead of propagating to Python (llvm-project#49036)"
    )


def pytest_collection_modifyitems(config, items):
    """Apply skip markers for ARM64 JIT exception-handling limitations."""
    # platform.machine() == 'arm64'   on macOS ARM64
    # platform.machine() == 'aarch64' on Linux ARM64
    is_arm64 = platform.machine() in ('arm64', 'aarch64')
    is_darwin = sys.platform == 'darwin'

    for item in items:
        # Original behaviour unchanged: skip_macos_arm64_jit fires only on macOS ARM64.
        if is_darwin and is_arm64:
            if item.get_closest_marker('skip_macos_arm64_jit'):
                item.add_marker(
                    pytest.mark.skip(
                        reason=
                        "JIT exception handling broken on macOS ARM64 (llvm-project#49036)"
                    ))

        # New: skip_arm64_jit fires on ANY ARM64 (macOS or Linux aarch64).
        if is_arm64:
            if item.get_closest_marker('skip_arm64_jit'):
                item.add_marker(
                    pytest.mark.skip(
                        reason=
                        "JIT exception handling broken on ARM64 (llvm-project#49036)"
                    ))
