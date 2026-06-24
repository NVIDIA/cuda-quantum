# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #


import platform
import pytest

def skip_if_not_arch(required_arch):
    """
    Skips the test if the current architecture does not match the required architecture.
    """
    if platform.machine() != required_arch:
        pytest.skip(f"Test requires architecture: {required_arch}")

def skip_if_not_arch_in(allowed_archs):
    """
    Skips the test if the current architecture is not in the list of allowed architectures.
    """
    if platform.machine() not in allowed_archs:
        pytest.skip(f"Test requires one of: {allowed_archs}")
