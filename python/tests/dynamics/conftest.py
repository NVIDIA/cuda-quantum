# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""
Safety-net fixture: guarantee target reset after every dynamics test so that
a failed teardown in a per-test fixture cannot leak the dynamics target to
other tests on the same pytest-xdist worker.
"""
import pytest
import cudaq


@pytest.fixture(autouse=True)
def ensure_target_reset():
    yield
    try:
        cudaq.reset_target()
    except Exception:
        pass
