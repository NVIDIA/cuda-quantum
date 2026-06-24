# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Regression test for issue #2346:
# A kernel called via an intermediate import path (gates.qft_ops.qft_kernel)
# must resolve even though it is registered under the fully-qualified path
# (algo_lib.gates.qft_ops.qft_kernel).
#
# Package layout (relative to this file):
#   algo_lib/__init__.py              <- from .gates import qft_ops
#   algo_lib/gates/__init__.py
#   algo_lib/gates/qft_ops.py         <- defines @cudaq.kernel def qft_kernel()

import cudaq
import pytest

from algo_lib import gates


@pytest.fixture(autouse=True)
def clear_registries():
    yield
    cudaq.__clearKernelRegistries()


def test_kernel_call_via_partial_module_path():

    @cudaq.kernel
    def test0():
        gates.qft_ops.qft_kernel()

    counts = cudaq.sample(test0)
    assert '0' in counts
