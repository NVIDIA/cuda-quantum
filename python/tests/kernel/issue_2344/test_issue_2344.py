# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Regression test for issue #2344:
# A kernel defined in another module (hidden_module.spooky_kernel) must NOT be
# callable without its module qualifier. The call `spooky_kernel()` inside
# test0 should raise a NameError because spooky_kernel is not in scope.
#
# Package layout (relative to this file):
#   hidden_module/__init__.py    <- defines @cudaq.kernel def spooky_kernel()

import cudaq
import pytest

import hidden_module  # noqa: F401 — imported for kernel registration side effect


@pytest.fixture(autouse=True)
def clear_registries():
    yield
    cudaq.__clearKernelRegistries()


def test_unqualified_cross_module_kernel_call_raises():

    @cudaq.kernel
    def test0():
        spooky_kernel()  # not imported — should not resolve

    with pytest.raises((NameError, RuntimeError)):
        cudaq.sample(test0)
