# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest
import numpy as np

import cudaq


@pytest.fixture(autouse=True)
def do_something():
    yield
    cudaq.__clearKernelRegistries()


def test_internal_library_kernels():
    from cudaq.lib import fermionic_swap

    angle = 0.2
    c = np.cos(angle / 2)
    si = np.sin(angle / 2)

    @cudaq.kernel
    def bar(angle: float):
        q = cudaq.qvector(2)
        x(q[0])
        fermionic_swap(angle, q[0], q[1])

    ss_01 = cudaq.StateMemoryView(cudaq.get_state(bar, angle))
    val1 = np.abs(ss_01[1] - (-1j * np.exp(1j * angle / 2.0) * si))
    val2 = np.abs(ss_01[2] - (np.exp(1j * angle / 2.0) * c))
    assert np.isclose(val1, 0.0, atol=1e-6)
    assert np.isclose(val2, 0.0, atol=1e-6)

    # Can also use the full module import path
    @cudaq.kernel
    def baz(angle: float):
        q = cudaq.qvector(2)
        x(q[0])
        cudaq.lib.fermionic_swap(angle, q[0], q[1])

    ss_01 = cudaq.StateMemoryView(cudaq.get_state(baz, angle))
    val1 = np.abs(ss_01[1] - (-1j * np.exp(1j * angle / 2.0) * si))
    val2 = np.abs(ss_01[2] - (np.exp(1j * angle / 2.0) * c))
    assert np.isclose(val1, 0.0, atol=1e-6)
    assert np.isclose(val2, 0.0, atol=1e-6)

    from cudaq.lib import givens

    angle = 0.2
    c = np.cos(angle)
    si = np.sin(angle)

    @cudaq.kernel
    def baz(angle: float):
        q = cudaq.qvector(2)
        x(q[0])
        givens(angle, q[0], q[1])

    ss_01 = cudaq.StateMemoryView(cudaq.get_state(baz, angle))
    ss_01.dump()
    assert np.isclose(ss_01[1], -si, 1e-3)
    assert np.isclose(ss_01[2], c, 1e-3)
