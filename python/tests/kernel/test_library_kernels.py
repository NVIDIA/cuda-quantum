# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import os

import pytest
import numpy as np

import cudaq

@pytest.fixture(autouse=True)
def do_something():
    if os.getenv("CUDAQ_PYTEST_EAGER_MODE") == 'OFF':
        cudaq.enable_jit()
    yield
    
    if cudaq.jit_enabled(): cudaq.__clearKernelRegistries()
    cudaq.disable_jit()

def test_internal_library_kernels():
    from cudaq.lib import fermionic_swap

    angle = 0.2
    c = np.cos(angle / 2)
    si = np.sin(angle / 2)

    @cudaq.kernel
    def bar(angle: float):
        q = cudaq.qlist(2)
        x(q[0])
        fermionic_swap(angle, q[0], q[1])

    ss_01 = cudaq.get_state(bar, angle)
    assert np.isclose(np.abs(ss_01[2] - (-1j * np.exp(1j * angle / 2.0) * si)),
                      0.0,
                      atol=1e-3)
    assert np.isclose(np.abs(ss_01[1] - (np.exp(1j * angle / 2.0) * c)),
                      0.0,
                      atol=1e-3)

    # Can also use the full module import path
    @cudaq.kernel
    def baz(angle: float):
        q = cudaq.qlist(2)
        x(q[0])
        cudaq.lib.fermionic_swap(angle, q[0], q[1])

    ss_01 = cudaq.get_state(baz, angle)
    assert np.isclose(np.abs(ss_01[2] - (-1j * np.exp(1j * angle / 2.0) * si)),
                      0.0,
                      atol=1e-3)
    assert np.isclose(np.abs(ss_01[1] - (np.exp(1j * angle / 2.0) * c)),
                      0.0,
                      atol=1e-3)


    from cudaq.lib import givens

    angle = 0.2
    c = np.cos(angle)
    si = np.sin(angle)

    # PYTEST Error, baz fails
    @cudaq.kernel
    def baz2(angle: float):
        q = cudaq.qlist(2)
        x(q[0])
        givens(angle, q[0], q[1])

    ss_01 = cudaq.get_state(baz2, angle)
    print(ss_01)
    assert np.isclose(ss_01[1], c, 1e-3)
    assert np.isclose(ss_01[2], -si, 1e-3)
