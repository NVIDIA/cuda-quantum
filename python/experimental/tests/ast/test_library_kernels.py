# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import os

import pytest
import numpy as np

import cudaq

# [RFC]:
# FIXME: Why does having this here instead of inside the respective test cause issues?
# cudaq.enable_jit()
# from cudaq.lib import fermionic_swap, givens


def test_fswap_lib_kernel():
    cudaq.enable_jit()
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
    assert np.isclose(np.abs(ss_01[1] - (-1j * np.exp(1j * angle / 2.0) * si)),
                      0.0,
                      atol=1e-3)
    assert np.isclose(np.abs(ss_01[2] - (np.exp(1j * angle / 2.0) * c)),
                      0.0,
                      atol=1e-3)


def test_givens_lib_kernel():
    cudaq.enable_jit()
    from cudaq.lib import givens

    angle = 0.2
    c = np.cos(angle)
    si = np.sin(angle)

    @cudaq.kernel
    def baz(angle: float):
        q = cudaq.qlist(2)
        x(q[0])
        givens(angle, q[0], q[1])

    print(baz)
    ss_01 = cudaq.get_state(baz, angle)
    print(ss_01)
    assert np.isclose(ss_01[1], -si, 1e-3)
    assert np.isclose(ss_01[2], c, 1e-3)
