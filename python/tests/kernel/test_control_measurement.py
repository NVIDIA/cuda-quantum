# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                        #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest

import cudaq


@pytest.fixture(autouse=True)
def do_something():
    cudaq.set_target("qpp-cpu")
    yield
    cudaq.reset_target()
    cudaq.__clearKernelRegistries()


def test_control_of_measuring_kernel_is_rejected():
    """cudaq.control over a kernel that measures must be a compile error, not
    a silently wrong circuit."""

    @cudaq.kernel
    def erase(q: cudaq.qview):
        h(q[2])
        fired = mz(q[2])
        if fired:
            z.ctrl(q[0], q[1])
            x(q[2])

    @cudaq.kernel
    def controlled():
        c = cudaq.qvector(1)
        q = cudaq.qvector(3)
        h(q[0])
        cudaq.control(erase, c[0], q)

    with pytest.raises(RuntimeError, match="could not autogenerate"):
        cudaq.get_state(controlled)


def test_adjoint_control_of_measuring_kernel_is_rejected():

    @cudaq.kernel
    def measure_only(q: cudaq.qview):
        x(q[0])
        mz(q[0])

    @cudaq.kernel
    def controlled():
        c = cudaq.qvector(1)
        q = cudaq.qvector(2)
        cudaq.adjoint(measure_only, q)
        cudaq.control(measure_only, c[0], q)

    with pytest.raises(RuntimeError, match="could not autogenerate"):
        cudaq.get_state(controlled)
