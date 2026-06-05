# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import sys

import pytest

import cudaq

skip_kernel_draw_on_macos = pytest.mark.skipif(
    sys.platform == 'darwin',
    reason='@cudaq.kernel draw needs LLVM host target (see set_data_layout)')


@pytest.fixture(autouse=True)
def reset_target():
    cudaq.reset_target()
    yield
    cudaq.__clearKernelRegistries()


def test_angular_encode_host_raises():
    with pytest.raises(RuntimeError, match="only in CUDA-Q kernels"):
        cudaq.contrib.angular_encode(None, [])


@skip_kernel_draw_on_macos
def test_angular_encode_draw_ry():

    @cudaq.kernel
    def kernel(angles: list[float]):
        q = cudaq.qvector(3)
        cudaq.contrib.angular_encode(q, angles, rotation='Y')

    drawn = cudaq.draw(kernel, [0.1, 0.2, 0.3])
    expected = """     ╭─────────╮
q0 : ┤ ry(0.1) ├
     ├─────────┤
q1 : ┤ ry(0.2) ├
     ├─────────┤
q2 : ┤ ry(0.3) ├
     ╰─────────╯
"""
    assert drawn == expected


@skip_kernel_draw_on_macos
def test_angular_encode_draw_rx():

    @cudaq.kernel
    def kernel(angles: list[float]):
        q = cudaq.qvector(2)
        cudaq.contrib.angular_encode(q, angles, rotation='X')

    drawn = cudaq.draw(kernel, [0.5, 1.0])
    assert 'rx(0.5)' in drawn
    assert 'rx(1)' in drawn or 'rx(1.0)' in drawn


def test_angular_encode_mismatched_static_angles():
    with pytest.raises(RuntimeError) as err:

        @cudaq.kernel
        def kernel():
            q = cudaq.qvector(2)
            cudaq.contrib.angular_encode(q, [0.1, 0.2, 0.3], rotation='Y')

        kernel.compile()

    assert 'number of angles must match' in repr(err.value)
