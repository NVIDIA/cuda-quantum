# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import pytest


@pytest.fixture(autouse=True)
def use_qpp_target():
    cudaq.set_target('qpp-cpu')
    yield
    cudaq.reset_target()


@pytest.mark.parametrize('index, expected', [(0, '10'), (1, '01')])
def test_qvector_captured_index(index, expected):
    indices = [index]

    @cudaq.kernel
    def flip():
        q = cudaq.qvector(2)
        x(q[indices[0]])

    counts = cudaq.sample(flip, shots_count=10)
    assert counts[expected] == 10


@pytest.mark.parametrize('size, expected', [(3, '111'), (4, '1011'),
                                            (5, '10011'), (6, '100011')])
def test_qvector_negative_modulo_index_wraps_around(size, expected):
    # `(i - 1) % size` must select the previous qubit around the ring for
    # every ring size. With an unsigned remainder, `-1 % size` was `0` for
    # every size that is not a power of two, so the first control targeted
    # itself.

    @cudaq.kernel
    def ring(size: int):
        q = cudaq.qvector(size)
        x(q[0])
        for i in range(size):
            x.ctrl(q[i], q[(i - 1) % size])

    counts = cudaq.sample(ring, size, shots_count=10)
    assert counts[expected] == 10


@pytest.mark.skip_arm64_jit
@pytest.mark.parametrize(
    'index',
    [2, 2**31, 2**61 - 1, 2**61, 2**61 + 1, 2**62, 2**62 + 1, 2**63 - 1])
def test_qvector_captured_index_out_of_range(index):
    indices = [index]

    @cudaq.kernel
    def flip():
        q = cudaq.qvector(2)
        x(q[indices[0]])

    with pytest.raises(
            RuntimeError,
            match=rf'Provided index \[{index}\] >= array size \[2\]'):
        cudaq.sample(flip, shots_count=10)
