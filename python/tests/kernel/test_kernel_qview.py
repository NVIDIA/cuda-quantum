# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import pytest


@pytest.fixture(autouse=True)
def setup_test_environment():
    cudaq.__clearKernelRegistries()
    yield


def test_qview_zero_length():

    @cudaq.kernel
    def kernel1(N: int):
        q = cudaq.qvector(N + N)
        qv = q[N:]
        x(qv[0:1])

    counts = cudaq.sample(kernel1, 2)
    print(counts)
    assert '0010' in counts


def test_qview_non_zero_length():

    @cudaq.kernel
    def kernel1(N: int):
        q = cudaq.qvector(N + N)
        qv = q[N:]
        x(qv[0:2])

    counts = cudaq.sample(kernel1, 2)
    print(counts)
    assert '0011' in counts
