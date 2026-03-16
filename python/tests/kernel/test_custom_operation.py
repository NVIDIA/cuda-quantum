# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest
import numpy as np
import cudaq

swap_matrix = np.array([1, 0, 0, 0,
                        0, 0, 1, 0,
                        0, 1, 0, 0,
                        0, 0, 0, 1], dtype=complex)


@pytest.fixture(autouse=True)
def clear_registries():
    cudaq.register_operation("custom_swap", swap_matrix)
    yield
    cudaq.__clearKernelRegistries()


def test_individual_qubit_refs():
    """custom_swap(q0, q1)"""

    @cudaq.kernel
    def kernel():
        qvec = cudaq.qvector(2)
        x(qvec[0])
        custom_swap(qvec[0], qvec[1])

    counts = cudaq.sample(kernel)
    assert counts.most_probable() == "01"


def test_qvector_direct():
    """custom_swap(qvec)"""

    @cudaq.kernel
    def kernel():
        qvec = cudaq.qvector(2)
        x(qvec[0])
        custom_swap(qvec)

    counts = cudaq.sample(kernel)
    assert counts.most_probable() == "01"


def test_starred_qvector():
    """custom_swap(*qvec)"""

    @cudaq.kernel
    def kernel():
        qvec = cudaq.qvector(2)
        x(qvec[0])
        custom_swap(*qvec)

    counts = cudaq.sample(kernel)
    assert counts.most_probable() == "01"