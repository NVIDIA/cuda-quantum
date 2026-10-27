# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import numpy as np
import pytest

from cudaq.runtime.utils import __createArgumentSet


def test_lists_stay_paired():
    assert __createArgumentSet([1.0, 2.0], [3.0, 4.0]) == [(1.0, 3.0),
                                                            (2.0, 4.0)]


def test_numpy_rows_stay_paired():
    got = __createArgumentSet(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    assert got == [(1.0, 3.0), (2.0, 4.0)]


def test_scalar_is_not_replaced_with_zero():
    with pytest.raises(RuntimeError, match="must be a list"):
        __createArgumentSet([1.0, 2.0, 3.0], 0.5)


def test_numpy_row_with_a_scalar_is_not_replaced_with_zero():
    with pytest.raises(RuntimeError, match="must be a list"):
        __createArgumentSet(np.array([1.0, 2.0, 3.0]), 0.5)
