# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import os
import pytest


@pytest.fixture(autouse=True)
def restore_default_level():
    yield
    cudaq.reset_compile_options()


@cudaq.kernel
def bell():
    q = cudaq.qvector(2)
    h(q[0])
    x.ctrl(q[0], q[1])


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_valid_levels(level):
    cudaq.set_compile_options(optimization_level=level)
    counts = cudaq.sample(bell, shots_count=10)
    assert set(bits for bits, _ in counts.items()) <= {"00", "11"}


@pytest.mark.parametrize("level", [-1, 4])
def test_invalid_levels(level):
    with pytest.raises(ValueError):
        cudaq.set_compile_options(optimization_level=level)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
