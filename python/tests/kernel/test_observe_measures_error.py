# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import pytest
from cudaq import spin


def test_observe_kernel_with_measures_fails():
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(1)
    kernel.h(q[0])
    kernel.mz(q[0])

    with pytest.raises(
            RuntimeError,
            match=r"kernels passed to observe cannot have measurements specified"
    ):
        cudaq.observe(kernel, spin.z(0), shots_count=100)
