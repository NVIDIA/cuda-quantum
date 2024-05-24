# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import numpy as np


def test_kernel_complex_capture_f64():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    c = np.array([1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)], dtype=complex)
    state = cudaq.State.from_data(c)

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(state)

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts

test_kernel_complex_capture_f64()
