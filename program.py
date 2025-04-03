# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import numpy as np

def assert_close(want, got, tolerance=1.e-5) -> bool:
    return abs(want - got) < tolerance

def check_overlap(entity_bell, entity_x):
    state1 = cudaq.get_state(entity_bell)
    state1.dump()
    state2 = cudaq.get_state(entity_x)
    state2.dump()
    assert assert_close(state1.overlap(state2), 1.0 / np.sqrt(2))


def check_overlap_param(entity):
    num_tests = 10
    for i in range(num_tests):
        angle1 = np.random.rand(
        ) * 2.0 * np.pi  # random angle in [0, 2pi] range
        state1 = cudaq.get_state(entity, angle1)
        print("First angle =", angle1)
        state1.dump()
        angle2 = np.random.rand(
        ) * 2.0 * np.pi  # random angle in [0, 2pi] range
        print("Second angle =", angle2)
        state2 = cudaq.get_state(entity, angle2)
        state2.dump()
        overlap = state1.overlap(state2)
        expected = np.abs(
            np.cos(angle1 / 2) * np.cos(angle2 / 2) +
            np.sin(angle1 / 2) * np.sin(angle2 / 2))
        assert assert_close(overlap, expected)


def test_overlap_param_kernel():

    @cudaq.kernel
    def kernel(theta: float):
        qreg = cudaq.qvector(1)
        rx(theta, qreg[0])

    check_overlap_param(kernel)

cudaq.set_target("remote-mqpu", auto_launch="1")
test_overlap_param_kernel()