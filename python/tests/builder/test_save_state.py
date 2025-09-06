# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os

import pytest
import numpy as np

import cudaq

@pytest.mark.parametrize('target', ['density-matrix-cpu', 'stim'])
def test_save_state_builtin(target: str):
    cudaq.set_target(target)

    noise = cudaq.NoiseModel()

    @cudaq.kernel
    def bell_depol2(d: float, flag: bool):
        q, r = cudaq.qubit(), cudaq.qubit()
        h(q)
        cudaq.save_state()
       
        x.ctrl(q, r) 
        cudaq.save_state()

        if flag:
            cudaq.apply_noise(cudaq.Depolarization2, d, q, r)
        else:
            cudaq.apply_noise(cudaq.Depolarization2, [d], q, r)

    counts = cudaq.sample(bell_depol2, 0.2, True, noise_model=noise)
    assert len(counts) == 4
    print(counts)

