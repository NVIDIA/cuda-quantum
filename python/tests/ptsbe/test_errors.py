# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import pytest
import cudaq
from .kernels import bell, rotation_kernel


def test_ptsbe_zero_shots_raises_or_empty(depol_noise):
    result = cudaq.ptsbe.sample(bell, noise_model=depol_noise, shots_count=0)
    with pytest.raises(RuntimeError, match="no results"):
        sum(result.count(bs) for bs in result)


def test_ptsbe_wrong_kernel_args_raises(depol_noise):
    with pytest.raises(RuntimeError, match="Invalid number of arguments"):
        cudaq.ptsbe.sample(bell, 1.0, 2.0, noise_model=depol_noise)


def test_ptsbe_broadcast_empty_args_returns_empty_list():
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("ry", cudaq.DepolarizationChannel(0.01))
    results = cudaq.ptsbe.sample(
        rotation_kernel,
        [],
        noise_model=noise,
        shots_count=10,
    )
    assert results == []


def test_ptsbe_sample_non_integer_shots_raises(depol_noise):
    with pytest.raises((RuntimeError, TypeError)):
        cudaq.ptsbe.sample(
            bell, noise_model=depol_noise, shots_count=10.5
        )
