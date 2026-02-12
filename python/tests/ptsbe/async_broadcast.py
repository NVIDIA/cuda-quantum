# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         #
# All rights reserved.                                                        #
#                                                                             #
# This source code and the accompanying materials are made available under   #
# the terms of the Apache License 2.0 which accompanies this distribution.   #
# ============================================================================ #
import math
import pytest
import cudaq
from .kernels import bell, rotation_kernel


def test_ptsbe_sample_async():
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("h", cudaq.DepolarizationChannel(0.01))
    shots = 200
    future = cudaq.ptsbe.sample_async(bell,
                                      noise_model=noise,
                                      shots_count=shots)
    result = future.get()
    assert isinstance(result, cudaq.SampleResult)
    total = sum(result.count(bs) for bs in result)
    assert total == shots
    bell_counts = result.count("00") + result.count("11")
    assert bell_counts > shots * 0.8


def test_ptsbe_broadcast():
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("ry", cudaq.DepolarizationChannel(0.01))
    shots = 200
    angles = [0.0, math.pi]
    results = cudaq.ptsbe.sample(
        rotation_kernel,
        angles,
        noise_model=noise,
        shots_count=shots,
    )
    assert isinstance(results, list)
    assert len(results) == 2
    for r in results:
        total = sum(r.count(bs) for bs in r)
        assert total == shots
    assert results[0].count("0") > shots * 0.8
    assert results[1].count("1") > shots * 0.8
