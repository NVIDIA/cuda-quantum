# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import math
import cudaq
from .kernels import bell, rotation_kernel


def test_ptsbe_broadcast_bit_flip_noise():
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("ry", cudaq.BitFlipChannel(0.1))
    shots = 2000
    angles = [0.0, math.pi]
    results = cudaq.ptsbe.sample(
        rotation_kernel,
        angles,
        noise_model=noise,
        shots_count=shots,
    )
    assert len(results) == 2
    for r in results:
        assert sum(r.count(bs) for bs in r) == shots

    p0 = results[0].count("0") / shots
    p1_flipped = results[0].count("1") / shots
    assert 0.85 <= p0 <= 0.95
    assert 0.05 <= p1_flipped <= 0.15

    p1 = results[1].count("1") / shots
    p0_flipped = results[1].count("0") / shots
    assert 0.85 <= p1 <= 0.95
    assert 0.05 <= p0_flipped <= 0.15


def test_ptsbe_sample_async_returns_future_like():
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(0.1),
                                num_controls=1)
    future = cudaq.ptsbe.sample_async(
        bell, noise_model=noise, shots_count=10
    )
    assert hasattr(future, "get")
    result = future.get()
    assert sum(result.count(bs) for bs in result) == 10


def test_ptsbe_broadcast_single_argument():
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("ry", cudaq.DepolarizationChannel(0.01))
    results = cudaq.ptsbe.sample(
        rotation_kernel,
        [0.0],
        noise_model=noise,
        shots_count=30,
    )
    assert isinstance(results, list)
    assert len(results) == 1
    assert sum(results[0].count(bs) for bs in results[0]) == 30


def test_ptsbe_broadcast_three_angles():
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("ry", cudaq.DepolarizationChannel(0.01))
    angles = [0.0, math.pi / 2, math.pi]
    results = cudaq.ptsbe.sample(
        rotation_kernel,
        angles,
        noise_model=noise,
        shots_count=50,
    )
    assert len(results) == 3
    for r in results:
        assert sum(r.count(bs) for bs in r) == 50


def test_ptsbe_sample_async_get_consumes_future():
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(0.05),
                                num_controls=1)
    future = cudaq.ptsbe.sample_async(
        bell, noise_model=noise, shots_count=15
    )
    r = future.get()
    total = sum(r.count(bs) for bs in r)
    assert total == 15
