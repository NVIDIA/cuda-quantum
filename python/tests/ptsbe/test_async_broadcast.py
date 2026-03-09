# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import math
import cudaq


def test_ptsbe_broadcast_bit_flip_noise(rotation_kernel):
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("ry", cudaq.BitFlipChannel(0.1))
    shots = 500
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
    assert 0.80 <= p0 <= 0.99
    assert 0.01 <= p1_flipped <= 0.20

    p1 = results[1].count("1") / shots
    p0_flipped = results[1].count("0") / shots
    assert 0.80 <= p1 <= 0.99
    assert 0.01 <= p0_flipped <= 0.20


def test_ptsbe_sample_async_returns_future_like(bell_kernel):
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(0.1), num_controls=1)
    future = cudaq.ptsbe.sample_async(bell_kernel,
                                      noise_model=noise,
                                      shots_count=10)
    assert hasattr(future, "get")
    result = future.get()
    assert sum(result.count(bs) for bs in result) == 10


def test_ptsbe_broadcast_single_argument(rotation_kernel):
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


def test_ptsbe_broadcast_three_angles(rotation_kernel):
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


def test_ptsbe_sample_async_get_consumes_future(bell_kernel):
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x",
                                cudaq.Depolarization2(0.05),
                                num_controls=1)
    future = cudaq.ptsbe.sample_async(bell_kernel,
                                      noise_model=noise,
                                      shots_count=15)
    r = future.get()
    assert sum(r.count(bs) for bs in r) == 15


def test_ptsbe_sample_async(bell_kernel):
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x",
                                cudaq.Depolarization2(0.01),
                                num_controls=1)
    shots = 50
    future = cudaq.ptsbe.sample_async(bell_kernel,
                                      noise_model=noise,
                                      shots_count=shots)
    result = future.get()
    assert isinstance(result, cudaq.SampleResult)
    assert sum(result.count(bs) for bs in result) == shots
    bell_counts = result.count("00") + result.count("11")
    assert bell_counts >= shots * 0.8, "Most outcomes should be 00/11"


def test_ptsbe_sample_async_many_no_race(bell_kernel):
    """Fire many async tasks and verify all succeed.

    Regression test for any lifetime race issues we might have with noise models.
    """
    shots = 20
    trials = 50
    futures = []
    for _ in range(trials):
        noise = cudaq.NoiseModel()
        noise.add_all_qubit_channel("x",
                                    cudaq.Depolarization2(0.01),
                                    num_controls=1)
        futures.append(
            cudaq.ptsbe.sample_async(bell_kernel,
                                     noise_model=noise,
                                     shots_count=shots))
    for i, f in enumerate(futures):
        result = f.get()
        total = sum(result.count(bs) for bs in result)
        assert total == shots, f"Trial {i}: expected {shots} shots, got {total}"


def test_ptsbe_broadcast(rotation_kernel):
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("ry", cudaq.DepolarizationChannel(0.01))
    shots = 50
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
        assert sum(r.count(bs) for bs in r) == shots
    assert results[0].count("0") > shots * 0.7
    assert results[1].count("1") > shots * 0.7
