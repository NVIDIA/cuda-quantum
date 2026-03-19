# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import cudaq


def test_ptsbe_sample_result_total_shots_matches_requested(
        depol_noise, bell_kernel):
    result = cudaq.ptsbe.sample(bell_kernel,
                                noise_model=depol_noise,
                                shots_count=100)
    assert sum(result.count(bs) for bs in result) == 100


def test_ptsbe_sample_single_shot(depol_noise, bell_kernel):
    result = cudaq.ptsbe.sample(bell_kernel,
                                noise_model=depol_noise,
                                shots_count=1)
    assert sum(result.count(bs) for bs in result) == 1
    assert len(result) >= 1


def test_ptsbe_sample_large_shots(depol_noise, bell_kernel):
    result = cudaq.ptsbe.sample(bell_kernel,
                                noise_model=depol_noise,
                                shots_count=2000)
    assert sum(result.count(bs) for bs in result) == 2000


def test_ptsbe_sample_returns_sample_result(depol_noise, bell_kernel):
    strategy = cudaq.ptsbe.ExhaustiveSamplingStrategy()
    result = cudaq.ptsbe.sample(
        bell_kernel,
        noise_model=depol_noise,
        shots_count=100,
        sampling_strategy=strategy,
    )
    assert isinstance(result, cudaq.SampleResult)
    assert len(result) > 0


def test_ptsbe_sample_custom_shots(depol_noise, bell_kernel):
    result = cudaq.ptsbe.sample(
        bell_kernel,
        noise_model=depol_noise,
        shots_count=50,
    )
    assert isinstance(result, cudaq.SampleResult)
    assert sum(result.count(bs) for bs in result) == 50


def test_ptsbe_sample_with_apply_noise_in_kernel(kernel_with_apply_noise):
    result = cudaq.ptsbe.sample(
        kernel_with_apply_noise,
        shots_count=100,
    )
    assert sum(result.count(bs) for bs in result) == 100
    assert len(result) >= 1


def test_ptsbe_sequential_data_empty_by_default(depol_noise, bell_kernel):
    result = cudaq.ptsbe.sample(bell_kernel,
                                noise_model=depol_noise,
                                shots_count=10)
    seq = result.get_sequential_data()
    assert len(seq) == 0


def test_ptsbe_sequential_data_populated_when_requested(depol_noise,
                                                        bell_kernel):
    result = cudaq.ptsbe.sample(bell_kernel,
                                noise_model=depol_noise,
                                shots_count=10,
                                include_sequential_data=True)
    seq = result.get_sequential_data()
    assert len(seq) == 10
    for bs in seq:
        assert len(bs) == 2
