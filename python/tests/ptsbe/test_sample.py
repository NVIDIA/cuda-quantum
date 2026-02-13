# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import pytest
import cudaq


@cudaq.kernel
def bell():
    q = cudaq.qvector(2)
    h(q[0])
    x.ctrl(q[0], q[1])
    mz(q)


@pytest.fixture
def bell_kernel():
    return bell


def test_ptsbe_sample_result_total_shots_matches_requested(
        depol_noise, bell_kernel):
    result = cudaq.ptsbe.sample(bell_kernel,
                                noise_model=depol_noise,
                                shots_count=100)
    total = sum(result.count(bs) for bs in result)
    assert total == 100


def test_ptsbe_sample_default_no_execution_data(depol_noise, bell_kernel):
    result = cudaq.ptsbe.sample(bell_kernel,
                                noise_model=depol_noise,
                                shots_count=10)
    assert not result.has_execution_data()


def test_ptsbe_sample_single_shot(depol_noise, bell_kernel):
    result = cudaq.ptsbe.sample(bell_kernel,
                                noise_model=depol_noise,
                                shots_count=1)
    total = sum(result.count(bs) for bs in result)
    assert total == 1
    assert len(result) >= 1


def test_ptsbe_sample_large_shots(depol_noise, bell_kernel):
    result = cudaq.ptsbe.sample(bell_kernel,
                                noise_model=depol_noise,
                                shots_count=2000)
    total = sum(result.count(bs) for bs in result)
    assert total == 2000


def test_ptsbe_sample_with_return_execution_data_true(depol_noise, bell_kernel):
    result = cudaq.ptsbe.sample(
        bell_kernel,
        noise_model=depol_noise,
        shots_count=25,
        return_execution_data=True,
    )
    assert result.has_execution_data()
    assert result.ptsbe_execution_data is not None


def test_ptsbe_sample_max_trajectories_one(depol_noise, bell_kernel):
    strategy = cudaq.ptsbe.ExhaustiveSamplingStrategy()
    result = cudaq.ptsbe.sample(
        bell_kernel,
        noise_model=depol_noise,
        shots_count=20,
        max_trajectories=1,
        sampling_strategy=strategy,
    )
    total = sum(result.count(bs) for bs in result)
    assert total == 20
