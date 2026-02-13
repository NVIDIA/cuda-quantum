# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import pytest
import cudaq

from test_common import (
    bell,
    make_depol_noise,
    ptsbe_target_setup,
    ptsbe_target_teardown,
)


@pytest.fixture(autouse=True)
def ptsbe_target():
    ptsbe_target_setup()
    yield
    ptsbe_target_teardown()


@pytest.fixture
def depol_noise():
    return make_depol_noise()


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
    strategy = cudaq.ptsbe.ExhaustiveSamplingStrategy()
    result = cudaq.ptsbe.sample(
        bell_kernel,
        noise_model=depol_noise,
        shots_count=50,
        sampling_strategy=strategy,
    )
    assert isinstance(result, cudaq.SampleResult)
    assert sum(result.count(bs) for bs in result) == 50


def test_ptsbe_sample_raises_without_noise_model(bell_kernel):
    with pytest.raises(RuntimeError, match="requires a noise_model"):
        cudaq.ptsbe.sample(bell_kernel)


def test_ptsbe_sample_raises_with_none_noise_model(bell_kernel):
    with pytest.raises(RuntimeError, match="requires a noise_model"):
        cudaq.ptsbe.sample(bell_kernel, noise_model=None)


def test_ptsbe_sample_rejects_negative_shots(depol_noise, bell_kernel):
    with pytest.raises(RuntimeError, match="shots_count"):
        cudaq.ptsbe.sample(bell_kernel, noise_model=depol_noise, shots_count=-1)


def test_ptsbe_sample_rejects_wrong_arity(depol_noise, bell_kernel):
    with pytest.raises(RuntimeError, match="Invalid number of arguments"):
        cudaq.ptsbe.sample(bell_kernel, 42, noise_model=depol_noise)


def test_ptsbe_sample_rejects_zero_max_trajectories(depol_noise, bell_kernel):
    with pytest.raises(RuntimeError, match="max_trajectories"):
        cudaq.ptsbe.sample(
            bell_kernel,
            noise_model=depol_noise,
            max_trajectories=0,
        )


def test_ptsbe_sample_rejects_negative_max_trajectories(depol_noise,
                                                        bell_kernel):
    with pytest.raises(RuntimeError, match="max_trajectories"):
        cudaq.ptsbe.sample(
            bell_kernel,
            noise_model=depol_noise,
            max_trajectories=-5,
        )
