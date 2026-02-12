# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import pytest
import cudaq
from .kernels import bell


def test_ptsbe_sample_returns_sample_result(depol_noise):
    strategy = cudaq.ptsbe.ExhaustiveSamplingStrategy()
    result = cudaq.ptsbe.sample(bell,
                                noise_model=depol_noise,
                                shots_count=100,
                                sampling_strategy=strategy)
    assert isinstance(result, cudaq.SampleResult)
    assert len(result) > 0


def test_ptsbe_sample_custom_shots(depol_noise):
    strategy = cudaq.ptsbe.ExhaustiveSamplingStrategy()
    result = cudaq.ptsbe.sample(bell,
                                noise_model=depol_noise,
                                shots_count=50,
                                sampling_strategy=strategy)
    assert isinstance(result, cudaq.SampleResult)
    assert len(result) > 0


def test_ptsbe_sample_max_trajectories(depol_noise):
    strategy = cudaq.ptsbe.ExhaustiveSamplingStrategy()
    result = cudaq.ptsbe.sample(
        bell,
        noise_model=depol_noise,
        shots_count=100,
        max_trajectories=50,
        sampling_strategy=strategy,
    )
    assert isinstance(result, cudaq.SampleResult)
    assert len(result) > 0


def test_ptsbe_sample_raises_without_noise_model():
    with pytest.raises(RuntimeError, match="requires a noise_model"):
        cudaq.ptsbe.sample(bell)


def test_ptsbe_sample_raises_with_none_noise_model():
    with pytest.raises(RuntimeError, match="requires a noise_model"):
        cudaq.ptsbe.sample(bell, noise_model=None)


def test_ptsbe_sample_rejects_negative_shots(depol_noise):
    with pytest.raises(RuntimeError, match="shots_count"):
        cudaq.ptsbe.sample(bell, noise_model=depol_noise, shots_count=-1)


def test_ptsbe_sample_rejects_wrong_arity(depol_noise):
    with pytest.raises(RuntimeError, match="Invalid number of arguments"):
        cudaq.ptsbe.sample(bell, 42, noise_model=depol_noise)


def test_ptsbe_sample_rejects_zero_max_trajectories(depol_noise):
    with pytest.raises(RuntimeError, match="max_trajectories"):
        cudaq.ptsbe.sample(bell, noise_model=depol_noise, max_trajectories=0)


def test_ptsbe_sample_rejects_negative_max_trajectories(depol_noise):
    with pytest.raises(RuntimeError, match="max_trajectories"):
        cudaq.ptsbe.sample(bell, noise_model=depol_noise, max_trajectories=-5)
